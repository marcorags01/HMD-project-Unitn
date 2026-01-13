# main.py
"""
Meal Kit Composer — main controller loop (Marina-like).

Pipeline:
  user_text
    -> NLU (LLM) produces MR JSON
    -> Tracker.creation() applies MR to state
    -> DM (LLM) proposes next action
    -> policy.apply_policy() enforces hard rules deterministically
    -> executor runs domain functions (support_fn) and updates Tracker
    -> NLG (templates) produces the user response

Notes:
- NLU and NLG are kept inside this file for now to avoid importing not-yet-existing modules.
  You can later move them to nlu.py / nlg.py without changing the overall architecture.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict
from nlg import NLG
from nlu import NLU
from utils import PROMPTS, get_args, load_model, generate
from support_fn import (
    parsing_json,
    load_recipes,
    can_generate_menus,
    generate_two_menus,
    get_day_details,
    swap_day_in_menu,
    repair_menu,
    update_avoid_items,
    generate_shopping_list,
)
from support_classes import (
    History, Tracker,
     ALLOWED_AVOID_ITEMS
)
from dm import DM
from policy import apply_policy, REQUESTABLE_SLOTS


logger = logging.getLogger("MealKitComposer")
logger.setLevel(logging.DEBUG)





def _provide_info_message(tracker: Tracker, intent: str, slot: str) -> str:
    intent = (intent or "").strip().lower()
    slot = (slot or "").strip().lower()

    if slot == "all":
        return (
            "PLAN slots and allowed values:\n"
            "- servings: 1–6\n"
            "- time_limit: FAST or NORMAL\n"
            "- calorie_level: LOW, MED, HIGH\n"
            "- avoid_items: none, or: " + ", ".join(sorted(ALLOWED_AVOID_ITEMS)) + "\n\n"
            "Other slots:\n"
            "- menu_id: 1 or 2\n"
            "- target_day: Mon–Fri\n"
            "- refine_type: SWAP_DAY / ADD_AVOID_ITEM / REMOVE_AVOID_ITEM\n"
            "- value: (SWAP_DAY) BEST_FIT; (avoid ops) one avoid item"
        )

    if slot == "servings":
        return "servings: integer 1–6."
    if slot == "time_limit":
        return "time_limit: FAST (≤25 min) or NORMAL (≤40 min)."
    if slot == "calorie_level":
        return "calorie_level: LOW, MED, or HIGH."
    if slot == "avoid_items":
        return "avoid_items: none, or a comma-separated list from: " + ", ".join(sorted(ALLOWED_AVOID_ITEMS)) + "."
    if slot == "menu_id":
        return "menu_id: 1 or 2."
    if slot == "target_day":
        return "target_day: Mon, Tue, Wed, Thu, Fri."
    if slot == "refine_type":
        return "refine_type: SWAP_DAY, ADD_AVOID_ITEM, REMOVE_AVOID_ITEM."
    if slot == "value":
        return "value: BEST_FIT for SWAP_DAY; or one avoid item for ADD/REMOVE."
    return "I can provide allowed values for: " + ", ".join(sorted(REQUESTABLE_SLOTS))


# ------------------------- Executor (domain calls + tracker updates) -------------------------

def execute_action(
    action: str,
    argument: str,
    tracker: Tracker,
    recipes: list[Dict[str, Any]],
    recipes_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Execute the final (policy-approved) action and mutate the tracker accordingly.
    Returns a payload dict for NLG.
    """
    payload: Dict[str, Any] = {}

    if action == "propose_menus":
        ok, missing = can_generate_menus(tracker)
        if not ok:
            payload["error"] = f"Before I can generate menus, I still need: {', '.join(missing)}."
            return payload

        try:
            menu1, menu2 = generate_two_menus(recipes, tracker.constraints)
            tracker.set_menus(menu1, menu2)

            payload["menu1"] = menu1
            payload["menu2"] = menu2
            payload["menu1_pretty"] = {d: recipes_by_id[str(rid)]["title"] for d, rid in menu1.items()}
            payload["menu2_pretty"] = {d: recipes_by_id[str(rid)]["title"] for d, rid in menu2.items()}
            return payload
        except Exception as e:
            payload["error"] = f"I couldn't generate menus with the current constraints: {e}"
            return payload

    if action == "set_active_menu":
        try:
            mid = int(argument)
        except Exception:
            payload["ok"] = False
            return payload

        ok = tracker.set_active_menu(mid)
        payload["ok"] = ok
        return payload

    if action == "show_day":
        if not tracker.has_active_menu():
            payload["error"] = "No active menu yet. Please select menu 1 or 2 first."
            return payload

        day = argument.strip()
        try:
            details = get_day_details(
                day=day,
                menu=tracker.active_menu or {},
                recipes_by_id=recipes_by_id,
                servings=int(tracker.constraints["servings"]),
                avoid_items=tracker.constraints.get("avoid_items") or [],
            )
            payload["details"] = details
            return payload
        except Exception as e:
            payload["error"] = f"I couldn't inspect that day: {e}"
            return payload

    if action == "swap_day":
        if not tracker.has_active_menu():
            payload["error"] = "No active menu yet. Please select menu 1 or 2 first."
            return payload

        day = argument.strip()
        try:
            updated, new_id = swap_day_in_menu(tracker.active_menu or {}, day, recipes, tracker.constraints)
            if new_id is None:
                payload["swapped"] = False
                return payload

            tracker.active_menu = updated
            # persist back into stored menu option as well
            if tracker.active_menu_id in (1, 2):
                tracker.menus[str(tracker.active_menu_id)] = dict(updated)

            payload["swapped"] = True
            payload["new_recipe_id"] = new_id
            payload["new_title"] = recipes_by_id[str(new_id)]["title"] if str(new_id) in recipes_by_id else str(new_id)
            return payload
        except Exception as e:
            payload["error"] = f"I couldn't swap that day: {e}"
            return payload

    if action == "update_avoid":
        if not tracker.has_active_menu():
            payload["error"] = "No active menu yet. Please select menu 1 or 2 first."
            return payload

        # argument: "OP, value"
        parts = [p.strip() for p in argument.split(",") if p.strip()]
        if len(parts) != 2:
            payload["error"] = "Invalid avoid update arguments."
            return payload

        op, val = parts[0], parts[1]
        ok, err = update_avoid_items(tracker.constraints, op, val)
        if not ok:
            payload["error"] = err or "Avoid update failed."
            return payload

        # repair menu to satisfy new constraints
        try:
            repaired_menu, repaired_days = repair_menu(
                tracker.active_menu or {},
                recipes_by_id=recipes_by_id,
                recipes=recipes,
                constraints=tracker.constraints,
            )
            tracker.active_menu = repaired_menu
            if tracker.active_menu_id in (1, 2):
                tracker.menus[str(tracker.active_menu_id)] = dict(repaired_menu)
            payload["repaired_days"] = repaired_days
            return payload
        except Exception as e:
            payload["error"] = f"Avoid list updated, but repair failed: {e}"
            return payload

    if action == "confirm_plan":
        if not tracker.has_active_menu():
            payload["error"] = "No active menu yet. Please select menu 1 or 2 first."
            return payload

        try:
            sl = generate_shopping_list(
                tracker.active_menu or {},
                recipes_by_id=recipes_by_id,
                servings=int(tracker.constraints["servings"]),
            )
            tracker.set_phase("CONFIRMED")
            payload["shopping_list"] = sl
            return payload
        except Exception as e:
            payload["error"] = f"I couldn't generate the shopping list: {e}"
            return payload

    if action == "provide_info":
        # argument expected: "intent, slot" (e.g., "plan, all" or "plan, avoid_items")
        raw = (argument or "").strip()

        if not raw:
            payload["message"] = (
                "Please specify what you want values for, e.g., provide_info(plan, all) "
                "or provide_info(plan, time_limit)."
            )
            return payload

        # Split only once to avoid accidental extra commas breaking parsing
        parts = [p.strip() for p in raw.split(",", 1) if p.strip()]

        if len(parts) == 1:
            intent_req, slot_req = parts[0], "all"   # default to all
        else:
            intent_req, slot_req = parts[0], parts[1]

        intent_req = intent_req.strip().lower()
        slot_req = slot_req.strip().lower()

        payload["message"] = _provide_info_message(tracker, intent_req, slot_req)
        return payload


    # request_info / fallback: no execution
    return payload


# ------------------------- Dialogue wrapper (Marina-like) -------------------------

class Dialogue:
    def __init__(self, model, tokenizer, args, logger, recipes_path: str):
        self.tracker = Tracker()
        self.history = History()

        self.nlu = NLU(self.history, model, tokenizer, args, logger)
        self.dm = DM(self.history, model, tokenizer, args, logger)
        self.nlg = NLG(self.history, model, tokenizer, args, logger)

        self.args = args
        self.logger = logger

        self.recipes, self.recipes_by_id = load_recipes(recipes_path)

    def start(self):
        starting = PROMPTS.get("START", "Hi. How can I help you?")
        print(starting)
        self.history.add_msg(starting, "assistant", "start")

        last_action = ""

        while True:
            user_text = input().strip()
            self.history.add_msg(user_text, "user", "input")

            # 1) NLU -> MR
            mr = self.nlu(user_text)

            # 2) Apply MR to tracker
            intent, _, _ = self.tracker.creation(mr, self.history, update=True)
            self.logger.info(f"Intent: {intent}")

            # 3) DM proposes action
            proposed_action, proposed_arg, _ = self.dm(self.tracker, mr, last_action=last_action)
            self.logger.info(f"DM proposed: {proposed_action}({proposed_arg})")

            # 4) Policy enforces hard rules
            action, arg, dbg = apply_policy(self.tracker, mr, proposed_action, proposed_arg)
            self.logger.info(f"Policy final: {action}({arg}) | reason={dbg.get('policy_reason')}")

            # 5) Execute action (domain services) and update tracker
            payload = execute_action(action, arg, self.tracker, self.recipes, self.recipes_by_id)

            # 6) NLG
            reply = self.nlg(
                action=action,
                argument=arg,
                tracker_state=self.tracker.to_state_dict(),
                payload=payload,
            )

            print(reply)
            self.history.add_msg(reply, "assistant", action)

            last_action = action


def main():
    logging.basicConfig(
        filename="meal_kit_composer.log",
        encoding="utf-8",
        filemode="a",
        level=logging.DEBUG,
    )
    logger.info("Starting Meal Kit Composer dialogue")

    args = get_args()
    model, tokenizer = load_model(args)

    # Default dataset path (adjust if needed)
    recipes_path = "recipes_30.json"

    dg = Dialogue(model, tokenizer, args, logger, recipes_path)
    dg.start()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")
