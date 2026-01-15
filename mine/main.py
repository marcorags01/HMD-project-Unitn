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


import logging
from typing import Any, Dict
from nlg import NLG
from nlu import NLU
from utils import PROMPTS, get_args, load_model
from support_fn import (
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
)
from dm import DM
from policy import apply_policy

from intents_schema import validate_mr



logger = logging.getLogger("MealKitComposer")
logger.setLevel(logging.DEBUG)





def _provide_info_message(tracker: Tracker, intent: str, slot: str) -> str:
    intent = (intent or "").strip().lower()
    slot = (slot or "").strip().lower()
    phase = getattr(tracker, "phase", "") or ""
    has_active = getattr(tracker, "has_active_menu", lambda: False)()

    if slot in {"all", ""}:
        if phase == "AWAITING_PLAN":
            return (
                "I can help you plan weekday dinners (Mon–Fri) and generate a shopping list.\n"
                "To start: how many servings should I plan for?"
            )

        if phase == "AWAITING_MENU_SELECTION":
            return (
                "I can suggest two weekly menu options and you can pick the one you prefer.\n"
                "If you tell me how many people you’re cooking for, whether you want quick meals, and anything to avoid, "
                "I’ll generate options."
            )

        if has_active or phase == "ACTIVE_MENU":
            return (
                "For your current plan, you can:\n"
                "- ask what’s planned on a day (e.g., “What’s on Tue?”)\n"
                "- swap a day (e.g., “Swap Wed”)\n"
                "- add/remove foods to avoid (e.g., “Avoid nuts”)\n"
                "- confirm to get the shopping list\n"
                "What would you like to do next?"
            )

        if phase == "CONFIRMED":
            return (
                "Your plan is already confirmed and the shopping list is ready.\n"
                "You can start a new plan (e.g., “Plan meals for 2 people, quick, balanced”), or type “exit”."
            )

        # Safe default
        return "How can I help with your meal plan?"

    # Slot-specific help (keep it conversational)
    if slot == "servings":
        return "How many servings should I plan for? (1–6)"
    if slot == "time_limit":
        return "Do you want quick meals, or is normal prep time OK?"
    if slot == "calorie_level":
        return "Are you aiming for lighter meals, balanced, or more filling?"
    if slot == "avoid_items":
        return "Any allergies or foods you want to avoid?"
    if slot == "menu_id":
        return "Do you prefer option 1 or option 2?"
    if slot == "target_day":
        return "Which day should I focus on—Mon, Tue, Wed, Thu, or Fri?"

    # Fallback
    return "What would you like to do next?"


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
            friendly = {
                "servings": "how many servings you need",
                "time_limit": "whether you prefer quick meals or normal prep time",
                "calorie_level": "whether you want lighter, balanced, or more filling meals",
                "avoid_items": "any foods you want to avoid",
            }
            need = [friendly.get(x, "one more detail") for x in (missing or [])]
            payload["error"] = "Before I can suggest menus, I still need " + "; ".join(need) + "."
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
            payload["error"] = str(e) or (
                "I couldn’t generate a weekly plan with those preferences. "
                "If you adjust them a bit, I can try again."
            )
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
            payload["error"] = "Please pick a menu option first (1 or 2)."
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
            payload["error"] = f"I couldn't show that day: {e}"
            return payload

    if action == "swap_day":
        if not tracker.has_active_menu():
            payload["error"] = "Please pick a menu option first (1 or 2)."
            return payload

        day = argument.strip()
        try:
            updated, new_id = swap_day_in_menu(tracker.active_menu or {}, day, recipes, tracker.constraints)
            if new_id is None:
                payload["swapped"] = False
                return payload

            tracker.active_menu = updated
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
            payload["error"] = "Please pick a menu option first (1 or 2)."
            return payload

        parts = [p.strip() for p in argument.split(",") if p.strip()]
        if len(parts) != 2:
            payload["error"] = "I didn’t catch that—tell me what to avoid (e.g., “avoid nuts”)."
            return payload

        op, val = parts[0], parts[1]
        ok, err = update_avoid_items(tracker.constraints, op, val)
        if not ok:
            payload["error"] = err or "I couldn’t update that avoid item."
            return payload

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
            payload["error"] = f"I updated your avoid list, but I couldn’t repair the plan: {e}"
            return payload

    if action == "confirm_plan":
        if not tracker.has_active_menu():
            payload["error"] = "Please pick a menu option first (1 or 2)."
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
        raw = (argument or "").strip()

        if not raw:
            payload["message"] = _provide_info_message(tracker, intent="plan", slot="all")
            return payload

        parts = [p.strip() for p in raw.split(",", 1) if p.strip()]
        if len(parts) == 1:
            intent_req, slot_req = parts[0], "all"
        else:
            intent_req, slot_req = parts[0], parts[1]

        payload["message"] = _provide_info_message(tracker, intent_req.strip().lower(), slot_req.strip().lower())
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
            raw = (user_text or "").strip()
            low = raw.lower()

            if low in {"exit", "quit", "q"}:
                print("Goodbye.")
                break

            if getattr(self.tracker, "phase", "") == "CONFIRMED" and low in {
                "finalize", "done", "thanks", "thank you", "bye"
            }:
                print("All set. Goodbye.")
                break

            if not raw:
                continue

            self.history.add_msg(user_text, "user", "input")

            # 1) NLU -> MR
            raw_mr = self.nlu(user_text)
            vr = validate_mr(raw_mr)
            mr = vr.normalized_mr

            mr = vr.normalized_mr  # use normalized for tracker/DM/policy
            print("DEBUG raw MR:", raw_mr) # debug print
            print("DEBUG mr_valid:", vr.valid, "errors:", vr.errors)
            print("DEBUG normalized MR:", mr)

            

            # 2) Apply MR to tracker
            intent, _, _ = self.tracker.creation(mr, self.history, update=True)
            print("DEBUG tracker.constraints:", self.tracker.constraints)
            print("DEBUG missing_plan_slots:", self.tracker.missing_plan_slots())
            print("DEBUG tracker.phase:", self.tracker.phase)
            self.logger.info(f"Intent: {intent}")

            # 3) DM proposes action
            proposed_action, proposed_arg, _ = self.dm(self.tracker, mr, last_action=last_action)
            self.logger.info(f"DM proposed: {proposed_action}({proposed_arg})")

            # 4) Policy enforces hard rules
            action, arg, dbg = apply_policy(self.tracker, mr, proposed_action, proposed_arg)
            print("DEBUG DM proposed:", proposed_action, "arg:", proposed_arg)
            print("DEBUG Policy final:", action, "arg:", arg, "reason:", dbg.get("policy_reason"))

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
