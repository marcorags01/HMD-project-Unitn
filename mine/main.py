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
    
"""

from __future__ import annotations


import logging
from typing import Any, Dict
import copy
from nlg import NLG
from nlu import NLU
from utils import PROMPTS, get_args, load_model
from support_fn import (
    load_recipes,
    generate_two_menus,
    get_day_details,
    suggest_swap_day_in_menu,
    swap_day_in_menu,
    repair_menu,
    update_avoid_items,
    generate_shopping_list,
    is_feasible,
)
from support_classes import (
    History, Tracker,
)
from dm import DM
from policy import apply_policy

from intents_schema import validate_mr






logger = logging.getLogger("MealKitComposer")
logger.setLevel(logging.DEBUG)





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
        
    if action == "show_week":
        if not tracker.has_active_menu():
            payload["error"] = "Please pick a menu option first (1 or 2)."
            return payload

        try:
            avoid_items = tracker.constraints.get("avoid_items") or []
            avoid_set = {str(x).strip().lower() for x in avoid_items if str(x).strip()}

            week_rows = []
            for day in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
                rid = (tracker.active_menu or {}).get(day)
                if rid is None:
                    # Defensive: keep structure stable even if menu is partial
                    week_rows.append({
                        "day": day,
                        "title": "(no meal set)",
                        "time_min": None,
                        "calorie_level": None,
                        "avoid_hits": [],
                    })
                    continue

                rid_str = str(rid)
                recipe = recipes_by_id.get(rid_str, {})

                contains = recipe.get("contains_tags") or []
                contains_set = {str(t).strip().lower() for t in contains if str(t).strip()}

                avoid_hits = sorted(list(avoid_set.intersection(contains_set)))

                week_rows.append({
                    "day": day,
                    "title": recipe.get("title", rid_str),
                    "time_min": recipe.get("time_min", None),
                    "calorie_level": recipe.get("calorie_level", None),
                    "avoid_hits": avoid_hits,
                })

            payload["week_overview"] = week_rows
            payload["constraints"] = dict(tracker.constraints or {})
            return payload

        except Exception as e:
            payload["error"] = f"I couldn't show the weekly plan: {e}"
            return payload

        
    if action == "suggest_swap_day":
        if not tracker.has_active_menu():
            payload["error"] = "Please pick a menu option first (1 or 2)."
            return payload

        day = argument.strip()
        try:
            sug_id = suggest_swap_day_in_menu(tracker.active_menu or {}, day, recipes, tracker.constraints)
            if sug_id is None:
                payload["suggested"] = False
                return payload

            # Store pending suggestion (no menu mutation)
            tracker.set_pending_swap(day, str(sug_id))

            payload["suggested"] = True
            payload["suggested_day"] = day
            payload["suggested_recipe_id"] = str(sug_id)
            payload["suggested_title"] = (
                recipes_by_id[str(sug_id)]["title"] if str(sug_id) in recipes_by_id else str(sug_id)
            )
            return payload
        except Exception as e:
            payload["error"] = f"I couldn't suggest an alternative for that day: {e}"
            return payload


    if action == "swap_day":
        if not tracker.has_active_menu():
            payload["error"] = "Please pick a menu option first (1 or 2)."
            return payload

        day = argument.strip()

        # ---- 1) If a matching pending suggestion exists, commit it (no recomputation) ----
        pending = getattr(tracker, "pending_action", None)
        if pending:
            p_type = str((pending or {}).get("type") or "").strip().upper()
            p_day = (pending or {}).get("day")
            p_rid = (pending or {}).get("recipe_id")

            if p_type == "SWAP_DAY" and p_day == day and p_rid:
                rid = str(p_rid)

                # Safety checks (no new suggestion computation):
                recipe = recipes_by_id.get(rid)
                if recipe is None:
                    tracker.pending_action = None
                    payload["error"] = "That suggested recipe is no longer available. Please ask for another alternative."
                    return payload

                # Ensure still feasible under current constraints
                if not is_feasible(recipe, tracker.constraints):
                    tracker.pending_action = None
                    payload["error"] = "That suggested meal no longer matches your preferences. Please ask for another alternative."
                    return payload

                # Commit pending swap deterministically
                updated = dict(tracker.active_menu or {})
                updated[day] = rid
                tracker.active_menu = updated
                if tracker.active_menu_id in (1, 2):
                    tracker.menus[str(tracker.active_menu_id)] = dict(updated)

                tracker.pending_action = None

                payload["swapped"] = True
                payload["new_recipe_id"] = rid
                payload["new_title"] = recipe["title"]
                payload["committed_from_pending"] = True
                return payload

        # ---- 2) Otherwise perform a normal swap (BEST_FIT) ----
        try:
            updated, new_id = swap_day_in_menu(tracker.active_menu or {}, day, recipes, tracker.constraints)
            if new_id is None:
                payload["swapped"] = False
                return payload

            tracker.active_menu = updated
            if tracker.active_menu_id in (1, 2):
                tracker.menus[str(tracker.active_menu_id)] = dict(updated)

            # Any explicit swap invalidates prior pending suggestion
            tracker.pending_action = None

            payload["swapped"] = True
            payload["new_recipe_id"] = new_id
            payload["new_title"] = recipes_by_id[str(new_id)]["title"] if str(new_id) in recipes_by_id else str(new_id)
            payload["committed_from_pending"] = False
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

        # NEW: snapshot old menu to compute substitutions
        old_menu = dict(tracker.active_menu or {})

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

            # NEW: add detailed substitutions day -> new recipe (and old recipe)
            repairs = []
            for day in repaired_days:
                old_id = str(old_menu.get(day, ""))
                new_id = str(repaired_menu.get(day, ""))

                if not new_id or old_id == new_id:
                    continue

                repairs.append({
                    "day": day,
                    "old_recipe_id": old_id,
                    "old_title": recipes_by_id.get(old_id, {}).get("title", old_id),
                    "new_recipe_id": new_id,
                    "new_title": recipes_by_id.get(new_id, {}).get("title", new_id),
                })

            payload["repairs"] = repairs
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

        # Default
        intent_req = "plan"
        slot_req = "all"

        if raw:
            parts = [p.strip() for p in raw.split(",", 1) if p.strip()]
            if len(parts) == 1:
                intent_req, slot_req = parts[0], "all"
            else:
                intent_req, slot_req = parts[0], parts[1]

        payload["help_intent"] = (intent_req or "plan").strip().lower()
        payload["help_slot"] = (slot_req or "all").strip().lower()
        payload["phase"] = getattr(tracker, "phase", "") or ""
        payload["has_active_menu"] = getattr(tracker, "has_active_menu", lambda: False)()

        return payload
    # request_info / fallback / unknown: no execution-side effects
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
        DEBUG = bool(getattr(self.args, "debug", False))


        while True:
            user_text = input().strip()
            raw = (user_text or "").strip()
            low = raw.lower()
            # Deterministic restart/reset escape hatch
            if low in {"restart", "reset", "start over"}:
                self.tracker.clear()
                self.history.clear()
                last_action = ""

                starting = PROMPTS.get("START", "Hi. How can I help you?")
                print(starting)
                self.history.add_msg(starting, "assistant", "start")
                continue


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

            # 1) NLU -> MR (dict or list[dict])
            raw_obj = self.nlu(user_text)

            # Freeze contract: downstream always sees list[dict]
            if raw_obj is None:
                raw_mrs: list[dict] = []
            elif isinstance(raw_obj, dict):
                raw_mrs = [raw_obj]
            elif isinstance(raw_obj, list):
                raw_mrs = [x for x in raw_obj if isinstance(x, dict)]
            else:
                raw_mrs = []

            if not raw_mrs:
                raw_mrs = [{"intent": "out_of_domain", "slots": {}}]


            # 2) Validate + normalize each MR (debug/robustness)
            validations = [validate_mr(m) for m in raw_mrs]
            mrs: list[dict] = []
            for v in validations:
                if v.valid:
                    mrs.append(v.normalized_mr)
                else:
                    mrs.append({"intent": "out_of_domain", "slots": {"ood_type": "INVALID_MR"}})

            if DEBUG:
                print("DEBUG raw MRs:", raw_mrs)
                for i, v in enumerate(validations):
                    print(f"DEBUG MR[{i}] valid:", v.valid, "errors:", v.errors)
                    print(f"DEBUG MR[{i}] normalized:", v.normalized_mr)
                print("DEBUG MRs ingested:", mrs)

            
            # 3) Apply MR(s) to tracker as ONE turn (always)
            self.tracker.ingest_turn(mrs, history=self.history)

            if DEBUG:
                print("DEBUG tracker.constraints:", self.tracker.constraints)
                print("DEBUG missing_plan_slots:", self.tracker.missing_plan_slots())
                print("DEBUG tracker.phase:", self.tracker.phase)
                print("DEBUG pending_mrs:", self.tracker.pending_mrs)

            # 4) Select ONE MR to address now
            selected_mr = self.tracker.select_next_mr()
            selected_mr_snapshot = copy.deepcopy(selected_mr)
                
            if DEBUG:
                print("DEBUG selected_mr:", selected_mr)

            # 5) DM proposes ONE action for the selected MR
            proposed_action, proposed_arg, _ = self.dm(self.tracker, selected_mr, last_action=last_action)
            self.logger.info(f"DM proposed: {proposed_action}({proposed_arg})")
            if DEBUG:
                print("DEBUG DM proposed:", proposed_action, "arg:", proposed_arg)

            # 6) Policy enforces hard rules (still against selected MR)
            action, arg, dbg = apply_policy(self.tracker, selected_mr, proposed_action, proposed_arg)
            # 6b) Track slot-filling context for robust fallback/reprompt behavior
            if action == "request_info":
                self.tracker.note_request_info(arg)
            elif action != "fallback":
                # We are moving on from slot-filling; clear reprompt context.
                self.tracker.clear_awaiting()

            if DEBUG:
                print("DEBUG Policy final:", action, "arg:", arg, "reason:", dbg.get("policy_reason"))
            self.logger.info(f"Policy final: {action}({arg}) | reason={dbg.get('policy_reason')}")

            # 7) Execute
            payload = execute_action(action, arg, self.tracker, self.recipes, self.recipes_by_id)

            # 8) Mutate tracker based on executed action
            self.tracker.update_pending_after_action(selected_mr_snapshot, action, payload)

            

            # 9) NLG
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
