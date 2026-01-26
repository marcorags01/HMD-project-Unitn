# main.py
"""
Meal Kit Composer — main controller loop.

Pipeline:
  user_text
    -> NLU (LLM) produces MR JSON
    -> Tracker.ingest_turn() applies MR(s) to state and queues pending MRs
    -> Tracker.select_next_mr() selects ONE MR to address now
    -> DM (LLM) proposes next action
    -> policy.apply_policy() enforces hard rules deterministically
    -> executor runs domain functions (support_fn) and updates Tracker
    -> NLG (templates) produces the user response
    
"""

from __future__ import annotations
import os
import logging
from typing import Any, Dict
import copy

# Hide HF advisory warnings like "generation flags are not valid..."
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
except Exception:
    pass

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
    avoid_hits as compute_avoid_hits,
)
from support_classes import (
    History, Tracker, normalize_day
)
from dm import DM
from policy import apply_policy
from intents_schema import validate_mr


logger = logging.getLogger("MealKitComposer")
logger.setLevel(logging.DEBUG)


# ------------------------- Continue-deferred gate (pre-NLU) -------------------------

YES_SET = {"yes", "y", "ok", "okay", "sure", "continue", "go ahead", "please"}
NO_SET  = {"no", "n", "nope", "not now", "cancel", "stop"}

# Actions that should NOT clear awaiting_slot (read-only / informational “interrupts”)
INTERRUPT_ACTIONS = {"show_day", "show_week", "provide_info"}

def parse_continue_reply(text: str) -> str:
    t = (text or "").strip().lower()
    # normalize repeated spaces
    t = " ".join(t.split())
    if t in YES_SET or t.startswith("yes ") or t.startswith("yes,"):
        return "YES"
    if t in NO_SET or t.startswith("no ") or t.startswith("no,"):
        return "NO"
    return "OTHER"


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

    # --- Menu generation / selection ---
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

    # --- Inspection actions ---
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

            week_rows = []
            for day in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
                rid = (tracker.active_menu or {}).get(day)
                if rid is None:
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

                # NEW: includes controlled-tag hits AND free-text keyword hits (title + ingredients)
                hits = compute_avoid_hits(recipe, avoid_items)

                week_rows.append({
                    "day": day,
                    "title": recipe.get("title", rid_str),
                    "time_min": recipe.get("time_min", None),
                    "calorie_level": recipe.get("calorie_level", None),
                    "avoid_hits": hits,
                })

            payload["week_overview"] = week_rows
            payload["constraints"] = dict(tracker.constraints or {})
            return payload


        except Exception as e:
            payload["error"] = f"I couldn't show the weekly plan: {e}"
            return payload

     # --- Refine actions ---   
    if action == "suggest_swap_day":
        if not tracker.has_active_menu():
            payload["error"] = "Please pick a menu option first (1 or 2)."
            return payload

        # Canonicalize day (defensive)
        day_raw = (argument or "").strip()
        day = normalize_day(day_raw) or day_raw
        day = day.strip()
        if not day:
            payload["error"] = "I didn’t catch which day you mean."
            return payload

        try:
            # if the user is asking again for the same day while a suggestion is pending,
            # treat that as an implicit rejection of the pending suggestion 
            pending = getattr(tracker, "pending_action", None)
            if isinstance(pending, dict):
                p_type = str(pending.get("type", "") or "").strip().upper()
                p_day = str(pending.get("day", "") or "").strip()
                if p_type == "SWAP_DAY" and p_day == day:
                    # record rejection + clear pending
                    if hasattr(tracker, "record_rejection_from_pending"):
                        tracker.record_rejection_from_pending(pending)
                    tracker.pending_action = None

            denied = getattr(tracker, "last_denied_action", None)
            if isinstance(denied, dict):
                if (
                    str(denied.get("type", "")).strip().upper() == "SWAP_DAY"
                    and str(denied.get("day", "")).strip() == day
                ):
                    denied_id = str(denied.get("recipe_id", "")).strip()
                    if denied_id:
                        payload["denied_recipe_id"] = denied_id
                        payload["denied_title"] = recipes_by_id.get(denied_id, {}).get("title", denied_id)
            # build exclusions from previously rejected suggestions 
            exclude_ids = set()
            rs = getattr(tracker, "rejected_suggestions", None)
            if isinstance(rs, dict):
                lst = rs.get(day)
                if isinstance(lst, list):
                    exclude_ids.update(str(x) for x in lst if x is not None)

            sug_id = suggest_swap_day_in_menu(
                tracker.active_menu or {},
                day,
                recipes,
                tracker.constraints,
                exclude_ids=exclude_ids,   
            )

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

                # Safety checks 
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

        # Snapshot old menu to compute substitutions
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

            # Add detailed substitutions day -> new recipe (and old recipe)
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

    # --- Confirmation ---
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

    # --- Help/info ---
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

# ------------------------- Dialogue wrapper  -------------------------

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
        

    def _say(self, text: str) -> None:
        print(f"Assistant: {text}\n")

    def _read_user(self) -> str:
        return input("User(you): ").strip()

    def start(self):
        print()
        starting = PROMPTS.get("START", "Hi. How can I help you?")
        self._say(starting)
        self.history.add_msg(starting, "assistant", "start")
        last_action = ""
        DEBUG = bool(getattr(self.args, "debug", False))

        while True:
            user_text = self._read_user()
            raw = (user_text or "").strip()
            low = raw.lower()
            # Deterministic restart/reset escape hatch
            if low in {"restart", "reset", "start over"}:
                self.tracker.clear()
                self.history.clear()
                last_action = ""

                starting = PROMPTS.get("START", "Hi. How can I help you?")
                self._say(starting)
                self.history.add_msg(starting, "assistant", "start")
                continue

            if low in {"exit", "quit", "q"}:
                self._say("Goodbye.")
                break

            if getattr(self.tracker, "phase", "") == "CONFIRMED" and low in {
                "finalize", "done", "thanks", "thank you", "bye"
            }:
                self._say("All set. Goodbye.")
                break

            if not raw:
                if getattr(self.tracker, "phase", "") == "CONFIRMED":
                    self._say("All set. Goodbye.")
                    break
                continue

            self.history.add_msg(user_text, "user", "input")

            # -------------------- EARLY CONTINUE_DEFERRED GATE (before NLU) --------------------
            bypass_raw_mrs = None  # if set, we skip NLU and treat these as the "raw_mrs" for this turn

            pending = getattr(self.tracker, "pending_action", None)
            if isinstance(pending, dict) and (pending.get("type") == "CONTINUE_DEFERRED"):
                decision = parse_continue_reply(user_text)

                next_summary = str(pending.get("next") or "the next step")

                if decision == "YES":
                    next_mr = self.tracker.pop_deferred()
                    # clear the gate
                    self.tracker.pending_action = None

                    self.tracker.pending_mrs.clear()

                    if not next_mr:
                        reply = "There is nothing else queued to continue."
                        self._say(reply)
                        self.history.add_msg(reply, "assistant", "fallback")
                        last_action = "fallback"
                        continue

                    # re-stamp to avoid prune_pending_by_turn dropping it as stale
                    next_mr = copy.deepcopy(next_mr)
                    next_mr.pop("_turn_id", None)


                    # bypass NLU: feed this MR into the usual validation/ingest pipeline below
                    bypass_raw_mrs = [next_mr]

                elif decision == "NO":
                    self.tracker.clear_deferred()
                    self.tracker.pending_action = None

                    reply = "Okay — I won’t run the queued follow-up."
                    self._say(reply)
                    self.history.add_msg(reply, "assistant", "fallback")
                    last_action = "fallback"
                    continue  # do not call NLU/DM

                else:  # OTHER
                    reply = f"Please reply yes/no first — do you want to continue with {next_summary}? (yes/no)"
                    self._say(reply)
                    self.history.add_msg(reply, "assistant", "fallback")
                    last_action = "fallback"
                    continue  # do not call NLU/DM
            
            # -------------------- Pipeline: NLU -> ingest -> DM -> policy -> execute -> NLG --------------------

            # 1) NLU -> MR (dict or list[dict])
            if bypass_raw_mrs is not None:
                raw_obj = bypass_raw_mrs
            else:
                raw_obj = self.nlu(user_text, awaiting_slot=getattr(self.tracker, "awaiting_slot", None))


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

            else:
                if action in INTERRUPT_ACTIONS:
                    pass  # keep awaiting_slot so the user can resume answering the pending question
                elif action != "fallback":
                    # Any state-changing action means we're no longer waiting for the previous slot
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

            self._say(reply)
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

    # Default dataset path 
    recipes_path = "recipes_30.json"

    dg = Dialogue(model, tokenizer, args, logger, recipes_path)
    dg.start()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")
