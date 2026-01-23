# support_classes.py
"""
Meal Kit Composer — minimal support classes.

This file intentionally stays small and dependency-light, while still being usable in a
pipeline architecture (NLU -> Tracker/State -> DM -> NLG).

It provides:
- History: lightweight conversation memory for prompting/debugging
- Tracker: the dialogue state holder + a minimal "apply NLU MR to state" updater

State schema (minimal):
{
  "phase": "AWAITING_PLAN" | "AWAITING_MENU_SELECTION" | "ACTIVE_MENU" | "CONFIRMED",
  "constraints": {"servings": int|None, "time_limit": str|None, "calorie_level": str|None, "avoid_items": list[str]},
  "menus": {"1": dict|None, "2": dict|None},
  "active_menu_id": int|None,
  "active_menu": dict|None,
  "last_referenced_day": str|None
}
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Literal
import copy
import re

Phase = Literal["AWAITING_PLAN", "AWAITING_MENU_SELECTION", "ACTIVE_MENU", "CONFIRMED"]

ALLOWED_DAYS = {"Mon", "Tue", "Wed", "Thu", "Fri"}


ALLOWED_TIME_LIMITS = {"FAST", "NORMAL"}
ALLOWED_CALORIE_LEVELS = {"LOW", "MED", "HIGH"}

ALLOWED_AVOID_ITEMS = {
    "nuts", "dairy", "gluten", "soy", "egg", "sesame", "fish", "shellfish", "meat"
}

POSSIBLE_INTENTS = {"plan", "select_menu", "inspect", "show_week", "refine", "confirm", "help", "out_of_domain"}


def is_nullish(x: Any) -> bool:
    """Treat common LLM-returned null variants as null."""
    return x is None or x == "null" or x == "None" or x == ""


def normalize_upper_enum(x: Any) -> Optional[str]:
    if is_nullish(x):
        return None
    if isinstance(x, str):
        return x.strip().upper()
    return str(x).strip().upper()


def normalize_day(x: Any) -> Optional[str]:
    if is_nullish(x):
        return None
    s = str(x).strip().lower()
    if not s:
        return None

    # canonical 3-letter
    s3 = s[:3].title()
    if s3 in ALLOWED_DAYS:
        return s3

    # full names / common variants
    mapping = {
        "monday": "Mon",
        "tuesday": "Tue",
        "wednesday": "Wed",
        "thursday": "Thu",
        "friday": "Fri",
        "tues": "Tue",
        "thur": "Thu",
        "thurs": "Thu",
        "weds": "Wed",
        # common typos you observed
        "wesnday": "Wed",
        "wensday": "Wed",
        "wesnday": "Wed",
    }
    return mapping.get(s)




# --- Avoid-item canonicalization (conservative) ---
# Map common surface forms to the controlled vocabulary tokens.
_AVOID_ALIASES: Dict[str, str] = {
    "eggs": "egg",
    "nut": "nuts",
    "meats": "meat",
    "shell fish": "shellfish", 
    "seafood": "fish",
}

_TRAIL_PUNCT = re.compile(r"[.,;:!?]+$")


def _canon_avoid_token(x: Any) -> str:
    """
    Canonicalize a single token:
    - lower/strip
    - strip trailing punctuation
    - apply a small alias map
    Does NOT invent new items.
    """
    if is_nullish(x):
        return ""
    s = str(x).strip().lower()
    if not s:
        return ""
    s = _TRAIL_PUNCT.sub("", s).strip()
    if not s:
        return ""
    return _AVOID_ALIASES.get(s, s)


_FILLER_PREFIX = re.compile(r"^\s*(avoid|no|without)\b\s*", re.IGNORECASE)

def normalize_avoid_items(x: Any) -> Optional[List[str]]:
    """
    Accept:
    - list[str]
    - str (single, comma-separated, or conjunction-separated)
    - other scalar -> str
    Returns:
    - list[str] (possibly empty) or None if nullish.
    Notes:
    - Conservative: only splits/cleans; does not infer new items.
    """
    if is_nullish(x):
        return None

    # 1) Normalize input into a flat list of text chunks
    chunks: List[str] = []
    if isinstance(x, list):
        chunks = [str(it).strip() for it in x if not is_nullish(it)]
    elif isinstance(x, str):
        chunks = [x.strip()]
    else:
        chunks = [str(x).strip()]

    items: List[str] = []
    seen = set()

    for chunk in chunks:
        if not chunk:
            continue

        # 2) Remove leading filler (only at the start)
        chunk = _FILLER_PREFIX.sub("", chunk).strip()

        if not chunk:
            continue

        # 3) Split on separators and conjunctions.
        #    - commas/semicolons/slashes/&
        #    - the word 'and' as a conjunction
        parts = re.split(r"\s*(?:,|;|/|&|\band\b)\s*", chunk, flags=re.IGNORECASE)

        for p in parts:
            p = _canon_avoid_token(p)
            if not p:
                continue

            # 4) De-dupe while preserving order
            if p not in seen:
                seen.add(p)
                items.append(p)

    return items

# --- Backward-compatible aliases (remove after one commit) ---
_is_nullish = is_nullish
_normalize_upper_enum = normalize_upper_enum
_normalize_day = normalize_day
_normalize_avoid_items = normalize_avoid_items


@dataclass
class History:
    """
    Minimal message history for prompting/debugging.
    Keeps parallel arrays for roles/messages and optional decoded intent labels.
    """
    number_last: int = 8
    messages: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    intents: List[str] = field(default_factory=list)

    last_intent: str = ""
    other_intents: List[str] = field(default_factory=list)

    def update_number_last(self, num: int) -> None:
        self.number_last = max(1, int(num))

    def update_last_intent(self, intent: str) -> None:
        self.last_intent = intent or ""

    def get_last_intent(self) -> str:
        return self.last_intent

    def insert_other_intent(self, intent: str) -> None:
        intent = intent or ""
        if not intent:
            return
        if intent != self.last_intent and intent not in self.other_intents:
            self.other_intents.append(intent)

    def pop_other_intent(self) -> None:
        if self.other_intents:
            self.other_intents = self.other_intents[1:]

    def add_msg(self, msg: str, role: str, intent: str = "") -> None:
        self.roles.append(role)
        self.messages.append(msg)
        self.intents.append(intent or "")

    def clear(self) -> None:
        self.messages.clear()
        self.roles.clear()
        self.intents.clear()
        self.last_intent = ""
        self.other_intents.clear()
        self.number_last = 8

    def to_msg_history(self, last_n: Optional[int] = None) -> List[Dict[str, str]]:
        n = self.number_last if last_n is None else max(1, int(last_n))
        hist = [{"role": r, "content": m} for r, m in zip(self.roles, self.messages)]
        return hist[-n:] if len(hist) > n else hist

    def last_iterations(self, last_n: Optional[int] = None) -> str:
        hist = self.to_msg_history(last_n=last_n)
        return "\n".join([f"{h['role']}: {h['content']}" for h in hist])

    def intent_history(self) -> str:
        return ", \n".join([i for i in self.intents if i])


@dataclass
class Tracker:
    """
    Dialogue-state container + minimal updater from NLU MRs.

    NOTE:
    - This class does NOT generate menus, swap recipes, or build shopping lists.
      Those belong in support_fn.py (DM/domain functions).
    - It only stores constraints, menu proposals, and the active menu pointer.
    """
    phase: Phase = "AWAITING_PLAN"
    constraints: Dict[str, Any] = field(default_factory=lambda: {
        "servings": None,
        "time_limit": None,
        "calorie_level": None,
        "avoid_items": None,
    })
    menus: Dict[str, Optional[Dict[str, Any]]] = field(default_factory=lambda: {"1": None, "2": None})
    active_menu_id: Optional[int] = None
    active_menu: Optional[Dict[str, Any]] = None
    last_referenced_day: Optional[str] = None
    pending_action: Optional[Dict[str, Any]] = None
    last_user_text: str = ""
    last_denied_action: Optional[Dict[str, Any]] = None
    pending_mrs: List[Dict[str, Any]] = field(default_factory=list)
    deferred_mrs: List[Dict[str, Any]] = field(default_factory=list)
    turn_id: int = 0
    awaiting_slot: Optional[str] = None
    reprompt_count: int = 0

    # Optional: keep last MR for debugging/logging
    last_user_mr: Optional[Dict[str, Any]] = None

    possible_intents: List[str] = field(default_factory=lambda: sorted(POSSIBLE_INTENTS))

    # --------- convenience / compatibility helpers  ---------

    def creation(self, input: Dict[str, Any], history: Optional[History] = None, update: bool = True) -> Tuple[str, int, int]:
        """
        Compatibility wrapper:
        - returns (intent, total_slots_for_intent, count_slots_provided_this_turn)

        The DM can ignore these numbers if it prefers.
        """
        intent = str(input.get("intent", "")).strip()
        if not intent:
            intent = "out_of_domain"

        if intent not in POSSIBLE_INTENTS:
            intent = "out_of_domain"

        if history is not None:
            history.update_last_intent(intent)

        total_slots = self._total_slots_for_intent(intent)
        count = 0

        if intent == "out_of_domain":

            # IMPORTANT: still apply MR so we can consume structured OOD subtypes (e.g., REFUSE_PENDING)
            if update:
                count = self.apply_mr(input)
            else:
                self.last_user_mr = {"intent": "out_of_domain", "slots": copy.deepcopy(input.get("slots", {}) or {})}
            return intent, total_slots, count

        if update:
            count = self.apply_mr(input)

        return intent, total_slots, count
    
    def creation_multi(
        self,
        mrs: List[Dict[str, Any]],
        history: Optional[History] = None,
        update: bool = True,
        ) -> Tuple[str, int, int]:
        """
        Multi-MR wrapper:
        - Applies a list of MRs as one user turn.
        - Returns (last_intent, total_slots_for_last_intent, count_slots_provided_total).
        """
        if not mrs:
            if history is not None:
                history.update_last_intent("out_of_domain")
            return "out_of_domain", 0, 0

        intents = [str(m.get("intent", "")).strip() or "out_of_domain" for m in mrs]
        last_intent = intents[-1]
        if last_intent not in POSSIBLE_INTENTS:
            last_intent = "out_of_domain"

        if history is not None:
            history.update_last_intent(last_intent)
            # store other intents for debug/trace
            history.other_intents.clear()
            for it in intents[:-1]:
                if it and it != last_intent:
                    history.insert_other_intent(it)

        total_slots = self._total_slots_for_intent(last_intent)
        count = 0
        if update:
            summary = self.apply_mrs(mrs)
            count = int(summary.get("count_provided", 0))

        return last_intent, total_slots, count

    def deferred_summary(self, mr: Optional[Dict[str, Any]]) -> str:
        """
        Deterministic short description of a deferred MR for the continue gate.
        Must not rely on the LLM.
        """
        if not isinstance(mr, dict):
            return "the next request"

        intent = str(mr.get("intent", "")).strip()
        slots = mr.get("slots", {}) or {}
        if not isinstance(slots, dict):
            slots = {}

        if intent == "inspect":
            day = normalize_day(slots.get("target_day", None))
            return f"show details for {day}" if day else "show day details"

        if intent == "show_week":
            return "show the weekly plan"

        if intent == "confirm":
            return "finalize the plan and generate the shopping list"

        if intent == "select_menu":
            return "choose a menu"

        if intent == "refine":
            r_type = normalize_upper_enum(slots.get("refine_type", None)) or ""
            day = normalize_day(slots.get("target_day", None))
            mode = normalize_upper_enum(slots.get("mode", None)) or ""
            value = slots.get("value", None)

            if r_type == "SWAP_DAY" and day:
                # In your NLU logs, mode is often SUGGEST
                if mode == "SUGGEST":
                    return f"suggest an alternative for {day}"
                return f"swap {day}"

            if r_type in {"ADD_AVOID_ITEM", "REMOVE_AVOID_ITEM"}:
                op = "add" if r_type == "ADD_AVOID_ITEM" else "remove"
                item = ""
                if value is not None and not is_nullish(value):
                    item = _canon_avoid_token(value)
                return f"update foods to avoid ({op} {item})" if item else "update foods to avoid"

            return "refine the plan"

        return "the next request"


    def _total_slots_for_intent(self, intent: str) -> int:
        if intent == "plan":
            return 4
        if intent == "select_menu":
            return 1
        if intent == "inspect":
            return 1
        if intent == "refine":
            return 4
        if intent == "confirm":
            return 0
        if intent == "show_week":
            return 0
        return 0

    # ------------------------- state inspection -----------------------------

    
    def missing_plan_slots(self) -> List[str]:
        from intents_schema import missing_plan_slots_from_constraints
        return missing_plan_slots_from_constraints(self.constraints)
    
    def menus_exist(self) -> bool:
        return (
            isinstance(self.menus, dict)
            and self.menus.get("1") is not None
            and self.menus.get("2") is not None
        )

    def has_active_menu(self) -> bool:
        return self.active_menu_id in (1, 2) and self.active_menu is not None

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "constraints": copy.deepcopy(self.constraints),
            "menus": copy.deepcopy(self.menus),
            "active_menu_id": self.active_menu_id,
            "active_menu": copy.deepcopy(self.active_menu),
            "last_referenced_day": self.last_referenced_day,
            "awaiting_slot": self.awaiting_slot,
            "reprompt_count": self.reprompt_count,
            "last_user_mr": copy.deepcopy(self.last_user_mr),
            "pending_action": copy.deepcopy(self.pending_action),
            "last_denied_action": copy.deepcopy(self.last_denied_action),
            "pending_mrs": copy.deepcopy(self.pending_mrs),
            "deferred_mrs_count": len(self.deferred_mrs),
            "deferred_next": copy.deepcopy(self.deferred_mrs[0]) if self.deferred_mrs else None,


        }

    # -------------------------- state mutation ------------------------------

    def note_request_info(self, slot: str) -> None:
        """
        Track that the system asked the user for a specific slot via request_info(slot).

        - If we ask for the same slot again consecutively, increment reprompt_count.
        - If we switch to a new slot, set awaiting_slot and reset reprompt_count.
        """
        slot = (slot or "").strip()
        if not slot:
            return  # defensive: ignore empty slot names

        if self.awaiting_slot == slot:
            self.reprompt_count += 1
        else:
            self.awaiting_slot = slot
            self.reprompt_count = 0

    def clear_awaiting(self) -> None:
        """
        Clear the "awaiting_slot" context and its reprompt counter.
        Call this when the system moves on from slot-filling to another action.
        """
        self.awaiting_slot = None
        self.reprompt_count = 0

    def _clear_awaiting_if_answered(self, slots: Dict[str, Any]) -> None:
        """
        Fix B: If we are awaiting a slot and this MR provides it (validly), clear awaiting.
        Keep validation lightweight and aligned with policy/sanitizers.
        """
        awaiting = getattr(self, "awaiting_slot", None)
        awaiting = str(awaiting).strip() if not is_nullish(awaiting) else ""
        if not awaiting:
            return

        if not isinstance(slots, dict):
            return

        if awaiting not in slots:
            return

        val = slots.get(awaiting, None)
        if is_nullish(val):
            return

        # Minimal validity checks for the two problematic fields in your logs
        if awaiting == "target_day":
            day = normalize_day(val)
            if day in ALLOWED_DAYS:
                self.clear_awaiting()
            return

        if awaiting == "menu_id":
            try:
                mid = int(val)
            except Exception:
                return
            if mid in (1, 2):
                self.clear_awaiting()
            return

        # For other slots, "non-null" is sufficient (policy/sanitizers validate next step)
        self.clear_awaiting()



    def set_menus(self, menu1: Dict[str, Any], menu2: Dict[str, Any]) -> None:
        """Called by DM after GENERATE_TWO_MENUS()."""
        self.menus["1"] = copy.deepcopy(menu1)
        self.menus["2"] = copy.deepcopy(menu2)
        self.phase = "AWAITING_MENU_SELECTION"
        # Selecting a new plan invalidates prior active choice
        self.active_menu_id = None
        self.active_menu = None
        self.last_referenced_day = None
        self.pending_action = None
        self.awaiting_slot = None
        self.reprompt_count = 0
        self.deferred_mrs = []




    def set_active_menu(self, menu_id: int) -> bool:
        """Called by DM after SELECT_MENU."""
        if menu_id not in (1, 2):
            return False
        key = str(menu_id)
        if self.menus.get(key) is None:
            return False
        self.active_menu_id = menu_id
        self.pending_action = None

        self.active_menu = copy.deepcopy(self.menus[key])
        self.phase = "ACTIVE_MENU"
        self.awaiting_slot = None
        self.reprompt_count = 0
        self.deferred_mrs = []


        return True

    def set_phase(self, phase: Phase) -> None:
        self.phase = phase

    def clear(self) -> None:
        self.phase = "AWAITING_PLAN"
        self.constraints = {"servings": None, "time_limit": None, "calorie_level": None, "avoid_items": None}
        self.menus = {"1": None, "2": None}
        self.active_menu_id = None
        self.active_menu = None
        self.last_referenced_day = None
        self.awaiting_slot = None
        self.reprompt_count = 0
        self.last_user_mr = None
        self.pending_action = None
        self.last_denied_action = None
        self.pending_mrs = []
        self.deferred_mrs = []
        self.turn_id = 0

    def enqueue_mrs(self, mrs: List[Dict[str, Any]]) -> None:
        """
        Add new MRs to the pending queue (deepcopied).
        Intended to be called once per user turn after validation/normalization.
        """
        if not mrs:
            return
        for mr in mrs:
            if isinstance(mr, dict):
                self.pending_mrs.append(copy.deepcopy(mr))

    def has_pending(self) -> bool:
        return bool(self.pending_mrs)

    def peek_pending(self, idx: int = 0) -> Optional[Dict[str, Any]]:
        if idx < 0 or idx >= len(self.pending_mrs):
            return None
        return self.pending_mrs[idx]

    def pop_pending(self, idx: int = 0) -> Optional[Dict[str, Any]]:
        if idx < 0 or idx >= len(self.pending_mrs):
            return None
        return self.pending_mrs.pop(idx)

    def remove_pending(self, mr: Dict[str, Any]) -> bool:
        """
        Remove the first pending MR that is deeply equal to `mr`.
        Returns True if removed.
        """
        for i, x in enumerate(self.pending_mrs):
            if x == mr:
                self.pending_mrs.pop(i)
                return True
        return False
    
    def prune_pending(self) -> None:
        """
        Conservative pruning to prevent obviously stale items from accumulating.
        This does NOT implement full consume/keep semantics (Step 5).
        """
        pruned: List[Dict[str, Any]] = []
        for mr in self.pending_mrs:
            intent = str(mr.get("intent", "")).strip()

            # If plan is already confirmed, drop non-help/out_of_domain.
            if self.phase == "CONFIRMED" and intent not in {"help", "out_of_domain"}:
                continue

            pruned.append(mr)

        self.pending_mrs = pruned

    def has_deferred(self) -> bool:
        return bool(self.deferred_mrs)

    def peek_deferred(self) -> Optional[Dict[str, Any]]:
        if not self.deferred_mrs:
            return None
        return self.deferred_mrs[0]

    def pop_deferred(self) -> Optional[Dict[str, Any]]:
        if not self.deferred_mrs:
            return None
        return self.deferred_mrs.pop(0)

    def clear_deferred(self) -> None:
        self.deferred_mrs = []


    def _stamp_turn(self, mrs: List[Dict[str, Any]], turn_id: int) -> None:
        for m in mrs:
            if isinstance(m, dict):
                m["_turn_id"] = turn_id

    def prune_pending_by_turn(self, keep_last_n_turns: int = 2) -> None:
        """
        Keep only MRs from the current turn and (optionally) the immediately previous turn.
        keep_last_n_turns=2 matches your prior keep_from = turn_id - 1 behavior.
        """
        keep_from = self.turn_id - (keep_last_n_turns - 1)
        self.pending_mrs = [
            m for m in self.pending_mrs
            if int(m.get("_turn_id", self.turn_id)) >= keep_from
        ]

    def _is_noop_ack(self, mr: Dict[str, Any]) -> bool:
        """True if this MR is an out_of_domain ACK that should be ignored."""
        if not isinstance(mr, dict):
            return False
        if str(mr.get("intent", "")).strip() != "out_of_domain":
            return False
        slots = mr.get("slots", {}) or {}
        if not isinstance(slots, dict):
            return False
        return str(slots.get("ood_type", "") or "").strip().upper() == "ACK"


    def _deferral_allowed_now(self) -> bool:
        """
        Allow multi-intent deferral only when we're in a stable state:
        - active menu selected
        - not in slot-filling
        - no pending confirmation (e.g., swap)
        - plan is complete
        """
        return (
            self.has_active_menu()
            and (self.awaiting_slot is None)
            and (self.pending_action is None)
            and (len(self.missing_plan_slots()) == 0)
        )


    def ingest_turn(self, mrs: List[Dict[str, Any]], history: Optional[History] = None) -> None:
        """
        Tracker-owned: advances turn, stamps, applies to state, enqueues, and prunes.

        Deferral-aware:
        - Drop no-op ACK MRs entirely
        - If safe and 2+ MRs, apply/enqueue only the first and defer the rest (FIFO)
        """
        # 1) advance turn
        self.turn_id += 1
        self.last_denied_action = None

        if not mrs:
            return

        # 2) Filter no-op ACKs before anything else
        filtered: List[Dict[str, Any]] = []
        for mr in mrs:
            if not isinstance(mr, dict):
                continue
            if self._is_noop_ack(mr):
                continue
            filtered.append(mr)

        if not filtered:
            # Only ACKs came in; treat as no-op
            return

        # 3) Decide whether deferral mode is allowed (minimally disruptive gating)
        do_deferral = self._deferral_allowed_now() and (len(filtered) >= 2)

        # Defensive: never defer if these are not satisfied (even if gate changes later)
        if do_deferral and (self.pending_action is not None or self.missing_plan_slots()):
            do_deferral = False

        if do_deferral:
            mrs_now = [filtered[0]]
            mrs_later = filtered[1:]

            # Replace (do not append) to avoid stale deferred from prior turns
            self.deferred_mrs = [copy.deepcopy(x) for x in mrs_later]

            # Stamp: now + (optionally) stamp deferred for debugging (you will re-stamp on activation later)
            self._stamp_turn(mrs_now, self.turn_id)
            

            # Apply ONLY the first MR to state
            self.creation_multi(mrs_now, history=history, update=True)

            # Enqueue ONLY the first MR
            mrs_now_non_plan = [m for m in mrs_now if str(m.get("intent", "")).strip() != "plan"]
            self.enqueue_mrs(mrs_now_non_plan)


        else:
            # Old behavior (but with ACKs removed)
            self._stamp_turn(filtered, self.turn_id)
            self.creation_multi(filtered, history=history, update=True)
            filtered_non_plan = [m for m in filtered if str(m.get("intent", "")).strip() != "plan"]
            self.enqueue_mrs(filtered_non_plan)


        # 5) prune by recency window
        self.prune_pending_by_turn(keep_last_n_turns=2)

        # 6) conservative prune (existing)
        self.prune_pending()

    
    def select_next_mr(self) -> Dict[str, Any]:
        """
        Returns the next MR to handle.
        If None would have been returned by select_next_pending_index(), synthesize an MR
        based on current state. Always stamps _turn_id for consistency.
        """
        idx = self.select_next_pending_index()
        if idx is None:
            if self.missing_plan_slots():
                mr = {"intent": "plan", "slots": {}}
                reason = "missing_plan_slots"
            elif self.phase == "AWAITING_PLAN" and not self.menus_exist():
                mr = {"intent": "plan", "slots": {}}
                reason = "plan_complete_need_menus"
            elif self.phase == "AWAITING_MENU_SELECTION" or (self.menus_exist() and not self.has_active_menu()):
                mr = {"intent": "select_menu", "slots": {}}
                reason = "awaiting_menu_selection"
            else:
                mr = {"intent": "out_of_domain", "slots": {}}
                reason = "no_pending_match"

            mr["_turn_id"] = self.turn_id
            mr["_synthetic"] = True
            mr["_synthetic_reason"] = reason
            mr["_selected_from_pending_index"] = None
            return mr

        # queued MR
        mr = self.pending_mrs[idx]

        # ensure it's stamped (defensive)
        if "_turn_id" not in mr:
            mr["_turn_id"] = self.turn_id

        mr["_synthetic"] = False
        mr["_synthetic_reason"] = None
        mr["_selected_from_pending_index"] = idx
        return mr


    def select_next_pending_index(self) -> Optional[int]:
        """
        Returns index in pending_mrs of the MR to handle next.
        None means: synthesize an MR (e.g., request missing plan slots).
        """
        if not self.pending_mrs:
            return None

        # 1) If plan is incomplete, handle plan MRs (newest-first).
        if self.missing_plan_slots():
            for i in range(len(self.pending_mrs) - 1, -1, -1):
                mr = self.pending_mrs[i]
                if str(mr.get("intent", "")).strip() == "plan":
                    return i
            return None  # synthetic plan

        # 2) If we must select a menu, handle select_menu MRs (newest-first).
        if self.phase == "AWAITING_MENU_SELECTION" or (self.menus_exist() and not self.has_active_menu()):
            for i in range(len(self.pending_mrs) - 1, -1, -1):
                mr = self.pending_mrs[i]
                if str(mr.get("intent", "")).strip() == "select_menu":
                    return i
            return None  # synthetic select_menu

        # 3) Otherwise, prioritize by intent, but pick newest matching MR.
        # Confirm should outrank refine; otherwise confirm is effectively unusable
        # as soon as any refine exists in the backlog.
        priority = ["confirm", "show_week", "inspect", "refine", "help", "out_of_domain"]
        for p in priority:
            for i in range(len(self.pending_mrs) - 1, -1, -1):
                mr = self.pending_mrs[i]
                if str(mr.get("intent", "")).strip() == p:
                    return i

        # 4) Fallback: newest MR, not oldest.
        return len(self.pending_mrs) - 1

    def _get_avoid_set(self) -> set[str]:
        cur = self.constraints.get("avoid_items", None)
        if cur is None:
            return set()
        if isinstance(cur, list):
            out: set[str] = set()
            for x in cur:
                tok = _canon_avoid_token(x)
                if tok and tok in ALLOWED_AVOID_ITEMS:
                    out.add(tok)
            return out
        return set()


    def _set_avoid_set(self, s: set[str]) -> None:
        # Keep stable ordering for determinism; store only allowed tokens.
        self.constraints["avoid_items"] = sorted([x for x in s if x in ALLOWED_AVOID_ITEMS])


    def _can_coerce_refine_to_plan_avoid(self) -> bool:
        # Coerce only while plan is incomplete and avoid_items is still missing
        return "avoid_items" in self.missing_plan_slots()

    def _intent_of(self, mr: Dict[str, Any]) -> str:
        return str((mr or {}).get("intent", "")).strip() or "out_of_domain"

    def _slots_of(self, mr: Dict[str, Any]) -> Dict[str, Any]:
        s = (mr or {}).get("slots", {}) or {}
        return s if isinstance(s, dict) else {}

    def _remove_first_by_intent(self, intent: str) -> bool:
        for i, x in enumerate(self.pending_mrs):
            if self._intent_of(x) == intent:
                self.pending_mrs.pop(i)
                return True
        return False
    
    def update_pending_after_action(self, selected_mr: Dict[str, Any], action: str, payload: Dict[str, Any]) -> None:
        """
        Consume/keep logic for pending_mrs after policy+execution.

        selected_mr should be the SNAPSHOT you selected in main (deepcopy),
        so equality removal works even if tracker state mutated.
        """
        action = (action or "").strip().lower()

        def _is_refuse_pending(mr: Dict[str, Any]) -> bool:
            if self._intent_of(mr) != "out_of_domain":
                return False
            s = self._slots_of(mr)
            return str(s.get("ood_type", "") or "").strip().upper() == "REFUSE_PENDING"

        def _is_success(action: str, payload: Dict[str, Any]) -> bool:
            if action in {"request_info", "fallback"}:
                return False
            return not bool((payload or {}).get("error"))
        
        resolved = _is_success(action, payload) or _is_refuse_pending(selected_mr)

        try:
            intent = self._intent_of(selected_mr)
            sel_turn_id = selected_mr.get("_turn_id")

            is_synthetic = bool(selected_mr.get("_synthetic", False))

            # 0) If we asked for info, keep everything (we're waiting for user input).
            if action == "request_info":
                pass

            # 1) Fallback: remove only out_of_domain, keep the rest.
            elif action == "fallback":
                if intent == "out_of_domain":
                    if not is_synthetic:
                        self.remove_pending(selected_mr)

            # 2) propose_menus: goal progressed, clear plan requests.
            elif action == "propose_menus":
                 if not (payload or {}).get("error"):
                    self.pending_mrs.clear()

            # 3) set_active_menu: if it worked, clear select_menu MRs
            elif action == "set_active_menu":
                ok = bool((payload or {}).get("ok", False))
                if ok:
                    self.pending_mrs = [m for m in self.pending_mrs if self._intent_of(m) != "select_menu"]
                # else: keep as-is

            # 4) show_day: if succeeded, remove the inspect MR we handled
            elif action == "show_day":
                if (payload or {}).get("details") is not None and not (payload or {}).get("error"):
                    if intent == "inspect":
                        if not is_synthetic: 
                            self.remove_pending(selected_mr)
                    else:
                        self._remove_first_by_intent("inspect")

            elif action == "show_week":
                if not (payload or {}).get("error"):
                    if intent == "show_week":
                        if not is_synthetic:
                            self.remove_pending(selected_mr)
                    else:
                        self._remove_first_by_intent("show_week")


            # 5) update_avoid: if succeeded, remove refine MR we handled (best-effort)
            elif action == "update_avoid":
                if not (payload or {}).get("error"):
                    if intent == "refine":
                        if not is_synthetic:
                            self.remove_pending(selected_mr)
                    else:
                        self._remove_first_by_intent("refine")

            # 6) suggest_swap_day: if suggested=True, consume the refine MR
            elif action == "suggest_swap_day":
                if bool((payload or {}).get("suggested", False)) and not (payload or {}).get("error"):
                    if intent == "refine":
                        if not is_synthetic:
                            self.remove_pending(selected_mr)
                    else:
                        self._remove_first_by_intent("refine")

            # 7) swap_day: if swapped=True, consume a refine MR (swap-type) or selected
            elif action == "swap_day":
                if bool((payload or {}).get("swapped", False)) and not (payload or {}).get("error"):
                    if intent in {"refine","confirm"}:
                        if not is_synthetic:
                            self.remove_pending(selected_mr)
                    else:
                        self._remove_first_by_intent("refine")
                        self._remove_first_by_intent("confirm")

            # 8) confirm_plan: if succeeded, clear confirm (and optionally clear all)
            elif action == "confirm_plan":
                if (payload or {}).get("shopping_list") is not None and not (payload or {}).get("error"):
                    self.pending_mrs = [m for m in self.pending_mrs if self._intent_of(m) != "confirm"]
                    # optional: clear all
                    # self.pending_mrs.clear()

            # 9) Default: if we successfully did something, remove the MR we tried to handle
            else:
                if not is_synthetic:
                    self.remove_pending(selected_mr)

            # Fix 3: after a successful non-fallback action, drop low-value leftovers from the same user turn
            if sel_turn_id is not None and _is_success(action, payload):
                self.pending_mrs = [
                    m for m in self.pending_mrs
                    if m.get("_turn_id") != sel_turn_id
                    or self._intent_of(m) not in {"help", "out_of_domain"}
                ]

            # ---- Structured deferral: set CONTINUE_DEFERRED gate when appropriate ----
            # Only set this gate when:
            # - no other pending gate is active (e.g., SWAP_DAY confirmation has precedence)
            # - we actually have deferred work
            # - we are not asking for more info (request_info already implies a gate)
            # - the action just succeeded (no error)
            if self.pending_action is None:
                if action != "request_info" and self.has_deferred() and resolved:
                    nxt = self.peek_deferred()
                    self.pending_action = {
                        "type": "CONTINUE_DEFERRED",
                        "next": self.deferred_summary(nxt),
                        "remaining": len(self.deferred_mrs),
                    }

        finally:
            # Always enforce recency window + conservative pruning
            self.prune_pending_by_turn(keep_last_n_turns=2)
            self.prune_pending()



    def set_pending_swap(self, day: str, recipe_id: str) -> None:
        self.pending_action = {"type": "SWAP_DAY", "day": day, "recipe_id": recipe_id}


    # -------------------------- MR application ------------------------------
    def _apply_slots(self, mr: Dict[str, Any]) -> int:
        """
        Apply a single MR’s slots to state
        """
        intent = str(mr.get("intent", "")).strip()
        slots = mr.get("slots", {}) or {}

        count_provided = 0

        if intent == "plan":
            servings = slots.get("servings", None)
            if not is_nullish(servings):
                try:
                    s = int(servings)
                    if 1 <= s <= 6:
                        self.constraints["servings"] = s
                        count_provided += 1
                except (TypeError, ValueError):
                    pass


            time_limit = normalize_upper_enum(slots.get("time_limit", None))
            if time_limit and time_limit in ALLOWED_TIME_LIMITS:
                self.constraints["time_limit"] = time_limit
                count_provided += 1

            cal = normalize_upper_enum(slots.get("calorie_level", None))
            if cal and cal in ALLOWED_CALORIE_LEVELS:
                self.constraints["calorie_level"] = cal
                count_provided += 1

            avoid_raw = normalize_avoid_items(slots.get("avoid_items", None))
            if avoid_raw is not None:
                if avoid_raw == []:
                    # "no" / "none": only counts as providing the slot if it was previously missing
                    if self.constraints.get("avoid_items", None) is None:
                        self.constraints["avoid_items"] = []
                        count_provided += 1
                    # else: already had an avoid list; "no" means no change, no increment
                else:
                    # Only accept allowed items; ignore unknowns.
                    cur = self._get_avoid_set()
                    before = len(cur)

                    for x in avoid_raw:
                        tok = _canon_avoid_token(x)
                        if tok and tok in ALLOWED_AVOID_ITEMS:
                            cur.add(tok)

                    if len(cur) > before:
                        self._set_avoid_set(cur)
                        count_provided += 1
                    else:
                        # If user provided only unknown avoids and avoid_items was missing,
                        # keep it missing so missing_plan_slots() continues to ask.
                        if self.constraints.get("avoid_items", None) is None:
                            self.constraints["avoid_items"] = None
        
            return count_provided

                
        if intent == "select_menu":
            menu_id = slots.get("menu_id", None)
            if not is_nullish(menu_id):
                try:
                    _ = int(menu_id)
                    count_provided += 1
                except (TypeError, ValueError):
                    pass
            return count_provided

        if intent == "inspect":
            day = normalize_day(slots.get("target_day", None))
            if day and day in ALLOWED_DAYS:
                self.last_referenced_day = day
                count_provided += 1
            return count_provided

        if intent == "refine":
            for k in ("refine_type", "target_day", "value", "mode"):
                if not is_nullish(slots.get(k, None)):
                    count_provided += 1

            day = normalize_day(slots.get("target_day", None))
            if day and day in ALLOWED_DAYS:
                self.last_referenced_day = day

            rt = normalize_upper_enum(slots.get("refine_type", None))
            val_raw = slots.get("value", None)
            tok = _canon_avoid_token(val_raw)

            if rt in ("ADD_AVOID_ITEM", "REMOVE_AVOID_ITEM"):
                # Only accept allowed items; ignore unknowns.
                if tok and tok in ALLOWED_AVOID_ITEMS:
                    cur = self._get_avoid_set()
                    before = len(cur)

                    if rt == "ADD_AVOID_ITEM":
                        cur.add(tok)
                    else:  # REMOVE_AVOID_ITEM
                        cur.discard(tok)

                    if len(cur) != before:
                        self._set_avoid_set(cur)
                        # counts as "provided" if it produced an actual state change
                        count_provided += 1

            return count_provided
        
        if intent == "show_week":
            return 0


        if intent == "confirm":
            return 0

        return 0

    
    def apply_mr(self, mr: Dict[str, Any]) -> int:
        """
        Apply a single NLU MR (flat JSON with intent+slots) to state.
        Returns how many non-null slot values were provided this turn (lightweight metric).
        """
        intent = str(mr.get("intent", "")).strip()
        slots = mr.get("slots", {}) or {}
        self.last_user_mr = {"intent": intent, "slots": copy.deepcopy(slots)}
        self.last_denied_action = None

        # OOD subtype: user refused a pending suggested swap
        if intent == "out_of_domain" and self.pending_action is not None:
            ood_type = str(slots.get("ood_type", "") or "").strip().upper()
            p_type = str(self.pending_action.get("type", "") or "").strip().upper()
            if ood_type == "REFUSE_PENDING" and p_type == "SWAP_DAY":
                self.last_denied_action = copy.deepcopy(self.pending_action)
                self.pending_action = None
                # No slot application needed for OOD
                return 0

        # Expire pending_action unless the user is responding to it.
        if self.pending_action is not None:
            if intent == "confirm":
                pass
            elif intent == "refine":
                p = self.pending_action
                p_type = str(p.get("type") or "").upper()
                r_type = normalize_upper_enum(slots.get("refine_type", None)) or ""
                r_day = normalize_day(slots.get("target_day", None))

                if not (p_type == "SWAP_DAY" and r_type == "SWAP_DAY" and r_day and r_day == p.get("day")):
                    self.pending_action = None
            else:
                self.pending_action = None

        count = self._apply_slots(mr)

        # Fix B: if this MR answers what we were waiting for, clear awaiting_slot
        self._clear_awaiting_if_answered(slots)

        return count


    

    def apply_mrs(self, mrs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply multiple NLU MRs sequentially as ONE user turn.

        Key difference vs calling apply_mr() in a loop:
        - pending_action expiry is handled ONCE per turn (multi-MR-safe)

        Returns a small summary useful for debugging.
        """
        if not mrs:
            return {"intents": [], "count_provided": 0}

        # Normalize intent strings defensively
        intents = [str(m.get("intent", "")).strip() for m in mrs]
        intents_set = set(intents)

        # Reset one-turn denial marker
        self.last_denied_action = None

        # If this turn contains an OOD subtype REFUSE_PENDING, treat it as refusal of the pending suggested swap
        if self.pending_action is not None and "out_of_domain" in intents_set:
            p_type = str(self.pending_action.get("type", "") or "").strip().upper()
            if p_type == "SWAP_DAY":
                for mr in mrs:
                    if str(mr.get("intent", "")).strip() != "out_of_domain":
                        continue
                    s = (mr.get("slots", {}) or {})
                    if isinstance(s, dict):
                        ood_type = str(s.get("ood_type", "") or "").strip().upper()
                        if ood_type == "REFUSE_PENDING":
                            self.last_denied_action = copy.deepcopy(self.pending_action)
                            self.pending_action = None
                            break

        # ---- Multi-MR-safe pending_action expiry (runs ONCE per turn) ----
        if self.pending_action is not None:
            keep_pending = False

            # 1) If any MR is confirm, keep pending (executor/policy decides meaning)
            if "confirm" in intents_set:
                keep_pending = True
            else:
                # 2) If any MR is refine SWAP_DAY for the SAME day, keep pending
                p = self.pending_action
                p_type = str(p.get("type") or "").upper()
                if p_type == "SWAP_DAY":
                    p_day = p.get("day")

                    for mr in mrs:
                        if str(mr.get("intent", "")).strip() != "refine":
                            continue
                        slots = mr.get("slots", {}) or {}
                        r_type = normalize_upper_enum(slots.get("refine_type", None)) or ""
                        r_day = normalize_day(slots.get("target_day", None))
                        if r_type == "SWAP_DAY" and r_day and r_day == p_day:
                            keep_pending = True
                            break

            if not keep_pending:
                self.pending_action = None

        # ---- Coerce refine avoid-items into plan slot filling when collecting plan ----
        if self._can_coerce_refine_to_plan_avoid():
            avoid_set = self._get_avoid_set()
            coerced_any = False

            for mr in mrs:
                if str(mr.get("intent", "")).strip() != "refine":
                    continue
                slots = mr.get("slots", {}) or {}
                r_type = normalize_upper_enum(slots.get("refine_type", None)) or ""
                value = slots.get("value", None)
                value_norm = normalize_avoid_items([value]) if value is not None else None
                # normalize_avoid_items returns list or None; we want single item
                item = None
                if isinstance(value_norm, list) and value_norm:
                    item = _canon_avoid_token(value_norm[0])

                if not item or item not in ALLOWED_AVOID_ITEMS:
                    continue

                if r_type == "ADD_AVOID_ITEM":
                    avoid_set.add(item)
                    coerced_any = True
                elif r_type == "REMOVE_AVOID_ITEM":
                    if item in avoid_set:
                        avoid_set.remove(item)
                        coerced_any = True


            if coerced_any:
                self._set_avoid_set(avoid_set)


        # ---- Apply each MR WITHOUT per-MR pending expiry ----
        count_total = 0
        for mr in mrs:
            count_total += self._apply_slots(mr)

        # Fix B: if any MR in this turn answers what we were waiting for, clear awaiting_slot
        for mr in mrs:
            s = mr.get("slots", {}) or {}
            if isinstance(s, dict):
                self._clear_awaiting_if_answered(s)
                if is_nullish(getattr(self, "awaiting_slot", None)) or str(getattr(self, "awaiting_slot", "")).strip() == "":
                    break

        # last_user_mr: keep the last MR of the turn (most recent)
        last = mrs[-1]
        self.last_user_mr = {
            "intent": str(last.get("intent", "")).strip(),
            "slots": copy.deepcopy(last.get("slots", {}) or {}),
        }

        return {"intents": intents, "count_provided": count_total}

