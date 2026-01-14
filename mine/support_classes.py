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


Phase = Literal["AWAITING_PLAN", "AWAITING_MENU_SELECTION", "ACTIVE_MENU", "CONFIRMED"]

ALLOWED_DAYS = {"Mon", "Tue", "Wed", "Thu", "Fri"}


ALLOWED_TIME_LIMITS = {"FAST", "NORMAL"}
ALLOWED_CALORIE_LEVELS = {"LOW", "MED", "HIGH"}

ALLOWED_AVOID_ITEMS = {
    "nuts", "dairy", "gluten", "soy", "egg", "sesame", "fish", "shellfish", "meat"
}

POSSIBLE_INTENTS = {"plan", "select_menu", "inspect", "refine", "confirm", "help", "out_of_domain"}


def _is_nullish(x: Any) -> bool:
    """Treat common LLM-returned null variants as null."""
    return x is None or x == "null" or x == "None" or x == ""


def _normalize_upper_enum(x: Any) -> Optional[str]:
    if _is_nullish(x):
        return None
    if isinstance(x, str):
        return x.strip().upper()
    return str(x).strip().upper()


def _normalize_day(x: Any) -> Optional[str]:
    """
    Validation-only day normalization:
    - Accept canonical day codes only (Mon/Tue/Wed/Thu/Fri), case-insensitive.
    - Do NOT map full names ("monday") or other aliases; NLU should do that.
    """
    if _is_nullish(x):
        return None
    if not isinstance(x, str):
        return None

    s = x.strip()
    if not s:
        return None

    # case-only normalization
    s3 = s[:3].title()  # "mon" -> "Mon", "MON" -> "Mon"
    return s3 if s3 in ALLOWED_DAYS else None



def _normalize_avoid_items(x: Any) -> Optional[List[str]]:
    """
    Accept:
    - list[str]
    - comma-separated str
    - single str
    Returns a list[str] (possibly empty) or None if nullish.
    """
    if _is_nullish(x):
        return None

    raw: List[str] 
    if isinstance(x, list):
        raw = [str(it).strip() for it in x]
    elif isinstance(x, str):
        # try splitting by commas
        raw = [p.strip() for p in x.split(",")] if "," in x else [x.strip()]
    else:
        raw = [str(x).strip()]

    items: List[str] = []
    for it in raw:
        if not it:
            continue
        items.append(it.lower())

    return items


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
        "avoid_items": [],  # empty list means "no avoids"
    })
    menus: Dict[str, Optional[Dict[str, Any]]] = field(default_factory=lambda: {"1": None, "2": None})
    active_menu_id: Optional[int] = None
    active_menu: Optional[Dict[str, Any]] = None
    last_referenced_day: Optional[str] = None

    # Optional: keep last MR for debugging/logging
    last_user_mr: Optional[Dict[str, Any]] = None

    possible_intents: List[str] = field(default_factory=lambda: sorted(POSSIBLE_INTENTS))

    # --------- convenience / compatibility helpers (Marina-like API) ---------

    def creation(self, input: Dict[str, Any], history: Optional[History] = None, update: bool = True) -> Tuple[str, int, int]:
        """
        Compatibility wrapper similar to Marina's tracker:
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
            self.last_user_mr = {"intent": "out_of_domain"}
            return intent, total_slots, 0

        if update:
            count = self.apply_mr(input)

        return intent, total_slots, count

    def _total_slots_for_intent(self, intent: str) -> int:
        if intent == "plan":
            return 4
        if intent == "select_menu":
            return 1
        if intent == "inspect":
            return 1
        if intent == "refine":
            return 3
        if intent == "confirm":
            return 0
        return 0

    # ------------------------- state inspection -----------------------------

    def missing_plan_slots(self) -> List[str]:
        missing: List[str] = []
        if self.constraints.get("servings") is None:
            missing.append("servings")
        if self.constraints.get("time_limit") is None:
            missing.append("time_limit")
        if self.constraints.get("calorie_level") is None:
            missing.append("calorie_level")
        # avoid_items defaults to [] (not missing)
        return missing

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
        }

    # -------------------------- state mutation ------------------------------

    def set_menus(self, menu1: Dict[str, Any], menu2: Dict[str, Any]) -> None:
        """Called by DM after GENERATE_TWO_MENUS()."""
        self.menus["1"] = copy.deepcopy(menu1)
        self.menus["2"] = copy.deepcopy(menu2)
        self.phase = "AWAITING_MENU_SELECTION"
        # Selecting a new plan invalidates prior active choice
        self.active_menu_id = None
        self.active_menu = None
        self.last_referenced_day = None

    def set_active_menu(self, menu_id: int) -> bool:
        """Called by DM after SELECT_MENU."""
        if menu_id not in (1, 2):
            return False
        key = str(menu_id)
        if self.menus.get(key) is None:
            return False
        self.active_menu_id = menu_id
        self.active_menu = copy.deepcopy(self.menus[key])
        self.phase = "ACTIVE_MENU"
        return True

    def set_phase(self, phase: Phase) -> None:
        self.phase = phase

    def clear(self) -> None:
        self.phase = "AWAITING_PLAN"
        self.constraints = {"servings": None, "time_limit": None, "calorie_level": None, "avoid_items": []}
        self.menus = {"1": None, "2": None}
        self.active_menu_id = None
        self.active_menu = None
        self.last_referenced_day = None
        self.last_user_mr = None

    # -------------------------- MR application ------------------------------

    def apply_mr(self, mr: Dict[str, Any]) -> int:
        """
        Apply a single NLU MR (flat JSON with intent+slots) to state.
        Returns how many non-null slot values were provided this turn (lightweight metric).
        """
        intent = str(mr.get("intent", "")).strip()
        slots = mr.get("slots", {}) or {}
        self.last_user_mr = {"intent": intent, "slots": copy.deepcopy(slots)}

        count_provided = 0

        if intent == "plan":
            # servings
            servings = slots.get("servings", None)
            if not _is_nullish(servings):
                try:
                    self.constraints["servings"] = int(servings)
                    count_provided += 1
                except (TypeError, ValueError):
                    pass

            # time_limit
            time_limit = _normalize_upper_enum(slots.get("time_limit", None))
            if time_limit and time_limit in ALLOWED_TIME_LIMITS:
                self.constraints["time_limit"] = time_limit
                count_provided += 1

            # calorie_level
            cal = _normalize_upper_enum(slots.get("calorie_level", None))
            if cal and cal in ALLOWED_CALORIE_LEVELS:
                self.constraints["calorie_level"] = cal
                count_provided += 1

            # avoid_items (optional; empty list is valid)
            avoid_raw = _normalize_avoid_items(slots.get("avoid_items", None))
            if avoid_raw is not None:
                # Keep only allowed values; DM can decide whether to ask clarification if something is dropped.
                cleaned = [a for a in avoid_raw if a in ALLOWED_AVOID_ITEMS]
                self.constraints["avoid_items"] = cleaned
                count_provided += 1

            # phase stays AWAITING_PLAN until DM actually generates menus
            return count_provided

        if intent == "select_menu":
            menu_id = slots.get("menu_id", None)
            if not _is_nullish(menu_id):
                try:
                    _ = int(menu_id)
                    count_provided += 1
                    # Do NOT set_active_menu here; executor does it after policy approves.
                except (TypeError, ValueError):
                    pass
            return count_provided


        if intent == "inspect":
            day = _normalize_day(slots.get("target_day", None))
            if day and day in ALLOWED_DAYS:
                self.last_referenced_day = day
                count_provided += 1
            return count_provided

        if intent == "refine":
            # We don't mutate menus here; DM/domain logic will.
            # We still record any provided fields for downstream DM logic.
            for k in ("refine_type", "target_day", "value"):
                if not _is_nullish(slots.get(k, None)):
                    count_provided += 1

            # Update last_referenced_day opportunistically if target_day is present
            day = _normalize_day(slots.get("target_day", None))
            if day and day in ALLOWED_DAYS:
                self.last_referenced_day = day

            return count_provided

        if intent == "confirm":
            # DM will enforce the selection gate and generate the shopping list.
            return 0

        # out_of_domain or unknown
        return 0
