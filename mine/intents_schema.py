# intents_schema.py
"""
Meal Kit Composer — intent/slot schema (minimal spec).

This module centralizes:
- the list of intents
- the slots per intent (flat JSON MR format)
- controlled vocabularies (enums) and validation rules
- minimal normalization utilities

It is grounded in the "Meal Kit Composer (Minimal Spec) — Project Reference Summary":
- flat NLU MR format
- minimal intents: plan, select_menu, inspect, refine, confirm
- controlled vocabularies and required-slot rules
- refine_type-dependent constraints (SWAP_DAY vs ADD/REMOVE_AVOID_ITEM)
:contentReference[oaicite:0]{index=0}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from support_classes import (
    POSSIBLE_INTENTS,
    ALLOWED_DAYS,
    ALLOWED_TIME_LIMITS,
    ALLOWED_CALORIE_LEVELS,
    ALLOWED_AVOID_ITEMS,
)


# ------------------------- Intent + slot inventory -------------------------

INTENT_SLOTS: Dict[str, List[str]] = {
    "plan": ["servings", "time_limit", "calorie_level", "avoid_items"],
    "select_menu": ["menu_id"],
    "inspect": ["target_day"],
    "refine": ["refine_type", "target_day", "value"],
    "confirm": [],
    "help": ["intent", "slot"],
    "out_of_domain": [],
    
}

# Required slots per intent (minimal configuration)
# Note: avoid_items is allowed to be null/empty; it is not required for PLAN completion.
REQUIRED_SLOTS: Dict[str, List[str]] = {
    "plan": ["servings", "time_limit", "calorie_level"],
    "select_menu": ["menu_id"],
    "inspect": ["target_day"],
    "refine": ["refine_type", "value"],  # target_day depends on refine_type
    "confirm": [],
    "out_of_domain": [],
}

# refine_type controlled vocab + refine-specific constraints
REFINE_TYPES = {"SWAP_DAY", "ADD_AVOID_ITEM", "REMOVE_AVOID_ITEM"}
SWAP_VALUE = "BEST_FIT"


# ------------------------- MR templates -------------------------

def mr_template(intent: str) -> Dict[str, Any]:
    """
    Return the canonical flat MR "shape" for an intent:
      {"intent": "...", "slots": {...}}
    with slots set to None (null).
    """
    intent = (intent or "").strip()
    if intent not in INTENT_SLOTS:
        intent = "out_of_domain"
    return {"intent": intent, "slots": {k: None for k in INTENT_SLOTS[intent]}}


# ------------------------- Normalization helpers -------------------------

def _is_nullish(x: Any) -> bool:
    return x is None or x == "null" or x == "None" or x == ""


def _upper(x: Any) -> Optional[str]:
    if _is_nullish(x):
        return None
    return str(x).strip().upper()


def _lower(x: Any) -> Optional[str]:
    if _is_nullish(x):
        return None
    return str(x).strip().lower()


def normalize_day(x: Any) -> Optional[str]:
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

    s3 = s[:3].title()  # "mon" -> "Mon", "MONDAY" -> "Mon"
    return s3 if s3 in ALLOWED_DAYS else None



def normalize_avoid_items(x: Any) -> Optional[List[str]]:
    """
    Normalize avoid_items to a list[str] (lowercase).
    Accepts:
      - list[str]
      - comma-separated string
      - single string
    Returns None if nullish.
    """
    if _is_nullish(x):
        return None

    if isinstance(x, list):
        raw = x
    else:
        s = str(x)
        raw = [p.strip() for p in s.split(",")] if "," in s else [s.strip()]

    out: List[str] = []
    for it in raw:
        it2 = (it or "").strip().lower()
        if it2:
            out.append(it2)
    return out


# ------------------------- Validation -------------------------

@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]
    normalized_mr: Dict[str, Any]


def normalize_mr(mr: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize an MR without enforcing correctness (validation is separate):
    - intent string cleanup
    - enum uppercasing for time_limit/calorie_level/refine_type
    - day normalization
    - avoid_items normalization
    - menu_id and servings numeric coercion where possible
    """
    intent = str(mr.get("intent", "")).strip()
    if intent not in POSSIBLE_INTENTS:
        intent = "out_of_domain"

    slots_in = mr.get("slots", {}) or {}
    slots_out: Dict[str, Any] = {}

    for k in INTENT_SLOTS.get(intent, []):
        slots_out[k] = slots_in.get(k, None)

    if intent == "plan":
        # servings
        if not _is_nullish(slots_out.get("servings")):
            try:
                slots_out["servings"] = int(slots_out["servings"])
            except Exception:
                pass
        # enums
        slots_out["time_limit"] = _upper(slots_out.get("time_limit"))
        slots_out["calorie_level"] = _upper(slots_out.get("calorie_level"))
        # avoid list
        slots_out["avoid_items"] = normalize_avoid_items(slots_out.get("avoid_items"))

    elif intent == "select_menu":
        if not _is_nullish(slots_out.get("menu_id")):
            try:
                slots_out["menu_id"] = int(slots_out["menu_id"])
            except Exception:
                pass

    elif intent == "inspect":
        slots_out["target_day"] = normalize_day(slots_out.get("target_day"))

    elif intent == "refine":
        slots_out["refine_type"] = _upper(slots_out.get("refine_type"))
        slots_out["target_day"] = normalize_day(slots_out.get("target_day"))
        # value: keep raw string for now; validation will interpret based on refine_type
        if not _is_nullish(slots_out.get("value")):
            slots_out["value"] = str(slots_out["value"]).strip()

    return {"intent": intent, "slots": slots_out}


def validate_mr(mr: Dict[str, Any]) -> ValidationResult:
    """
    Validate an MR against the minimal intent/slot/value specification.

    This does NOT apply the "menu selection gate" (that is DM policy, not schema).
    It only checks:
    - intent is known
    - required slots are present and well-typed
    - enums are in controlled vocab
    - refine_type-dependent constraints
    - avoid_items values are in the avoid vocabulary

    Returns (valid, errors, normalized_mr).
    """
    nm = normalize_mr(mr)
    intent = nm["intent"]
    slots = nm["slots"]

    errors: List[str] = []

    # Unknown intent
    if intent == "out_of_domain":
        return ValidationResult(valid=True, errors=[], normalized_mr=nm)

    # Required slots (base)
    for req in REQUIRED_SLOTS.get(intent, []):
        if _is_nullish(slots.get(req)):
            errors.append(f"Missing required slot: {req}")

    # Intent-specific validation
    if intent == "plan":
        # servings range
        s = slots.get("servings")
        if not _is_nullish(s):
            if not isinstance(s, int) or not (1 <= s <= 6):
                errors.append("servings must be an int in range 1–6")

        tl = slots.get("time_limit")
        if not _is_nullish(tl) and tl not in ALLOWED_TIME_LIMITS:
            errors.append(f"time_limit must be one of {sorted(ALLOWED_TIME_LIMITS)}")

        cal = slots.get("calorie_level")
        if not _is_nullish(cal) and cal not in ALLOWED_CALORIE_LEVELS:
            errors.append(f"calorie_level must be one of {sorted(ALLOWED_CALORIE_LEVELS)}")

        avoid = slots.get("avoid_items")
        if avoid is not None:
            if not isinstance(avoid, list):
                errors.append("avoid_items must be a list of strings (or null)")
            else:
                unknown = [a for a in avoid if a not in ALLOWED_AVOID_ITEMS]
                if unknown:
                    errors.append(
                        "Unknown avoid item(s): "
                        + ", ".join(sorted(set(unknown)))
                        + f". Allowed: {', '.join(sorted(ALLOWED_AVOID_ITEMS))}"
                    )

    elif intent == "select_menu":
        mid = slots.get("menu_id")
        if not _is_nullish(mid):
            if mid not in (1, 2):
                errors.append("menu_id must be 1 or 2")

    elif intent == "inspect":
        day = slots.get("target_day")
        if not _is_nullish(day) and day not in ALLOWED_DAYS:
            errors.append(f"target_day must be one of {sorted(ALLOWED_DAYS)}")

    elif intent == "refine":
        rt = slots.get("refine_type")
        if not _is_nullish(rt) and rt not in REFINE_TYPES:
            errors.append(f"refine_type must be one of {sorted(REFINE_TYPES)}")

        # refine_type-dependent rules
        if rt == "SWAP_DAY":
            day = slots.get("target_day")
            if _is_nullish(day):
                errors.append("target_day is required when refine_type=SWAP_DAY")
            elif day not in ALLOWED_DAYS:
                errors.append(f"target_day must be one of {sorted(ALLOWED_DAYS)}")

            val = slots.get("value")
            if _is_nullish(val):
                errors.append("value is required when refine_type=SWAP_DAY")
            elif str(val).strip().upper() != SWAP_VALUE:
                errors.append("value must be BEST_FIT when refine_type=SWAP_DAY")

            # Normalize value to canonical
            if not _is_nullish(slots.get("value")):
                slots["value"] = SWAP_VALUE

        elif rt in ("ADD_AVOID_ITEM", "REMOVE_AVOID_ITEM"):
            # target_day should be null (minimal spec)
            if not _is_nullish(slots.get("target_day")):
                errors.append("target_day must be null when refine_type is ADD/REMOVE_AVOID_ITEM")

            val = _lower(slots.get("value"))
            if _is_nullish(val):
                errors.append("value is required for ADD/REMOVE_AVOID_ITEM")
            elif val not in ALLOWED_AVOID_ITEMS:
                errors.append(
                    f"value must be one of {sorted(ALLOWED_AVOID_ITEMS)} for ADD/REMOVE_AVOID_ITEM"
                )
            else:
                slots["value"] = val  # canonical lowercase

    valid = len(errors) == 0
    return ValidationResult(valid=valid, errors=errors, normalized_mr=nm)


def slots_to_fill(intent: str, slots: Dict[str, Any]) -> List[str]:
    """
    Utility for a DM: given an intent+slots, return which required slots are missing.
    For refine, respects refine_type-dependent requirements.
    """
    intent = (intent or "").strip()
    if intent not in INTENT_SLOTS:
        return []

    missing: List[str] = []

    base_required = REQUIRED_SLOTS.get(intent, [])
    for req in base_required:
        if _is_nullish(slots.get(req)):
            missing.append(req)

    if intent == "refine":
        rt = _upper(slots.get("refine_type"))
        if rt == "SWAP_DAY":
            if _is_nullish(slots.get("target_day")):
                missing.append("target_day")
            if _is_nullish(slots.get("value")):
                missing.append("value")

    # Unique, deterministic order
    seen = set()
    out: List[str] = []
    for m in missing:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out
