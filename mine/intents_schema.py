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
    is_nullish,
    normalize_day,
    normalize_avoid_items,
    normalize_upper_enum,
)


# ------------------------- Intent + slot inventory -------------------------

INTENT_SLOTS: Dict[str, List[str]] = {
    "plan": ["servings", "time_limit", "calorie_level", "avoid_items"],
    "select_menu": ["menu_id"],
    "inspect": ["target_day"],
    "refine": ["refine_type", "target_day", "value", "mode"],
    "confirm": [],
    "show_week": [],
    "help": ["intent", "slot"],
    "out_of_domain": [],
    
}

# Required slots per intent (minimal configuration)
REQUIRED_SLOTS: Dict[str, List[str]] = {
    "plan": ["servings", "time_limit", "calorie_level", "avoid_items"],
    "select_menu": ["menu_id"],
    "inspect": ["target_day"],
    "refine": ["refine_type", "value"],  # target_day depends on refine_type
    "confirm": [],
    "show_week": [],
    "help": ["intent"],
    "out_of_domain": [],
}

# refine_type controlled vocab + refine-specific constraints
REFINE_TYPES = {"SWAP_DAY", "ADD_AVOID_ITEM", "REMOVE_AVOID_ITEM"}
SWAP_VALUE = "BEST_FIT"
REFINE_MODES = {"SUGGEST", "COMMIT"}



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

def _upper(x: Any) -> Optional[str]:
    return normalize_upper_enum(x)


def _lower(x: Any) -> Optional[str]:
    if is_nullish(x):
        return None
    return str(x).strip().lower()


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
        if not is_nullish(slots_out.get("servings")):
            try:
                slots_out["servings"] = int(slots_out["servings"])
            except Exception:
                pass
        # enums
        slots_out["time_limit"] = _upper(slots_out.get("time_limit"))
        slots_out["calorie_level"] = _upper(slots_out.get("calorie_level"))
        # avoid list
        avoid_norm = normalize_avoid_items(slots_out.get("avoid_items"))

        # If user explicitly means "no avoids", canonicalize to empty list []
        if avoid_norm is None:
            raw = slots_out.get("avoid_items")
            if isinstance(raw, str) and raw.strip().lower() in {"none", "no", "nothing", "nope", "n/a", "na"}:
                avoid_norm = []

        slots_out["avoid_items"] = avoid_norm

    elif intent == "select_menu":
        if not is_nullish(slots_out.get("menu_id")):
            try:
                slots_out["menu_id"] = int(slots_out["menu_id"])
            except Exception:
                pass

    elif intent == "inspect":
        slots_out["target_day"] = normalize_day(slots_out.get("target_day"))

    elif intent == "refine":
        slots_out["refine_type"] = _upper(slots_out.get("refine_type"))
        slots_out["target_day"] = normalize_day(slots_out.get("target_day"))
        slots_out["mode"] = _upper(slots_out.get("mode"))  # NEW
        # value: keep raw string for now; validation will interpret based on refine_type
        if not is_nullish(slots_out.get("value")):
            slots_out["value"] = str(slots_out["value"]).strip()


    elif intent == "help":
        slots_out["intent"] = _lower(slots_out.get("intent"))  # keep lowercase for matching keys
        if not is_nullish(slots_out.get("slot")):
            slots_out["slot"] = str(slots_out["slot"]).strip()
        else:
            slots_out["slot"] = None

    elif intent == "show_week":
     pass


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
        if is_nullish(slots.get(req)):
            errors.append(f"Missing required slot: {req}")

    # Intent-specific validation
    if intent == "plan":
        # servings range
        s = slots.get("servings")
        if not is_nullish(s):
            if not isinstance(s, int) or not (1 <= s <= 6):
                errors.append("servings must be an int in range 1–6")

        tl = slots.get("time_limit")
        if not is_nullish(tl) and tl not in ALLOWED_TIME_LIMITS:
            errors.append(f"time_limit must be one of {sorted(ALLOWED_TIME_LIMITS)}")

        cal = slots.get("calorie_level")
        if not is_nullish(cal) and cal not in ALLOWED_CALORIE_LEVELS:
            errors.append(f"calorie_level must be one of {sorted(ALLOWED_CALORIE_LEVELS)}")

        avoid = slots.get("avoid_items")
        if not is_nullish(avoid):
            if not isinstance(avoid, list):
                errors.append("avoid_items must be a list of strings")
            else:
                non_str = [a for a in avoid if not isinstance(a, str)]
                if non_str:
                    errors.append("avoid_items must be a list of strings")
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
        if not is_nullish(mid):
            if mid not in (1, 2):
                errors.append("menu_id must be 1 or 2")

    elif intent == "inspect":
        day = slots.get("target_day")
        if not is_nullish(day) and day not in ALLOWED_DAYS:
            errors.append(f"target_day must be one of {sorted(ALLOWED_DAYS)}")

    elif intent == "refine":
        rt = slots.get("refine_type")
        if not is_nullish(rt) and rt not in REFINE_TYPES:
            errors.append(f"refine_type must be one of {sorted(REFINE_TYPES)}")

        mode = slots.get("mode")
        if not is_nullish(mode) and mode not in REFINE_MODES:
            errors.append(f"mode must be one of {sorted(REFINE_MODES)}")
            slots["mode"] = None
            
        # refine_type-dependent rules
        if rt == "SWAP_DAY":
            day = slots.get("target_day")
            if is_nullish(day):
                errors.append("target_day is required when refine_type=SWAP_DAY")
            elif day not in ALLOWED_DAYS:
                errors.append(f"target_day must be one of {sorted(ALLOWED_DAYS)}")

            val = slots.get("value")
            if is_nullish(val):
                errors.append("value is required when refine_type=SWAP_DAY")
            elif str(val).strip().upper() != SWAP_VALUE:
                errors.append("value must be BEST_FIT when refine_type=SWAP_DAY")

            # Normalize value to canonical
            if not is_nullish(slots.get("value")):
                slots["value"] = SWAP_VALUE

        elif rt in ("ADD_AVOID_ITEM", "REMOVE_AVOID_ITEM"):
            # target_day should be null (minimal spec)
            if not is_nullish(slots.get("target_day")):
                errors.append("target_day must be null when refine_type is ADD/REMOVE_AVOID_ITEM")

            val = _lower(slots.get("value"))
            if is_nullish(val):
                errors.append("value is required for ADD/REMOVE_AVOID_ITEM")
            elif val not in ALLOWED_AVOID_ITEMS:
                errors.append(
                    f"value must be one of {sorted(ALLOWED_AVOID_ITEMS)} for ADD/REMOVE_AVOID_ITEM"
                )
            else:
                slots["value"] = val  # canonical lowercase

    elif intent == "help":
        # intent: required; must be a known intent name
        hi = slots.get("intent")
        if not is_nullish(hi):
            hi_norm = str(hi).strip().lower()
            if hi_norm not in INTENT_SLOTS:
                errors.append(f"help.intent must be one of {sorted(INTENT_SLOTS.keys())}")
            else:
                slots["intent"] = hi_norm  # canonical

                # slot: optional; if provided must be a valid slot name for that intent
                hs = slots.get("slot")
                if not is_nullish(hs):
                    hs_norm = str(hs).strip()
                    if hs_norm not in INTENT_SLOTS[hi_norm]:
                        errors.append(
                            f"help.slot must be one of {sorted(INTENT_SLOTS[hi_norm])} for intent={hi_norm}"
                        )
                    else:
                        slots["slot"] = hs_norm  # canonical

    elif intent == "show_week":
     pass


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
        if is_nullish(slots.get(req)):
            missing.append(req)

    if intent == "refine":
        rt = _upper(slots.get("refine_type"))
        if rt == "SWAP_DAY":
            if is_nullish(slots.get("target_day")):
                missing.append("target_day")
            if is_nullish(slots.get("value")):
                missing.append("value")

    # Unique, deterministic order
    seen = set()
    out: List[str] = []
    for m in missing:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def missing_plan_slots_from_constraints(constraints: Dict[str, Any]) -> List[str]:
    """
    Authoritative definition of plan completeness.

    Required for plan:
      - servings
      - time_limit
      - calorie_level
      - avoid_items  (empty list [] is valid; null is missing)
    """
    required = ["servings", "time_limit", "calorie_level", "avoid_items"]
    missing: List[str] = []
    for k in required:
        if is_nullish(constraints.get(k)):
            missing.append(k)
    return missing