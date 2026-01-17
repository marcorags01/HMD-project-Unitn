# policy.py
"""
Meal Kit Composer — deterministic DM guard rails (policy layer).

Purpose:
- Take (tracker, user MR, proposed_action, proposed_argument)
- Enforce hard workflow rules deterministically
- Return a safe (action, argument, debug_info)

This module keeps DM lightweight (LLM picks an action), while policy guarantees:
- missing PLAN slots are requested
- menu selection gate before inspect/refine/confirm
- refine constraints (SWAP_DAY requires day; avoid updates require value)
- robust fallbacks for invalid/unsafe actions

Action set (must match DM prompt + NLG expectations):
- request_info(slot)
- provide_info(intent, slot)
- propose_menus()
- set_active_menu(menu_id)
- show_day(target_day)
- swap_day(target_day)
- update_avoid(op, value)
- confirm_plan()
- fallback()
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from support_classes import (
    ALLOWED_DAYS,
    ALLOWED_AVOID_ITEMS,
    is_nullish,
    normalize_day,
)
from intents_schema import normalize_mr


ALLOWED_DM_ACTIONS = {
    "request_info",
    "provide_info",
    "propose_menus",
    "set_active_menu",
    "show_day",
    "swap_day",
    "update_avoid",
    "confirm_plan",
    "fallback",
}

REQUESTABLE_SLOTS = {
    "servings",
    "time_limit",
    "calorie_level",
    "avoid_items",
    "menu_id",
    "target_day",
    "refine_type",
    "value",
    "all",
}

PROVIDE_INFO_INTENTS = {"plan", "select_menu", "inspect", "refine", "confirm"}


def _split_args(arg_str: str) -> List[str]:
    if arg_str is None:
        return []
    s = str(arg_str).strip()
    if not s:
        return []
    return [a.strip() for a in s.split(",") if a.strip()]


def _join_args(args: List[str]) -> str:
    return ", ".join([str(a).strip() for a in args if str(a).strip()])


def apply_policy(
    tracker: Any,
    mr: Dict[str, Any],
    proposed_action: str,
    proposed_argument: str = "",
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Guardrail-only policy:
    - Enforce hard preconditions (plan completeness, menus existence, menu selection gate)
    - Otherwise accept DM proposal (sanitized)
    """
    nm = normalize_mr(mr)
    intent = nm.get("intent", "out_of_domain")
    slots = nm.get("slots", {}) or {}

    # --- HELP intent: route to provide_info (must be early, before plan/menu gates) ---
    if intent == "help":
        req_slot = slots.get("slot")
        req_intent = slots.get("intent")

        if is_nullish(req_slot):
            req_slot = "all"
        req_slot = str(req_slot).strip()

        if is_nullish(req_intent) or str(req_intent).strip() not in PROVIDE_INFO_INTENTS:
            req_intent = "plan"
        req_intent = str(req_intent).strip()

        if req_slot not in REQUESTABLE_SLOTS:
            req_slot = "all"

        return _final(
            "provide_info",
            _join_args([req_intent, req_slot]),
            nm,
            proposed_action,
            proposed_argument,
            "help->provide_info",
        )

    phase = getattr(tracker, "phase", "")

    #--Post confirmation behavior--
    # If the plan is already confirmed and the user confirms again (e.g., "finalize", "done"),
    # do not restart the workflow; just acknowledge closure.
    if phase == "CONFIRMED" and intent == "confirm":
        return _final("fallback", "", nm, proposed_action, proposed_argument, "already_confirmed")
    has_active = getattr(tracker, "has_active_menu", lambda: False)()

    # 0) Out-of-domain handling
    # In mid-workflow states, the user may produce acknowledgements or short commands
    # that the NLU cannot reliably classify (e.g., "ok", "go ahead", "finalize").
    # If the DM proposes a valid action, allow it to pass through guard rails.
    if intent == "out_of_domain":
        if phase in {"AWAITING_MENU_SELECTION", "ACTIVE_MENU", "CONFIRMED"} and proposed_action in ALLOWED_DM_ACTIONS and proposed_action != "fallback":
            # Continue to normal guard rails below.
            pass
        else:
            return _final("fallback", "", nm, proposed_action, proposed_argument, "intent=out_of_domain")

    # 1) PLAN slot completion gate (hard rule, but let DM choose WHICH missing field to ask next)
    missing_plan: List[str] = []
    if hasattr(tracker, "missing_plan_slots"):
        missing_plan = tracker.missing_plan_slots() or []

    if missing_plan:
        pa = (proposed_action or "").strip()
        parg = (proposed_argument or "").strip()

        # If DM is already asking for one missing field, allow it (smoother ordering)
        if pa == "request_info" and parg in missing_plan:
            return _final(
                "request_info",
                parg,
                nm,
                proposed_action,
                proposed_argument,
                f"missing_plan_slot=dm_selected:{parg}",
            )

        # Otherwise, force a request for ONE missing field (deterministic fallback)
        return _final(
            "request_info",
            missing_plan[0],
            nm,
            proposed_action,
            proposed_argument,
            f"missing_plan_slot=forced:{missing_plan[0]}",
        )
    

    # 2) Menus must exist before menu-dependent actions.
    # Do NOT use phase as proxy; check actual menu availability.
    menus_exist = bool(getattr(tracker, "menus", None))

    menu_dependent_actions = {"set_active_menu", "show_day", "swap_day", "update_avoid", "confirm_plan"}

    if not menus_exist:
        # If DM tries anything that implies menus exist, force propose_menus()
        if proposed_action in menu_dependent_actions:
            return _final(
                "propose_menus",
                "",
                nm,
                proposed_action,
                proposed_argument,
                "need_menus_before_menu_ops",
            )

        # Avoid asking for menu_id before menus exist (confusing UX)
        if (proposed_action or "").strip() == "request_info" and (proposed_argument or "").strip() == "menu_id":
            return _final(
                "propose_menus",
                "",
                nm,
                proposed_action,
                proposed_argument,
                "need_menus_before_menu_id",
            )


    # 3) Menu selection gate: must have an active menu before show/swap/update/confirm
    active_required_actions = {"show_day", "swap_day", "update_avoid", "confirm_plan"}
    if not has_active and proposed_action in active_required_actions:
        return _final("request_info", "menu_id", nm, proposed_action, proposed_argument, "menu_gate_action")

    # 4) Otherwise: accept DM proposal, just sanitize action/args minimally
    safe_action, safe_arg = sanitize_proposed_action(tracker, intent, slots, proposed_action, proposed_argument)
    return _final(safe_action, safe_arg, nm, proposed_action, proposed_argument, "guardrail_accept_or_sanitize")


def sanitize_proposed_action(
    tracker: Any,
    intent: str,
    slots: Dict[str, Any],
    proposed_action: str,
    proposed_argument: str,
) -> Tuple[str, str]:
    a = (proposed_action or "").strip()
    arg = (proposed_argument or "").strip()

    if is_nullish(arg):
        arg = ""  # canonicalize missing DM argument

    if a not in ALLOWED_DM_ACTIONS:
        return "fallback", ""

    if a == "request_info":
        if not arg:
            return "fallback", ""
        if arg not in REQUESTABLE_SLOTS:
            return "fallback", ""
        return a, arg

    if a == "provide_info":
        parts = _split_args(arg)
        if len(parts) != 2 or is_nullish(parts[0]) or is_nullish(parts[1]):
            return "fallback", ""
        if parts[0] not in PROVIDE_INFO_INTENTS or parts[1] not in REQUESTABLE_SLOTS:
            return "fallback", ""
        return a, _join_args(parts)

    if a == "set_active_menu":
        try:
            mid = int(_split_args(arg)[0] if "," in arg else arg)
        except Exception:
            return "request_info", "menu_id"
        return ("set_active_menu", str(mid)) if mid in (1, 2) else ("request_info", "menu_id")

    if a in {"show_day", "swap_day"}:
        day = normalize_day(arg)
        return (a, day) if day in ALLOWED_DAYS else ("request_info", "target_day")
   
    if a == "update_avoid":
        parts = _split_args(arg)
        if len(parts) != 2 or is_nullish(parts[0]) or is_nullish(parts[1]):
            return "request_info", "value"
        op = parts[0].strip().upper()
        val = parts[1].strip().lower()
        if op not in {"ADD_AVOID_ITEM", "REMOVE_AVOID_ITEM"}:
            return "fallback", ""
        if val not in ALLOWED_AVOID_ITEMS:
            return "request_info", "value"
        return "update_avoid", _join_args([op, val])

    if a in {"propose_menus", "confirm_plan", "fallback"}:
        return a, ""

    return "fallback", ""



def _final(
    action: str,
    argument: str,
    normalized_mr: Dict[str, Any],
    proposed_action: str,
    proposed_argument: str,
    reason: str,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Standardize return + include debug metadata for logging.
    """
    debug = {
        "policy_reason": reason,
        "normalized_mr": normalized_mr,
        "proposed": {"action": proposed_action, "argument": proposed_argument},
        "final": {"action": action, "argument": argument},
    }
    return action, argument, debug
