"""
Meal Kit Composer — deterministic policy layer (DM guard rails + intent routing).

Purpose:
- Enforce hard workflow constraints (slot-filling, menu generation/selection gates, pending-action rules).
- Deterministically map certain user intents to actions (help/select_menu/show_week/inspect/refine/confirm).
- Otherwise accept the DM proposal after sanitizing (action, argument).

I/O:
- Input: (tracker, normalized MR, DM-proposed action+argument)
- Output: (final_action, final_argument, debug_info)

Notes:
- Arguments are plain strings; multi-arg actions use comma-separated values:
  provide_info: "intent, slot" ; update_avoid: "OP, value".
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


# =============================================================================
# Constants and controlled vocab
# =============================================================================

ALLOWED_DM_ACTIONS = {
    "request_info",
    "provide_info",
    "propose_menus",
    "set_active_menu",
    "show_day",
    "show_week",
    "suggest_swap_day",
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
    "yes_no_swap",
    "all",
}

PROVIDE_INFO_INTENTS = {"plan", "select_menu", "inspect", "refine", "confirm", "show_week"}

# =============================================================================
# Argument encoding helpers (comma-separated args)
# =============================================================================

def _split_args(arg_str: str) -> List[str]:
    if arg_str is None:
        return []
    s = str(arg_str).strip()
    if not s:
        return []
    return [a.strip() for a in s.split(",") if a.strip()]


def _join_args(args: List[str]) -> str:
    return ", ".join([str(a).strip() for a in args if str(a).strip()])

# =============================================================================
# MR interpretation helpers (e.g., suggest vs commit)
# =============================================================================

def _mr_requests_suggestion(intent: str, slots: Dict[str, Any]) -> bool:
    """
    True if the MR indicates a non-committing 'suggest an alternative' request
    for refine_type=SWAP_DAY.

    Primary signal (new standard): slots['mode'] == "SUGGEST"
    Backward-compatible fallback: value contains "SUGGEST"/"ALTERNATIVE"/"PROPOSE"
    """
    if intent != "refine":
        return False

    refine_type = str(slots.get("refine_type") or "").strip().upper()
    if refine_type != "SWAP_DAY":
        return False

    mode = slots.get("mode") or slots.get("swap_mode")
    m_str = str(mode).strip().upper() if not is_nullish(mode) else ""

    # New standard
    if m_str == "SUGGEST":
        return True

    # Backward-compatible fallback (older NLU behavior)
    v = slots.get("value")
    v_str = str(v).strip().upper() if not is_nullish(v) else ""
    return v_str in {"SUGGEST", "ALTERNATIVE", "PROPOSE"}


# =============================================================================
# Policy entrypoint
# =============================================================================

def apply_policy(
    tracker: Any,
    mr: Dict[str, Any],
    proposed_action: str,
    proposed_argument: str = "",
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Policy entrypoint:
    - Normalizes MR and enforces hard workflow constraints.
    - Deterministically routes certain intents to actions (help/select_menu/show_week/inspect/refine/confirm).
    - Otherwise accepts the DM proposal after sanitization.
    """

    nm = normalize_mr(mr)
    intent = nm.get("intent", "out_of_domain")
    slots = nm.get("slots", {}) or {}

    # --- HELP intent: route to provide_info (must be early, before plan/menu gates) ---
    if intent == "help":
        req_slot = slots.get("help_slot")
        req_intent = slots.get("help_intent")

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
    awaiting_slot = getattr(tracker, "awaiting_slot", None)
    awaiting_slot = str(awaiting_slot).strip() if not is_nullish(awaiting_slot) else ""


    #------ Post confirmation behavior------

    # If the plan is already confirmed and the user confirms again (e.g., "finalize", "done"),
    # do not restart the workflow; just acknowledge closure.
    pending = getattr(tracker, "pending_action", None)

    if phase == "CONFIRMED" and intent == "confirm" and not pending:
        return _final("fallback", "", nm, proposed_action, proposed_argument, "already_confirmed")

    # --- CONTINUE gate safety net
    ptype = str((pending or {}).get("type") or "").strip().upper() if isinstance(pending, dict) else ""
    if ptype == "CONTINUE_DEFERRED":
        return _final("fallback", "", nm, proposed_action, proposed_argument, "continue_gate_block")

    has_active = getattr(tracker, "has_active_menu", lambda: False)()


    # 0) Out-of-domain handling
    # In mid-workflow states, the user may produce acknowledgements or short commands
    # that the NLU cannot reliably classify (e.g., "ok", "go ahead", "finalize").
    # If the DM proposes a valid action, allow it to pass through guard rails.
    if intent == "out_of_domain":
        ood_type = str(slots.get("ood_type") or "").strip().upper()
        if ood_type == "REFUSE_PENDING":
            return _final(
                "fallback",
                "",
                nm,
                proposed_action,
                proposed_argument,
                "ood_refuse_pending->fallback",
            )
        
        if ood_type == "INVALID_ANSWER":
            # Always reprompt the awaited slot deterministically (except in CONFIRMED).
            if phase != "CONFIRMED" and awaiting_slot in REQUESTABLE_SLOTS:
                return _final(
                    "request_info",
                    awaiting_slot,
                    nm,
                    proposed_action,
                    proposed_argument,
                    f"ood_invalid_answer->reprompt_awaiting:{awaiting_slot}",
                )
            # If we are waiting for menu selection but awaiting_slot isn't set, reprompt menu_id.
            if phase == "AWAITING_MENU_SELECTION" and not has_active:
                return _final(
                    "request_info",
                    "menu_id",
                    nm,
                    proposed_action,
                    proposed_argument,
                    "ood_invalid_answer->reprompt_menu_id",
                )
            return _final(
                "fallback",
                "",
                nm,
                proposed_action,
                proposed_argument,
                "ood_invalid_answer->fallback",
            )


        if phase in {"AWAITING_MENU_SELECTION", "ACTIVE_MENU", "CONFIRMED"} and proposed_action in ALLOWED_DM_ACTIONS and proposed_action != "fallback":
            # Continue to normal guard rails below.
            pass
        else:
            # If we are awaiting a specific slot (slot-filling context), do not drop to generic fallback.
            # Reprompt the same slot deterministically (except in CONFIRMED).
            if phase != "CONFIRMED" and awaiting_slot in REQUESTABLE_SLOTS:
                return _final(
                    "request_info",
                    awaiting_slot,
                    nm,
                    proposed_action,
                    proposed_argument,
                    f"intent=out_of_domain->reprompt_awaiting:{awaiting_slot}",
                )

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
    

    # 2) Menu availability check
    if hasattr(tracker, "menus_exist"):
        menus_exist = tracker.menus_exist()
    else:
        menus = getattr(tracker, "menus", None)
        menus_exist = bool(menus) and menus.get("1") is not None and menus.get("2") is not None

    # 2.1) If PLAN is complete and menus are not yet generated, force proposing menus.
    # This prevents the DM from re-asking already-filled plan slots due to per-turn MR nulls.
    if not menus_exist and not missing_plan and phase == "AWAITING_PLAN":
        return _final(
            "propose_menus",
            "",
            nm,
            proposed_action,
            proposed_argument,
            "plan_complete->propose_menus",
        )
    
    # --- SELECT_MENU must deterministically activate the chosen menu ---
    if intent == "select_menu":
        # If menus are not available yet, generate them first
        if not menus_exist:
            return _final(
                "propose_menus",
                "",
                nm,
                proposed_action,
                proposed_argument,
                "select_menu->need_menus",
            )

        mid = slots.get("menu_id")

        # Coerce to int if possible
        try:
            mid = int(mid) if not is_nullish(mid) else None
        except Exception:
            mid = None

        # Valid selection -> commit it (ignore DM proposal)
        if mid in (1, 2):
            return _final(
                "set_active_menu",
                str(mid),
                nm,
                proposed_action,
                proposed_argument,
                "select_menu->set_active_menu",
            )

        # Missing/invalid menu_id -> reprompt
        return _final(
            "request_info",
            "menu_id",
            nm,
            proposed_action,
            proposed_argument,
            "select_menu_missing_or_invalid_menu_id",
        )



    menu_dependent_actions = {"set_active_menu", "show_day", "show_week", "suggest_swap_day","swap_day", "update_avoid", "confirm_plan"}

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

    # 2.2) show_week intent routing (deterministic)
    if intent == "show_week":
        if missing_plan:
            # (This will usually be handled by the earlier missing_plan gate,
            # but keeping it here makes the intent self-contained.)
            return _final("request_info", missing_plan[0], nm, proposed_action, proposed_argument, "show_week->missing_plan")

        if not menus_exist:
            return _final("propose_menus", "", nm, proposed_action, proposed_argument, "show_week->need_menus")

        if not has_active:
            return _final("request_info", "menu_id", nm, proposed_action, proposed_argument, "show_week->need_active_menu")

        return _final("show_week", "", nm, proposed_action, proposed_argument, "show_week->ok")


    # 3) Menu selection gate: must have an active menu before show/swap/update/confirm
    if not has_active and intent in {"inspect", "refine", "confirm", "show_week"}:
        return _final("request_info", "menu_id", nm, proposed_action, proposed_argument, "menu_gate_intent")


    # 4) Pending suggestion confirmation routing:
    # If there is a pending suggested SWAP_DAY and the user confirms (e.g., "yes/ok/do it"),
    # force a commit of that pending swap rather than generating a shopping list or other action.
    pending = getattr(tracker, "pending_action", None)
    if pending and intent == "confirm":
        p_type = str((pending or {}).get("type") or "").strip().upper()
        p_day = (pending or {}).get("day")
        if p_type == "SWAP_DAY" and p_day in ALLOWED_DAYS:
            # Commit the pending suggestion deterministically.
            return _final(
                "swap_day",
                str(p_day),
                nm,
                proposed_action,
                proposed_argument,
                "pending_swap_confirm->commit",
            )

    if intent == "confirm":
        if missing_plan:
            return _final(
                "request_info",
                missing_plan[0],
                nm,
                proposed_action,
                proposed_argument,
                "confirm->missing_plan",
            )
        if not menus_exist:
            return _final(
                "propose_menus",
                "",
                nm,
                proposed_action,
                proposed_argument,
                "confirm->need_menus",
            )
        if not has_active:
            return _final(
                "request_info",
                "menu_id",
                nm,
                proposed_action,
                proposed_argument,
                "confirm->need_active_menu",
            )
        return _final(
            "confirm_plan",
            "",
            nm,
            proposed_action,
            proposed_argument,
            "confirm->confirm_plan",
        )

    # 5) Intent-to-action routing (inspect/refine/confirm/show_week)
    # Prevent DM from re-asking for slots that are already present in the MR.
    if intent == "inspect":
        day = slots.get("target_day")
        day = normalize_day(day)  # ensure "wednesday"/"Wed"/typos normalize
        if day in ALLOWED_DAYS:
            return _final(
                "show_day",
                str(day),
                nm,
                proposed_action,
                proposed_argument,
                "inspect->show_day",
            )
        return _final(
            "request_info",
            "target_day",
            nm,
            proposed_action,
            proposed_argument,
            "inspect_missing_day->request_target_day",
        )

    if intent == "refine":
        rtype = str(slots.get("refine_type") or "").strip().upper()

        if rtype == "SWAP_DAY":
            day = slots.get("target_day")
            day = normalize_day(day)

            if day not in ALLOWED_DAYS:
                return _final(
                    "request_info",
                    "target_day",
                    nm,
                    proposed_action,
                    proposed_argument,
                    "refine_swap_missing_day->request_target_day",
                )

            # Suggest vs commit decided by the MR, not by the DM proposal
            if _mr_requests_suggestion(intent, slots):
                return _final(
                    "suggest_swap_day",
                    str(day),
                    nm,
                    proposed_action,
                    proposed_argument,
                    "refine_swap->suggest_swap_day",
                )
            return _final(
                "swap_day",
                str(day),
                nm,
                proposed_action,
                proposed_argument,
                "refine_swap->swap_day",
            )
        elif rtype in {"ADD_AVOID_ITEM", "REMOVE_AVOID_ITEM"}:
            val = slots.get("value")
            val = str(val).strip().lower() if not is_nullish(val) else ""

            if not val:
                return _final(
                    "request_info",
                    "value",
                    nm,
                    proposed_action,
                    proposed_argument,
                    "refine_avoid_missing_value->request_value",
                )

            if val not in ALLOWED_AVOID_ITEMS:
                return _final(
                    "request_info",
                    "value",
                    nm,
                    proposed_action,
                    proposed_argument,
                    "refine_avoid_invalid_value->request_value",
                )

            # Deterministically execute avoid update (ignore DM proposal)
            return _final(
                "update_avoid",
                _join_args([rtype, val]),
                nm,
                proposed_action,
                proposed_argument,
                "refine_avoid->update_avoid",
            )


    # Non-committing suggestion guardrail:
    # If MR indicates the user asked for an alternative suggestion (not an applied swap),
    # rewrite swap_day -> suggest_swap_day deterministically.
    if _mr_requests_suggestion(intent, slots):
        if (proposed_action or "").strip() == "swap_day":
            proposed_action = "suggest_swap_day"

    
    # 6) Accept/sanitize DM proposal
    safe_action, safe_arg = sanitize_proposed_action(proposed_action, proposed_argument)

    # If sanitization collapses to fallback while we are awaiting a slot, reprompt that slot deterministically
    # (except in CONFIRMED, where we keep the closure behavior).
    if safe_action == "fallback" and phase != "CONFIRMED" and awaiting_slot in REQUESTABLE_SLOTS:
        return _final(
            "request_info",
            awaiting_slot,
            nm,
            proposed_action,
            proposed_argument,
            f"sanitize_fallback->reprompt_awaiting:{awaiting_slot}",
        )

    return _final(safe_action, safe_arg, nm, proposed_action, proposed_argument, "guardrail_accept_or_sanitize")


# =============================================================================
# DM proposal sanitization
# =============================================================================

def sanitize_proposed_action(proposed_action: str, proposed_argument: str,) -> Tuple[str, str]:
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

    if a in {"show_day", "suggest_swap_day", "swap_day"}:
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

    if a in {"propose_menus", "show_week", "confirm_plan", "fallback"}:
        return a, ""

    return "fallback", ""

# =============================================================================
# Return packaging / debug metadata
# =============================================================================

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
