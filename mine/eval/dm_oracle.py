"""Oracle DM policy used for intrinsic DM evaluation.

This is NOT used by the runtime system. It exists only to provide a stable
reference mapping from (tracker state, selected MR) -> (expected action, arg)
for action-classification evaluation as suggested in the course material.

Design goals:
- Deterministic and lightweight.
- Mirrors the intended workflow gates (slot-filling, menu selection gating,
  swap suggestion/commit) without calling the LLM DM.
- Produces actions from the system's closed action set.
"""

from __future__ import annotations

from typing import Dict, Tuple, Any

from support_classes import Tracker


def oracle_dm_action(tracker: Tracker, mr: Dict[str, Any]) -> Tuple[str, str]:
    """Return (action, arg) as the intended next system step.

    Notes
    - This oracle does not attempt to generate domain-specific payloads.
    - It assumes MR has already been validated/normalized (intents_schema.validate_mr).
    """
    intent = str((mr or {}).get("intent", "")).strip()
    slots = (mr or {}).get("slots") or {}
    if not isinstance(slots, dict):
        slots = {}

    # -------- Hard workflow gates (spec-level) --------

    # 1) If we still miss plan slots, we must ask the next missing slot
    missing = tracker.missing_plan_slots()
    if missing:
        return "request_info", str(missing[0])

    # 2) If we don't have menus yet, we must propose them
    if tracker.phase == "AWAITING_PLAN" and not tracker.menus_exist():
        return "propose_menus", ""

    # 3) If menus exist but no active menu, require selection
    if (tracker.phase == "AWAITING_MENU_SELECTION") or (tracker.menus_exist() and not tracker.has_active_menu()):
        if intent == "select_menu" and "menu_id" in slots:
            return "set_active_menu", str(slots["menu_id"])
        return "request_info", "menu_id"

    # 4) If a SWAP_DAY suggestion is pending, the next "commit" should swap that day
    pending = getattr(tracker, "pending_action", None)
    if isinstance(pending, dict) and pending.get("type") == "SWAP_DAY":
        # If the user requests a commit (Swap X please), do the swap
        if intent == "refine":
            refine_type = str(slots.get("refine_type", "")).strip().upper()
            mode = str(slots.get("mode", "")).strip().upper()
            if refine_type == "SWAP_DAY" and mode in {"COMMIT", "CONFIRM"}:
                day = str(slots.get("target_day", "") or pending.get("day", "")).strip()
                if day:
                    return "swap_day", day

    # -------- Intent-to-action mapping --------

    if intent == "inspect":
        day = str(slots.get("target_day", "")).strip()
        if not day:
            return "request_info", "target_day"
        return "show_day", day

    if intent == "show_week":
        return "show_week", ""

    if intent == "confirm":
        # If no active menu, the selection gate above should have fired.
        return "confirm_plan", ""

    if intent == "help":
        # Runtime "help" is handled via policy -> provide_info. Keep as request_info(help)
        return "provide_info", ""

    if intent == "refine":
        refine_type = str(slots.get("refine_type", "")).strip().upper()
        mode = str(slots.get("mode", "")).strip().upper()  # SUGGEST/COMMIT (optional)
        if refine_type == "SWAP_DAY":
            day = str(slots.get("target_day", "")).strip()
            if not day:
                return "request_info", "target_day"
            if mode == "SUGGEST":
                return "suggest_swap_day", day
            # Default: commit
            return "swap_day", day

        if refine_type in {"ADD_AVOID_ITEM", "REMOVE_AVOID_ITEM"}:
            val = str(slots.get("value", "")).strip()
            if not val:
                return "request_info", "avoid_items"
            # Encode into arg string as runtime expects: "ADD_AVOID_ITEM, egg"
            return "update_avoid", f"{refine_type}, {val}"

        # Unknown refine -> fallback
        return "fallback", ""

    # Plan intent with no missing slots at this point usually means "regenerate menus"
    if intent == "plan":
        # If user changed constraints after menus exist, you might propose new menus.
        # This is the safest default.
        return "propose_menus", ""

    # Anything else
    return "fallback", ""
