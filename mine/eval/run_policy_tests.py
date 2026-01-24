"""Deterministic unit tests for policy.py.

Run:
  python -m eval.run_policy_tests

These tests require no LLM.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from support_classes import Tracker
from policy import apply_policy


def _mk_tracker(phase: str = "AWAITING_PLAN") -> Tracker:
    t = Tracker()
    t.set_phase(phase)
    return t


def _assert(action: str, arg: str, got: Tuple[str, str, Dict[str, Any]], label: str) -> None:
    a, b, dbg = got
    assert (a, b) == (action, arg), f"{label}: expected {action}({arg}) got {a}({b}) reason={dbg.get('policy_reason')}"


def main() -> None:
    # 1) Slot-filling gate: empty plan -> ask first missing slot
    tr = _mk_tracker("AWAITING_PLAN")
    mr = {"intent": "plan", "slots": {}}
    _assert("request_info", "servings", apply_policy(tr, mr, "fallback", ""), "plan_empty")

    # 2) Complete plan -> propose menus
    tr = _mk_tracker("AWAITING_PLAN")
    tr.constraints.update({"servings": 2, "time_limit": "FAST", "calorie_level": "MED", "avoid_items": []})
    mr = {"intent": "plan", "slots": {"servings": 2, "time_limit": "FAST", "calorie_level": "MED", "avoid_items": []}}
    _assert("propose_menus", "", apply_policy(tr, mr, "fallback", ""), "plan_complete")

    # 3) Menu selection gate: inspect before selecting menu
    tr = _mk_tracker("AWAITING_MENU_SELECTION")
    tr.menus = {"1": {"Mon": "1"}, "2": {"Mon": "2"}}
    mr = {"intent": "inspect", "slots": {"target_day": "Mon"}}
    _assert("request_info", "menu_id", apply_policy(tr, mr, "show_day", "Mon"), "menu_gate_inspect")

    # 4) show_week requires active menu
    tr = _mk_tracker("ACTIVE_MENU")
    tr.menus = {"1": {"Mon": "1"}, "2": {"Mon": "2"}}
    tr.active_menu_id = 1
    tr.active_menu = {"Mon": "1"}
    mr = {"intent": "show_week", "slots": {}}
    _assert("show_week", "", apply_policy(tr, mr, "fallback", ""), "show_week_active")

    # 5) confirm without active menu -> request menu
    tr = _mk_tracker("AWAITING_MENU_SELECTION")
    tr.menus = {"1": {"Mon": "1"}, "2": {"Mon": "2"}}
    tr.constraints.update({"servings": 2, "time_limit": "FAST", "calorie_level": "MED", "avoid_items": []})
    mr = {"intent": "confirm", "slots": {}}
    _assert("request_info", "menu_id", apply_policy(tr, mr, "confirm_plan", ""), "confirm_need_active")

    # 6) refine swap suggestion -> suggest_swap_day
    tr = _mk_tracker("ACTIVE_MENU")
    tr.active_menu_id = 1
    tr.active_menu = {"Mon": "1", "Tue": "2", "Wed": "3", "Thu": "4", "Fri": "5"}
    mr = {"intent": "refine", "slots": {"refine_type": "SWAP_DAY", "target_day": "Tue", "value": "BEST_FIT", "mode": "SUGGEST"}}
    _assert("suggest_swap_day", "Tue", apply_policy(tr, mr, "fallback", ""), "refine_suggest")

    # 7) pending swap confirmation: confirm commits
    tr = _mk_tracker("ACTIVE_MENU")
    tr.active_menu_id = 1
    tr.active_menu = {"Mon": "1", "Tue": "2", "Wed": "3", "Thu": "4", "Fri": "5"}
    tr.pending_action = {"type": "SWAP_DAY", "day": "Tue", "recipe_id": "999"}
    mr = {"intent": "confirm", "slots": {}}
    _assert("swap_day", "Tue", apply_policy(tr, mr, "confirm_plan", ""), "pending_swap_confirm")

    print("All policy tests passed.")


if __name__ == "__main__":
    main()
