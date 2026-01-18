"""turn_planner.py

Meal Kit Composer — deterministic multi-step controller.

This module is designed to sit *between* NLU (which may output 1+ MRs)
and the existing pipeline components:

  policy.apply_policy(...)
  execute_action(...)
  nlg.NLG(...)

It provides two main entry points:

  - plan_steps(tracker, mrs) -> List[ProposedStep]
  - run_steps(tracker, steps, apply_policy, execute_action, nlg) -> TurnResult

Core design goals
-----------------
1) Deterministic ordering across multiple user intents in one message.
2) Preserve your existing intent logic (refine_type=SWAP_DAY with slots.mode).
3) Stop as soon as user input is required (request_info) or a menu choice is required
   (after propose_menus).
4) Keep confirm as the *final* step if feasible (your requirement).
5) Execute help late (your choice B): after feasible actions, but before confirm
   (so confirm remains final).

Notes
-----
- This module does NOT call the DM. It maps MRs to actions deterministically.
  Policy remains the authoritative guard-rail and may rewrite actions.
- Tracker mutation is performed by execute_action (plus tracker.apply_mrs elsewhere).

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from intents_schema import normalize_mr
from support_classes import is_nullish, normalize_day, normalize_upper_enum


# ------------------------------- Data types -------------------------------


@dataclass(frozen=True)
class ProposedStep:
    """A single planned step for the current user turn."""

    kind: str  # plan_gate | select_menu | refine_avoid | refine_swap | inspect | help | confirm
    mr: Dict[str, Any]
    proposed_action: str
    proposed_argument: str


@dataclass
class ExecutedStep:
    """A step after policy + execution."""

    step: ProposedStep
    final_action: str
    final_argument: str
    policy_debug: Dict[str, Any]
    payload: Dict[str, Any]
    rendered: str


@dataclass
class TurnResult:
    executed_steps: List[ExecutedStep]
    final_reply: str
    stop_reason: str  # completed | needs_user_input | needs_menu_selection


# --------------------------- Step planning logic --------------------------


def _stable_filter(mrs: List[Dict[str, Any]], pred: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
    return [m for m in mrs if pred(m)]


def _pick_last(mrs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return mrs[-1] if mrs else None


def _as_str(x: Any) -> str:
    return "" if is_nullish(x) else str(x).strip()


def _join_args(*parts: str) -> str:
    clean = [p.strip() for p in parts if p is not None and str(p).strip()]
    return ", ".join(clean)


def _propose_for_plan_gate(tracker: Any) -> Tuple[str, str]:
    """Propose the *next* deterministic action when a plan MR is present.

    We keep this very small and let policy remain authoritative.
    """
    missing = []
    if hasattr(tracker, "missing_plan_slots"):
        try:
            missing = tracker.missing_plan_slots() or []
        except Exception:
            missing = []

    if missing:
        # Ask the first missing slot (intents_schema.missing_plan_slots_from_constraints
        # already defines deterministic ordering).
        return "request_info", str(missing[0])

    # If plan is complete, suggest proposing menus (policy will also enforce this).
    # We only propose if menus do not exist yet.
    menus_exist = False
    if hasattr(tracker, "menus_exist"):
        try:
            menus_exist = bool(tracker.menus_exist())
        except Exception:
            menus_exist = False

    if not menus_exist:
        return "propose_menus", ""

    # Otherwise: nothing urgent to do.
    return "fallback", ""


def _propose_for_select_menu(mr: Dict[str, Any]) -> Tuple[str, str]:
    slots = mr.get("slots", {}) or {}
    mid = slots.get("menu_id")
    if is_nullish(mid):
        return "request_info", "menu_id"
    return "set_active_menu", _as_str(mid)


def _propose_for_refine_avoid(mr: Dict[str, Any]) -> Tuple[str, str]:
    slots = mr.get("slots", {}) or {}
    op = normalize_upper_enum(slots.get("refine_type")) or ""
    val = _as_str(slots.get("value")).lower()
    # Argument format expected by policy/executor: "ADD_AVOID_ITEM, nuts"
    return "update_avoid", _join_args(op, val)


def _propose_for_refine_swap(mr: Dict[str, Any]) -> Tuple[str, str]:
    slots = mr.get("slots", {}) or {}
    day = normalize_day(slots.get("target_day"))
    mode = normalize_upper_enum(slots.get("mode"))

    if not day:
        return "request_info", "target_day"

    # Your decision: keep refine_type=SWAP_DAY; use mode to distinguish.
    if mode == "SUGGEST":
        return "suggest_swap_day", day

    # Default to COMMIT when mode is missing/unknown.
    return "swap_day", day


def _propose_for_inspect(mr: Dict[str, Any]) -> Tuple[str, str]:
    slots = mr.get("slots", {}) or {}
    day = normalize_day(slots.get("target_day"))
    if not day:
        return "request_info", "target_day"
    return "show_day", day


def _propose_for_help(mr: Dict[str, Any]) -> Tuple[str, str]:
    slots = mr.get("slots", {}) or {}
    intent_req = _as_str(slots.get("intent") or slots.get("help_intent")).lower() or "plan"
    slot_req = _as_str(slots.get("slot") or slots.get("help_slot")).lower() or "all"
    return "provide_info", _join_args(intent_req, slot_req)



def plan_steps(tracker: Any, mrs: List[Dict[str, Any]]) -> List[ProposedStep]:
    """Build an ordered list of steps for a single user message.

    Ordering rules (deterministic)
    ------------------------------
    Within a single user turn, execute steps in this order (regardless of MR order):

      1) plan gate (request_info / propose_menus only)
      2) select_menu -> set_active_menu
      3) refine avoid updates first
      4) refine swap (suggest/commit)
      5) inspect -> show_day
      6) help -> provide_info (after other feasible actions)
      7) confirm -> confirm_plan (always final, if feasible)

    Within each category, original user order is preserved (stable).

    Policy remains the final authority and may override these proposals.
    """

    if not mrs:
        return []

    # Normalize each MR defensively (schema-level normalization only).
    norm_mrs: List[Dict[str, Any]] = []
    for mr in mrs:
        if isinstance(mr, dict):
            norm_mrs.append(normalize_mr(mr))

    # Partition
    plan_mrs = _stable_filter(norm_mrs, lambda m: m.get("intent") == "plan")
    sel_mrs = _stable_filter(norm_mrs, lambda m: m.get("intent") == "select_menu")
    refine_mrs = _stable_filter(norm_mrs, lambda m: m.get("intent") == "refine")
    inspect_mrs = _stable_filter(norm_mrs, lambda m: m.get("intent") == "inspect")
    help_mrs = _stable_filter(norm_mrs, lambda m: m.get("intent") == "help")
    confirm_mrs = _stable_filter(norm_mrs, lambda m: m.get("intent") == "confirm")

    # Refine split: avoid updates vs swap
    def _is_avoid_refine(m: Dict[str, Any]) -> bool:
        rt = normalize_upper_enum((m.get("slots", {}) or {}).get("refine_type"))
        return rt in {"ADD_AVOID_ITEM", "REMOVE_AVOID_ITEM"}

    def _is_swap_refine(m: Dict[str, Any]) -> bool:
        rt = normalize_upper_enum((m.get("slots", {}) or {}).get("refine_type"))
        return rt == "SWAP_DAY"

    refine_avoid = _stable_filter(refine_mrs, _is_avoid_refine)
    refine_swap = _stable_filter(refine_mrs, _is_swap_refine)

    steps: List[ProposedStep] = []

    
    # 1) plan gate: include at most one step (even if multiple plan MRs)
    # Only include it if it produces an actionable gate (otherwise we'd add a noisy "fallback" step).
    if plan_mrs:
        gate_action, gate_arg = _propose_for_plan_gate(tracker)
        if gate_action in {"request_info", "propose_menus"}:
            steps.append(
                ProposedStep(
                    kind="plan_gate",
                    mr=_pick_last(plan_mrs) or {"intent": "plan", "slots": {}},
                    proposed_action=gate_action,
                    proposed_argument=gate_arg,
                )
            )


    # 2) select_menu: include at most one step, using the *last* selection
    if sel_mrs:
        mr = _pick_last(sel_mrs)
        a, arg = _propose_for_select_menu(mr or {"intent": "select_menu", "slots": {}})
        steps.append(ProposedStep("select_menu", mr or {"intent": "select_menu", "slots": {}}, a, arg))

    # 3) refine avoid updates (can be multiple)
    for mr in refine_avoid:
        a, arg = _propose_for_refine_avoid(mr)
        steps.append(ProposedStep("refine_avoid", mr, a, arg))

    # 4) refine swap (can be multiple)
    for mr in refine_swap:
        a, arg = _propose_for_refine_swap(mr)
        steps.append(ProposedStep("refine_swap", mr, a, arg))

    # 5) inspect (can be multiple)
    for mr in inspect_mrs:
        a, arg = _propose_for_inspect(mr)
        steps.append(ProposedStep("inspect", mr, a, arg))

    # 6) help (execute after other feasible actions; but before confirm)
    for mr in help_mrs:
        a, arg = _propose_for_help(mr)
        steps.append(ProposedStep("help", mr, a, arg))

    # 7) confirm (always last if present; execute at most one)
    if confirm_mrs:
        steps.append(
            ProposedStep(
                kind="confirm",
                mr=_pick_last(confirm_mrs) or {"intent": "confirm", "slots": {}},
                proposed_action="confirm_plan",
                proposed_argument="",
            )
        )

    return steps


# ------------------------------ Step execution ----------------------------


def run_steps(
    tracker: Any,
    steps: List[ProposedStep],
    apply_policy_fn: Callable[[Any, Dict[str, Any], str, str], Tuple[str, str, Dict[str, Any]]],
    execute_action_fn: Callable[[str, str, Any, Any, Any], Dict[str, Any]],
    nlg_fn: Any,  # NOTE: pass the NLG object (self.nlg), not a function
    recipes: Any,
    recipes_by_id: Any,
) -> TurnResult:
    """Execute planned steps sequentially until a hard stop.

    Hard stop conditions
    --------------------
    - policy returns request_info(...): requires user input
    - final action is propose_menus: user must pick menu next

    Rendering
    ---------
    Execute all feasible steps first, then call nlg_fn.render_steps(...) ONCE.
    """

    executed: List[ExecutedStep] = []
    stop_reason = "completed"

    for step in steps:
        # Policy is authoritative.
        final_action, final_arg, dbg = apply_policy_fn(
            tracker,
            step.mr,
            step.proposed_action,
            step.proposed_argument,
        )

        payload = execute_action_fn(final_action, final_arg, tracker, recipes, recipes_by_id)

        # Record the executed step (rendered left empty; rendering happens once at end)
        executed.append(
            ExecutedStep(
                step=step,
                final_action=final_action,
                final_argument=final_arg,
                policy_debug=dbg or {},
                payload=payload or {},
                rendered="",
            )
        )

        # ---- hard stops ----
        if final_action == "request_info":
            stop_reason = "needs_user_input"
            break

        if final_action == "propose_menus":
            stop_reason = "needs_menu_selection"
            break

    # Single render call for the whole turn (no multiple LLM calls)
    if hasattr(nlg_fn, "render_steps"):
        final_reply = nlg_fn.render_steps(
            executed_steps=executed,
            tracker_state=tracker.to_state_dict() if hasattr(tracker, "to_state_dict") else {},
        )
    else:
        # Fallback (should not happen once you add render_steps)
        final_reply = "\n\n".join([e.rendered for e in executed if getattr(e, "rendered", "")])

    return TurnResult(executed_steps=executed, final_reply=final_reply, stop_reason=stop_reason)
