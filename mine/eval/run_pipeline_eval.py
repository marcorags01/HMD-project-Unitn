"""End-to-end (pipeline) evaluation without touching system files.

Modes
- oracle: uses MRs provided in the scenario file (no NLU/DM model needed).

Metrics
- success_rate: scenarios that reach expected final phase and invariants.
- policy_reason traces and transcripts are saved for debugging.

Run:
  python -m eval.run_pipeline_eval --scenarios eval/scenarios/scenarios.json \
    --recipes recipes_30.json --out eval_outputs/pipeline_report.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from support_classes import Tracker, History
from intents_schema import validate_mr
from policy import apply_policy
from main import execute_action, parse_continue_reply
from support_fn import load_recipes


def _as_list_mrs(obj: Any) -> List[Dict[str, Any]]:
    if obj is None:
        return []
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []


@dataclass
class StepResult:
    user: str
    selected_intent: str
    action: str
    arg: str
    payload_error: Optional[str]
    phase: str
    pending_action: Any

def _apply_initial_state(tracker: Tracker, initial: Any) -> None:
    if not isinstance(initial, dict):
        return

    # Phase
    if "phase" in initial and initial["phase"] is not None:
        tracker.phase = initial["phase"]

    # Constraints
    c = initial.get("constraints")
    if isinstance(c, dict):
        tracker.constraints.update(c)

    # Menus + active menu
    m = initial.get("menus")
    if isinstance(m, dict):
        tracker.menus.update(m)

    if "active_menu_id" in initial:
        tracker.active_menu_id = initial.get("active_menu_id")

    if "active_menu" in initial:
        tracker.active_menu = initial.get("active_menu")

    # Optional fields (only if you actually store them in scenarios)
    if "pending_action" in initial:
        tracker.pending_action = initial.get("pending_action")
    if "awaiting_slot" in initial:
        tracker.awaiting_slot = initial.get("awaiting_slot")
    if "reprompt_count" in initial:
        tracker.reprompt_count = int(initial.get("reprompt_count") or 0)


def run_scenario(scn: Dict[str, Any], recipes, recipes_by_id) -> Dict[str, Any]:
    tracker = Tracker()
    history = History()
    _apply_initial_state(tracker, scn.get("initial_state"))
    transcript: List[StepResult] = []
    last_action = ""

    for turn in scn.get("turns", []):
        user_text = str(turn.get("user_text") or "").strip()
        has_mrs = bool(_as_list_mrs(turn.get("mrs")))
        if not user_text and not has_mrs:
            continue

        # Mirror early CONTINUE_DEFERRED gate from main.py
        bypass_mrs = None
        pending = getattr(tracker, "pending_action", None)
        if user_text and isinstance(pending, dict) and pending.get("type") == "CONTINUE_DEFERRED":
            decision = parse_continue_reply(user_text)
            if decision == "YES":
                next_mr = tracker.pop_deferred()
                tracker.pending_action = None
                tracker.pending_mrs.clear()
                if next_mr:
                    next_mr = dict(next_mr)
                    next_mr.pop("_turn_id", None)
                    bypass_mrs = [next_mr]
                else:
                    bypass_mrs = [{"intent": "out_of_domain", "slots": {"ood_type": "INVALID_MR"}}]
            elif decision == "NO":
                tracker.clear_deferred()
                tracker.pending_action = None
                # Treat as a handled turn with fallback reply
                transcript.append(StepResult(user=user_text, selected_intent="out_of_domain", action="fallback", arg="", payload_error=None, phase=tracker.phase, pending_action=tracker.pending_action))
                last_action = "fallback"
                continue
            else:
                # Reprompt; don't advance the pipeline
                transcript.append(StepResult(user=user_text, selected_intent="out_of_domain", action="fallback", arg="", payload_error=None, phase=tracker.phase, pending_action=tracker.pending_action))
                last_action = "fallback"
                continue

        raw_mrs = _as_list_mrs(bypass_mrs if bypass_mrs is not None else turn.get("mrs"))
        if not raw_mrs:
            # In oracle mode, every turn should provide MRs unless it's a continue-gate reply.
            raw_mrs = [{"intent": "out_of_domain", "slots": {"ood_type": "INVALID_MR"}}]

        # Validate + normalize each MR (same behavior as main.py)
        mrs: List[Dict[str, Any]] = []
        for m in raw_mrs:
            v = validate_mr(m)
            mrs.append(v.normalized_mr if v.valid else {"intent": "out_of_domain", "slots": {"ood_type": "INVALID_MR"}})

        tracker.ingest_turn(mrs, history=history)

        selected_mr = tracker.select_next_mr()
        selected_snapshot = dict(selected_mr)

        # DM stub: always propose fallback; policy should do the deterministic routing.
        proposed_action, proposed_arg = "fallback", ""

        action, arg, dbg = apply_policy(tracker, selected_mr, proposed_action, proposed_arg)
        if action == "request_info":
            tracker.note_request_info(arg)
        else:
            INTERRUPT_ACTIONS = {"show_day", "show_week", "provide_info"}
            if action not in INTERRUPT_ACTIONS and action != "fallback":
                tracker.clear_awaiting()

        payload = execute_action(action, arg, tracker, recipes, recipes_by_id)
        tracker.update_pending_after_action(selected_snapshot, action, payload)

        transcript.append(
            StepResult(
                user=user_text,
                selected_intent=str(selected_mr.get("intent")),
                action=action,
                arg=arg,
                payload_error=(payload or {}).get("error"),
                phase=tracker.phase,
                pending_action=tracker.pending_action,
            )
        )
        last_action = action

    # Post conditions
    expected = (scn.get("expected") or scn.get("expect") or {})
    if "phase" in expected and "final_phase" not in expected:
        expected["final_phase"] = expected["phase"]
    if "has_active_menu" in expected and "must_have_active_menu" not in expected:
        expected["must_have_active_menu"] = expected["has_active_menu"]
    ok = True

    if "final_phase" in expected:
        ok = ok and (tracker.phase == expected["final_phase"])

    if expected.get("must_have_active_menu") is True:
        ok = ok and tracker.has_active_menu()

    if "constraints_complete" in expected:
        want = bool(expected["constraints_complete"])
        have = (len(tracker.missing_plan_slots()) == 0)
        ok = ok and (have == want)

    return {
        "id": scn.get("id"),
        "name": scn.get("name"),
        "ok": bool(ok),
        "final_phase": tracker.phase,
        "has_active_menu": tracker.has_active_menu(),
        "transcript": [sr.__dict__ for sr in transcript],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--recipes", default="recipes_30.json")
    ap.add_argument("--out", default="pipeline_report.json")
    args = ap.parse_args()

    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))
    recipes, recipes_by_id = load_recipes(args.recipes)

    results = [run_scenario(s, recipes, recipes_by_id) for s in scenarios]
    success = sum(1 for r in results if r["ok"])

    report = {
        "n": len(results),
        "success": success,
        "success_rate": (success / len(results)) if results else 0.0,
        "results": results,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"n": report["n"], "success_rate": report["success_rate"]}, indent=2))


if __name__ == "__main__":
    main()
