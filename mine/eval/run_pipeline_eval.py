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


def run_scenario(scn: Dict[str, Any], recipes, recipes_by_id) -> Dict[str, Any]:
    tracker = Tracker()
    history = History()

    transcript: List[StepResult] = []
    last_action = ""

    for turn in scn.get("turns", []):
        user_text = str(turn.get("user_text") or "").strip()
        if not user_text:
            continue

        # Mirror early CONTINUE_DEFERRED gate from main.py
        bypass_mrs = None
        pending = getattr(tracker, "pending_action", None)
        if isinstance(pending, dict) and pending.get("type") == "CONTINUE_DEFERRED":
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
    expected = scn.get("expected", {}) or {}
    ok = True

    if "final_phase" in expected:
        ok = ok and (tracker.phase == expected["final_phase"])

    if expected.get("must_have_active_menu") is True:
        ok = ok and tracker.has_active_menu()

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
