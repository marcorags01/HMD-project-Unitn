"""Intrinsic DM evaluation (action classification), per course guidance.

What it measures
- Given a tracker state and a selected MR (meaning representation),
  the DM proposes the next action from the closed action set.
- We report macro-F1 / accuracy on actions.
- We also report argument exact-match *conditional on action match*.

Gold labels
- By default we use a deterministic oracle mapping (eval.dm_oracle.oracle_dm_action)
  that encodes the intended workflow gates.
- This avoids circular evaluation where "gold" comes from the same DM being tested.

Evaluation modes
1) Scenario replay: --scenarios (JSON, same schema as eval/scenarios/*.json)
   The harness replays turns, updates tracker via main.execute_action,
   and evaluates the DM at each turn.

2) Flat dataset: --data (JSONL) with explicit (tracker_state, mr, gold_action, gold_arg).


"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eval.runner_utils import build_args, load_llm, get_logger
from eval.dm_oracle import oracle_dm_action
from intents_schema import validate_mr
from dm import DM
from policy import apply_policy
from support_classes import Tracker
from support_fn import load_recipes
from main import execute_action, parse_continue_reply


# ------------------------- Helpers -------------------------

def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _tracker_from_state_dict(state: Dict[str, Any]) -> Tracker:
    t = Tracker()
    if not isinstance(state, dict):
        return t
    # Shallow fields
    for k, v in state.items():
        if hasattr(t, k):
            setattr(t, k, v)
    return t


def _maybe_override_dm_prompt(prompt_path: Optional[Path]) -> None:
    """Optionally override the DM prompt without editing system files.

    If provided, the prompt file is treated as a *full* DM system prompt.
    We patch utils.PROMPTS keys so dm.DM will use it.
    """
    if prompt_path is None:
        return
    from utils import PROMPTS  # local import
    import dm as dm_mod

    text = prompt_path.read_text(encoding="utf-8")
    PROMPTS["DM_START"] = text
    PROMPTS["DM_ACTIONS"] = ""
    PROMPTS["DM_RULES"] = ""
    PROMPTS["DM_END"] = ""
    dm_mod.DM_EXTRA_RULES = ""


@dataclass
class _ActionRecord:
    scenario_id: str
    turn_index: int
    user_text: str
    selected_intent: str
    gold_action: str
    gold_arg: str
    dm_action: str
    dm_arg: str
    final_action: str
    final_arg: str
    policy_reason: Optional[str]


def _f1_by_label(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    tp = {l: 0 for l in labels}
    fp = {l: 0 for l in labels}
    fn = {l: 0 for l in labels}
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    per = {}
    f1s = []
    for l in labels:
        precision = tp[l] / max(1, tp[l] + fp[l])
        recall = tp[l] / max(1, tp[l] + fn[l])
        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))
        per[l] = {"precision": precision, "recall": recall, "f1": f1, "support": tp[l] + fn[l]}
        f1s.append(f1)

    macro_f1 = sum(f1s) / max(1, len(f1s))
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(1, len(y_true))
    return {"labels": labels, "macro_f1": macro_f1, "accuracy": acc, "per_label": per}


def _eval_flat_dataset(dm: DM, rows: List[Dict[str, Any]], logger) -> Tuple[Dict[str, Any], List[_ActionRecord]]:
    recs: List[_ActionRecord] = []
    gold_actions: List[str] = []
    dm_actions: List[str] = []
    final_actions: List[str] = []

    for i, row in enumerate(rows):
        scenario_id = str(row.get("id", f"ex{i:05d}"))
        tracker = _tracker_from_state_dict(row.get("tracker_state") or {})
        mr = row.get("mr") or {}
        v = validate_mr(mr)
        mr = v.normalized_mr if v.valid else {"intent": "out_of_domain", "slots": {"ood_type": "INVALID_MR"}}

        gold_a = str(row.get("gold_action", "")).strip()
        gold_arg = str(row.get("gold_arg", "")).strip()

        dm_a, dm_arg, _ = dm(tracker, mr, last_action=None)
        final_a, final_arg, dbg = apply_policy(tracker, mr, dm_a, dm_arg)

        recs.append(
            _ActionRecord(
                scenario_id=scenario_id,
                turn_index=int(row.get("turn_index", 0)),
                user_text=str(row.get("user_text", "")),
                selected_intent=str(mr.get("intent", "")),
                gold_action=gold_a,
                gold_arg=gold_arg,
                dm_action=dm_a,
                dm_arg=dm_arg,
                final_action=final_a,
                final_arg=final_arg,
                policy_reason=str(dbg.get("policy_reason")) if isinstance(dbg, dict) else None,
            )
        )
        gold_actions.append(gold_a)
        dm_actions.append(dm_a)
        final_actions.append(final_a)

    report = {
        "n": len(recs),
        "dm": _f1_by_label(gold_actions, dm_actions),
        "final": _f1_by_label(gold_actions, final_actions),
        "arg_exact_match_given_action_dm": (
            sum(1 for r in recs if r.dm_action == r.gold_action and r.dm_arg.strip() == r.gold_arg.strip())
            / max(1, sum(1 for r in recs if r.dm_action == r.gold_action))
        ),
        "arg_exact_match_given_action_final": (
            sum(1 for r in recs if r.final_action == r.gold_action and r.final_arg.strip() == r.gold_arg.strip())
            / max(1, sum(1 for r in recs if r.final_action == r.gold_action))
        ),
    }
    return report, recs


def _eval_scenarios(dm: DM, scenarios: List[Dict[str, Any]], recipes_path: str, logger) -> Tuple[Dict[str, Any], List[_ActionRecord]]:
    recipes, recipes_by_id = load_recipes(recipes_path)

    recs: List[_ActionRecord] = []
    gold_actions: List[str] = []
    dm_actions: List[str] = []
    final_actions: List[str] = []

    for scn in scenarios:
        sid = str(scn.get("id", "scenario"))
        tracker = _tracker_from_state_dict(scn.get("initial_state") or {})

        turns = scn.get("turns") or []
        if not isinstance(turns, list):
            continue

        last_action = None
        for ti, turn in enumerate(turns):
            user_text = str((turn or {}).get("user_text", ""))

            # Emulate main.py early CONTINUE_DEFERRED gate.
            bypass_mrs = None
            pending = getattr(tracker, "pending_action", None)
            if isinstance(pending, dict) and pending.get("type") == "CONTINUE_DEFERRED":
                decision = parse_continue_reply(user_text)
                if decision == "YES":
                    next_mr = tracker.pop_deferred()
                    tracker.pending_action = None
                    tracker.pending_mrs.clear()
                    bypass_mrs = [next_mr] if next_mr else []
                elif decision == "NO":
                    tracker.clear_deferred()
                    tracker.pending_action = None
                    # DM not invoked
                    continue
                else:
                    # DM not invoked
                    continue

            raw_mrs = bypass_mrs if bypass_mrs is not None else ((turn or {}).get("mrs") or [])
            if isinstance(raw_mrs, dict):
                raw_mrs = [raw_mrs]
            if not isinstance(raw_mrs, list):
                raw_mrs = []

            # Validate/normalize MRs like main.py
            mrs: List[Dict[str, Any]] = []
            for m in raw_mrs:
                v = validate_mr(m)
                if v.valid:
                    mrs.append(v.normalized_mr)
                else:
                    mrs.append({"intent": "out_of_domain", "slots": {"ood_type": "INVALID_MR"}})

            tracker.ingest_turn(mrs, history=None)

            selected_mr = tracker.select_next_mr()
            selected_snapshot = json.loads(json.dumps(selected_mr))  # deep copy without importing copy

            gold_a, gold_arg = oracle_dm_action(tracker, selected_mr)

            dm_a, dm_arg, _ = dm(tracker, selected_mr, last_action=last_action)
            final_a, final_arg, dbg = apply_policy(tracker, selected_mr, dm_a, dm_arg)

            recs.append(
                _ActionRecord(
                    scenario_id=sid,
                    turn_index=ti,
                    user_text=user_text,
                    selected_intent=str(selected_mr.get("intent", "")),
                    gold_action=gold_a,
                    gold_arg=gold_arg,
                    dm_action=dm_a,
                    dm_arg=dm_arg,
                    final_action=final_a,
                    final_arg=final_arg,
                    policy_reason=str(dbg.get("policy_reason")) if isinstance(dbg, dict) else None,
                )
            )
            gold_actions.append(gold_a)
            dm_actions.append(dm_a)
            final_actions.append(final_a)

            # Mirror main.py awaiting-slot handling (important for subsequent gating).
            if final_a == "request_info":
                tracker.note_request_info(final_arg)
            else:
                INTERRUPT_ACTIONS = {"show_day", "show_week", "provide_info"}
                if final_a in INTERRUPT_ACTIONS:
                    pass
                elif final_a != "fallback":
                    tracker.clear_awaiting()

            payload = execute_action(final_a, final_arg, tracker, recipes, recipes_by_id)
            tracker.update_pending_after_action(selected_snapshot, final_a, payload)

            last_action = final_a

    report = {
        "n": len(recs),
        "dm": _f1_by_label(gold_actions, dm_actions),
        "final": _f1_by_label(gold_actions, final_actions),
        "arg_exact_match_given_action_dm": (
            sum(1 for r in recs if r.dm_action == r.gold_action and r.dm_arg.strip() == r.gold_arg.strip())
            / max(1, sum(1 for r in recs if r.dm_action == r.gold_action))
        ),
        "arg_exact_match_given_action_final": (
            sum(1 for r in recs if r.final_action == r.gold_action and r.final_arg.strip() == r.gold_arg.strip())
            / max(1, sum(1 for r in recs if r.final_action == r.gold_action))
        ),
    }
    return report, recs


# ------------------------- CLI -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3", help="Model key: llama3|llama31|qwen3")
    ap.add_argument("--prompt", default="", help="Optional DM prompt override (full system prompt text file)")
    ap.add_argument("--scenarios", default="", help="Scenario JSON (eval/scenarios/*.json)")
    ap.add_argument("--data", default="", help="Flat JSONL dataset with gold actions")
    ap.add_argument("--recipes", default="recipes_30.json", help="Recipes JSON used when replaying scenarios")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16", choices=["f32", "fp16", "bf16"])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--out", default="eval_outputs/dm_report.json")
    ap.add_argument("--pred-out", default="eval_outputs/dm_preds.jsonl")
    args = ap.parse_args()

    logger = get_logger("dm_eval")

    prompt_path = Path(args.prompt) if args.prompt else None
    _maybe_override_dm_prompt(prompt_path)

    llm_args = build_args(
        model_key=args.model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        parallel=args.parallel,
        debug=False,
    )
    model, tokenizer = load_llm(llm_args)

    dm = DM(history=None, model=model, tokenizer=tokenizer, args=llm_args, logger=logger)

    recs: List[_ActionRecord]
    if args.scenarios:
        scenarios = _read_json(Path(args.scenarios))
        if args.limit and args.limit > 0:
            scenarios = scenarios[: args.limit]
        report, recs = _eval_scenarios(dm, scenarios, recipes_path=args.recipes, logger=logger)
    elif args.data:
        rows = _read_jsonl(Path(args.data))
        if args.limit and args.limit > 0:
            rows = rows[: args.limit]
        report, recs = _eval_flat_dataset(dm, rows, logger=logger)
    else:
        raise SystemExit("Provide either --scenarios or --data")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"Wrote report to {out_path}")

    pred_path = Path(args.pred_out)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
    logger.info(f"Wrote predictions to {pred_path}")


if __name__ == "__main__":
    main()
