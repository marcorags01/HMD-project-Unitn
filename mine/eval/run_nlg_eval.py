"""Automatic NLG evaluation (BLEU/ROUGE + constraint checks), per course guidance.

The course material suggests using surface-form metrics (BLEU, ROUGE) for NLG
when reference texts are available.

This script supports two workflows:

A) Gold JSONL (recommended)
    --data eval/data/nlg_gold.jsonl
Each line must contain:
  {
    "id": "...",
    "action": "...",
    "argument": "...",
    "tracker_state": {...},
    "payload": {...},
    "reference": "..."
  }

B) Scenario replay (fallback)
    --scenarios eval/scenarios/scenarios_v1.json --recipes recipes_30.json
We replay scenarios using a deterministic oracle DM to obtain (action, payload)
pairs, then compare model NLG against a deterministic reference renderer.

Important: BLEU/ROUGE are weak for open-ended generation. In the report, treat
them as *rough proxies*, and complement with constraint/format checks.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eval.runner_utils import build_args, load_llm, get_logger
from eval.text_metrics import bleu_score, rouge_n_f1, rouge_l_f1
from eval.dm_oracle import oracle_dm_action
from intents_schema import validate_mr
from nlg import NLG
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
    for k, v in state.items():
        if hasattr(t, k):
            setattr(t, k, v)
    return t


def _maybe_override_nlg_prompt(prompt_path: Optional[Path]) -> None:
    """Optionally override NLG system prompt without editing system files."""
    if prompt_path is None:
        return
    from utils import PROMPTS  # local import

    text = prompt_path.read_text(encoding="utf-8")
    PROMPTS["NLG_START"] = text
    # keep NLG_END as-is unless user put it in the file; they can fully override by adding it themselves


def _reference_render(action: str, arg: str, payload: Dict[str, Any]) -> str:
    """Deterministic reference generator for scenario-replay mode.

    This is intentionally simple and content-driven (block inclusion), so BLEU/ROUGE
    mostly reflect whether the system preserved critical payload blocks.
    """
    action = (action or "").strip()
    arg = (arg or "").strip()
    p = payload or {}

    if action == "request_info":
        q = {
            "servings": "How many servings?",
            "time_limit": "Quick meals or normal prep?",
            "calorie_level": "Low/medium/high calories?",
            "avoid_items": "Any allergies or foods to avoid? (e.g., nuts, dairy, egg)",
            "menu_id": "Which option—1 or 2?",
            "target_day": "Which day (Mon/Tue/Wed/Thu/Fri)?",
        }.get(arg, f"Please provide: {arg}")
        return q

    if action == "propose_menus":
        m1 = p.get("menu1_pretty") or p.get("menu1") or ""
        m2 = p.get("menu2_pretty") or p.get("menu2") or ""
        return f"Here are two menu options.\n\nOption 1:\n{m1}\n\nOption 2:\n{m2}\n\nWhich option—1 or 2?"

    if action == "set_active_menu":
        return f"Okay — I’ll use option {arg}. Ask me to show a day, swap a day, or show the week."

    if action == "show_day":
        details = p.get("details") or ""
        return f"Here are the details for {arg}:\n{details}"

    if action == "show_week":
        w = p.get("week_overview") or ""
        return f"Here’s your week plan (Monday to Friday):\n{w}"

    if action == "suggest_swap_day":
        title = p.get("suggested_title") or p.get("suggested") or ""
        return f"Suggestion for {arg}: {title}\nIf you want it, say: swap {arg}."

    if action == "swap_day":
        title = p.get("new_title") or p.get("swapped") or ""
        return f"Done — swapped {arg}. New recipe: {title}"

    if action == "update_avoid":
        return "Done — I updated foods to avoid."

    if action == "confirm_plan":
        sl = p.get("shopping_list") or ""
        return f"All set. Here is your shopping list:\n{sl}"

    if action == "provide_info":
        return "Help: tell me servings, prep time, calories, and any foods to avoid. You can also ask to show/swap days."

    return "Sorry — I didn’t catch that. Could you rephrase?"


def _action_leak(text: str) -> bool:
    """Hard constraint: NLG must not output raw action strings."""
    if not text:
        return False
    banned = [
        "request_info(",
        "provide_info(",
        "propose_menus(",
        "set_active_menu(",
        "show_day(",
        "show_week(",
        "suggest_swap_day(",
        "swap_day(",
        "update_avoid(",
        "confirm_plan(",
        "fallback(",
    ]
    t = text.lower()
    return any(b in t for b in banned)


@dataclass
class _NLGRecord:
    id: str
    action: str
    argument: str
    bleu: float
    rouge1_f1: float
    rouge2_f1: float
    rougeL_f1: float
    leaked_action: bool
    pred: str
    ref: str


def _score_one(pred: str, ref: str) -> Tuple[float, float, float, float]:
    return (
        bleu_score(pred, ref),
        rouge_n_f1(pred, ref, n=1),
        rouge_n_f1(pred, ref, n=2),
        rouge_l_f1(pred, ref),
    )


def _eval_jsonl(nlg: NLG, rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[_NLGRecord]]:
    recs: List[_NLGRecord] = []
    for i, row in enumerate(rows):
        ex_id = str(row.get("id", f"ex{i:05d}"))
        action = str(row.get("action", "")).strip()
        arg = str(row.get("argument", "")).strip()
        tracker_state = row.get("tracker_state") or {}
        payload = row.get("payload") or {}
        ref = str(row.get("reference", "")).strip()

        pred = nlg(action=action, argument=arg, tracker_state=tracker_state, payload=payload)
        b, r1, r2, rl = _score_one(pred, ref)
        recs.append(_NLGRecord(ex_id, action, arg, b, r1, r2, rl, _action_leak(pred), pred, ref))

    report = {
        "n": len(recs),
        "bleu": sum(r.bleu for r in recs) / max(1, len(recs)),
        "rouge1_f1": sum(r.rouge1_f1 for r in recs) / max(1, len(recs)),
        "rouge2_f1": sum(r.rouge2_f1 for r in recs) / max(1, len(recs)),
        "rougeL_f1": sum(r.rougeL_f1 for r in recs) / max(1, len(recs)),
        "action_leak_rate": sum(1 for r in recs if r.leaked_action) / max(1, len(recs)),
    }
    return report, recs


def _eval_scenarios(nlg: NLG, scenarios: List[Dict[str, Any]], recipes_path: str) -> Tuple[Dict[str, Any], List[_NLGRecord]]:
    recipes, recipes_by_id = load_recipes(recipes_path)
    recs: List[_NLGRecord] = []

    for scn in scenarios:
        sid = str(scn.get("id", "scenario"))
        tracker = _tracker_from_state_dict(scn.get("initial_state") or {})

        turns = scn.get("turns") or []
        if not isinstance(turns, list):
            continue

        for ti, turn in enumerate(turns):
            user_text = str((turn or {}).get("user_text", ""))

            # emulate CONTINUE_DEFERRED gate
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
                    continue
                else:
                    continue

            raw_mrs = bypass_mrs if bypass_mrs is not None else ((turn or {}).get("mrs") or [])
            if isinstance(raw_mrs, dict):
                raw_mrs = [raw_mrs]
            if not isinstance(raw_mrs, list):
                raw_mrs = []

            # validate like main
            mrs: List[Dict[str, Any]] = []
            for m in raw_mrs:
                v = validate_mr(m)
                if v.valid:
                    mrs.append(v.normalized_mr)
                else:
                    mrs.append({"intent": "out_of_domain", "slots": {"ood_type": "INVALID_MR"}})

            tracker.ingest_turn(mrs, history=None)
            selected_mr = tracker.select_next_mr()
            selected_snapshot = json.loads(json.dumps(selected_mr))

            # Use oracle DM to choose action (keeps this an NLG evaluation, not DM eval).
            oa, oarg = oracle_dm_action(tracker, selected_mr)
            action, arg, _dbg = apply_policy(tracker, selected_mr, oa, oarg)

            # awaiting-slot bookkeeping
            if action == "request_info":
                tracker.note_request_info(arg)
            else:
                INTERRUPT_ACTIONS = {"show_day", "show_week", "provide_info"}
                if action in INTERRUPT_ACTIONS:
                    pass
                elif action != "fallback":
                    tracker.clear_awaiting()

            payload = execute_action(action, arg, tracker, recipes, recipes_by_id)
            tracker.update_pending_after_action(selected_snapshot, action, payload)

            pred = nlg(action=action, argument=arg, tracker_state=tracker.to_state_dict(), payload=payload)
            ref = _reference_render(action, arg, payload)
            b, r1, r2, rl = _score_one(pred, ref)
            recs.append(_NLGRecord(f"{sid}:{ti}", action, arg, b, r1, r2, rl, _action_leak(pred), pred, ref))

    report = {
        "n": len(recs),
        "bleu": sum(r.bleu for r in recs) / max(1, len(recs)),
        "rouge1_f1": sum(r.rouge1_f1 for r in recs) / max(1, len(recs)),
        "rouge2_f1": sum(r.rouge2_f1 for r in recs) / max(1, len(recs)),
        "rougeL_f1": sum(r.rougeL_f1 for r in recs) / max(1, len(recs)),
        "action_leak_rate": sum(1 for r in recs if r.leaked_action) / max(1, len(recs)),
        "note": "Scenario-replay mode uses a deterministic reference renderer; interpret BLEU/ROUGE as block-preservation proxies.",
    }
    return report, recs


# ------------------------- CLI -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3", help="Model key: llama3|llama31|qwen3")
    ap.add_argument("--prompt", default="", help="Optional NLG prompt override (system prompt text file)")
    ap.add_argument("--data", default="", help="Gold JSONL with reference texts")
    ap.add_argument("--scenarios", default="", help="Scenario JSON (fallback mode)")
    ap.add_argument("--recipes", default="recipes_30.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16", choices=["f32", "fp16", "bf16"])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--out", default="eval_outputs/nlg_report.json")
    ap.add_argument("--pred-out", default="eval_outputs/nlg_preds.jsonl")
    args = ap.parse_args()

    logger = get_logger("nlg_eval")

    prompt_path = Path(args.prompt) if args.prompt else None
    _maybe_override_nlg_prompt(prompt_path)

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
    nlg = NLG(history=None, model=model, tokenizer=tokenizer, args=llm_args, logger=logger)

    if args.data:
        rows = _read_jsonl(Path(args.data))
        if args.limit and args.limit > 0:
            rows = rows[: args.limit]
        report, recs = _eval_jsonl(nlg, rows)
    elif args.scenarios:
        scenarios = _read_json(Path(args.scenarios))
        if args.limit and args.limit > 0:
            scenarios = scenarios[: args.limit]
        report, recs = _eval_scenarios(nlg, scenarios, recipes_path=args.recipes)
    else:
        raise SystemExit("Provide either --data or --scenarios")

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
