"""Run NLU automatic evaluation.

Example:
  python -m eval.run_nlu_eval --model qwen3 --prompt eval/prompts/nlu_base.txt \
    --data eval/data/nlu_synth.jsonl --out eval_outputs/nlu_report.json

This script does not modify system files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from eval.runner_utils import build_args, load_llm, get_logger
from eval.nlu_runner import run_nlu
from eval.metrics import score_nlu_examples


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3", help="Model key: llama3|llama31|qwen3")
    ap.add_argument("--prompt", required=True, help="Path to system prompt text file")
    ap.add_argument("--data", required=True, help="Path to JSONL dataset")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16", choices=["f32", "fp16", "bf16"])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--out", default="nlu_report.json")
    ap.add_argument("--pred-out", default="nlu_predictions.jsonl")
    args_cli = ap.parse_args()

    logger = get_logger("nlu_eval")

    args = build_args(
        model_key=args_cli.model,
        device=args_cli.device,
        dtype=args_cli.dtype,
        max_new_tokens=args_cli.max_new_tokens,
        temperature=args_cli.temperature,
        top_p=args_cli.top_p,
        parallel=args_cli.parallel,
    )

    model, tokenizer = load_llm(args)

    prompt_text = Path(args_cli.prompt).read_text(encoding="utf-8")
    rows = _read_jsonl(Path(args_cli.data))
    if args_cli.limit and args_cli.limit > 0:
        rows = rows[: args_cli.limit]

    scored_examples: List[Dict[str, Any]] = []

    pred_out_path = Path(args_cli.pred_out)
    pred_out_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_out_path.open("w", encoding="utf-8") as pf:
        for ex in rows:
            user_text = str(ex.get("user_text") or "")
            awaiting_slot = ex.get("awaiting_slot")
            recent_turns = ex.get("recent_turns")
            gold = ex.get("gold")

            pred = run_nlu(
                model=model,
                tokenizer=tokenizer,
                args=args,
                user_text=user_text,
                system_prompt=prompt_text,
                awaiting_slot=awaiting_slot,
                recent_turns=recent_turns,
            )

            out_ex = dict(ex)
            out_ex["pred"] = pred
            pf.write(json.dumps(out_ex, ensure_ascii=False) + "\n")

            scored_examples.append({"gold": gold, "pred": pred})

    scores = score_nlu_examples(scored_examples)

    report = {
        "model": args_cli.model,
        "prompt": str(args_cli.prompt),
        "data": str(args_cli.data),
        "n": scores.n,
        "intent_acc": scores.intent_acc,
        "exact_match": scores.exact_match,
        "slot_precision": scores.slot_precision,
        "slot_recall": scores.slot_recall,
        "slot_f1": scores.slot_f1,
        "generation": {
            "dtype": args_cli.dtype,
            "max_new_tokens": args_cli.max_new_tokens,
            "temperature": args_cli.temperature,
            "top_p": args_cli.top_p,
            "parallel": bool(args_cli.parallel),
        },
        "predictions_file": str(pred_out_path),
    }

    out_path = Path(args_cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("Wrote report to %s", out_path)
    logger.info("Intent acc=%.3f | Slot F1=%.3f | Exact=%.3f", scores.intent_acc, scores.slot_f1, scores.exact_match)


if __name__ == "__main__":
    main()
