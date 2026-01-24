"""Automatic metrics for Meal Kit Composer evaluation.

Strictness policy
- Both gold and predicted MRs are normalized with intents_schema.normalize_mr().
- Slot scoring is performed over (slot,value) pairs. For list slots, values are
  treated as sets (order-insensitive) but compared at element-level.

This keeps evaluation aligned with the pipeline's own canonicalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from intents_schema import normalize_mr


def _as_list_mrs(obj: Any) -> List[Dict[str, Any]]:
    if obj is None:
        return []
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []


def _slot_pairs(mr: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Convert normalized MR to comparable (slot,value) pairs."""
    nm = normalize_mr(mr)
    slots = nm.get("slots", {}) or {}
    out: List[Tuple[str, str]] = []

    for k, v in slots.items():
        if isinstance(v, list):
            for item in sorted({str(x).strip().lower() for x in v if x is not None}):
                out.append((k, item))
        else:
            out.append((k, str(v).strip().lower()))

    out.sort()
    return out


@dataclass
class Scores:
    n: int
    intent_acc: float
    exact_match: float
    slot_precision: float
    slot_recall: float
    slot_f1: float


def score_nlu_examples(examples: Iterable[Dict[str, Any]]) -> Scores:
    """Compute micro metrics over a stream of examples.

    Expected example schema:
      {"gold": [MR,...], "pred": [MR,...]}
    """

    n = 0
    intent_ok = 0
    exact_ok = 0

    tp = 0
    fp = 0
    fn = 0

    for ex in examples:
        n += 1
        gold = _as_list_mrs(ex.get("gold"))
        pred = _as_list_mrs(ex.get("pred"))

        gold_n = [normalize_mr(m) for m in gold]
        pred_n = [normalize_mr(m) for m in pred]

        gold_intents = [m.get("intent") for m in gold_n]
        pred_intents = [m.get("intent") for m in pred_n]
        if gold_intents == pred_intents:
            intent_ok += 1

        if len(gold_n) == len(pred_n) and all(gold_n[i] == pred_n[i] for i in range(len(gold_n))):
            exact_ok += 1

        gold_pairs = set()
        pred_pairs = set()
        for m in gold_n:
            gold_pairs.update(_slot_pairs(m))
        for m in pred_n:
            pred_pairs.update(_slot_pairs(m))

        tp += len(gold_pairs & pred_pairs)
        fp += len(pred_pairs - gold_pairs)
        fn += len(gold_pairs - pred_pairs)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return Scores(
        n=n,
        intent_acc=intent_ok / n if n else 0.0,
        exact_match=exact_ok / n if n else 0.0,
        slot_precision=prec,
        slot_recall=rec,
        slot_f1=f1,
    )
