"""Text similarity metrics for NLG automatic evaluation.

Implemented in pure Python (no external deps) to keep the eval harness portable.

Metrics:
- BLEU-4 (token-level, cumulative, with smoothing)
- ROUGE-1 / ROUGE-2 (F1)
- ROUGE-L (F1 via LCS)

These are *surface-form* metrics; interpret cautiously for open-ended generation.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Tuple, Dict


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", re.UNICODE)


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _WORD_RE.findall(text.lower())


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(pred: str, ref: str, max_n: int = 4, smooth: float = 1.0) -> float:
    """Compute a smoothed BLEU score (0..1)."""
    pred_toks = tokenize(pred)
    ref_toks = tokenize(ref)
    if not pred_toks or not ref_toks:
        return 0.0

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        pred_counts = _ngram_counts(pred_toks, n)
        ref_counts = _ngram_counts(ref_toks, n)
        overlap = 0
        total = 0
        for ng, c in pred_counts.items():
            total += c
            overlap += min(c, ref_counts.get(ng, 0))
        # smoothing avoids log(0)
        precisions.append((overlap + smooth) / (total + smooth))

    # geometric mean
    log_p = sum(math.log(p) for p in precisions) / max_n

    # brevity penalty
    bp = 1.0
    if len(pred_toks) < len(ref_toks):
        bp = math.exp(1 - (len(ref_toks) / max(1, len(pred_toks))))

    return float(bp * math.exp(log_p))


def rouge_n_f1(pred: str, ref: str, n: int = 1) -> float:
    """ROUGE-N F1 (0..1)"""
    pred_toks = tokenize(pred)
    ref_toks = tokenize(ref)
    if not pred_toks or not ref_toks or len(pred_toks) < n or len(ref_toks) < n:
        return 0.0

    pc = _ngram_counts(pred_toks, n)
    rc = _ngram_counts(ref_toks, n)
    overlap = sum(min(pc[k], rc.get(k, 0)) for k in pc.keys())
    p = overlap / max(1, sum(pc.values()))
    r = overlap / max(1, sum(rc.values()))
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _lcs_len(a: List[str], b: List[str]) -> int:
    """Length of longest common subsequence (DP)."""
    # O(len(a)*len(b)) DP; fine for short utterances.
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


def rouge_l_f1(pred: str, ref: str) -> float:
    pred_toks = tokenize(pred)
    ref_toks = tokenize(ref)
    if not pred_toks or not ref_toks:
        return 0.0
    lcs = _lcs_len(pred_toks, ref_toks)
    p = lcs / max(1, len(pred_toks))
    r = lcs / max(1, len(ref_toks))
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)
