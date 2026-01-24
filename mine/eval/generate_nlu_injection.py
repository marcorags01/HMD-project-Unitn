"""Generate a scalable, template-based (string-injection) NLU dataset.

This script produces a JSONL file compatible with `python -m eval.run_nlu_eval`.

The dataset mixes:
- High-coverage template injection (systematic slot/value combinations)
- Natural paraphrase variants (multiple templates per intent)

It also includes multi-intent turns (e.g., refine + inspect) and slot-answer turns
using `AWAITING_SLOT` + `RECENT_TURNS` contexts.

Usage (from project root):
  python -m eval.generate_nlu_injection \
    --out eval/data/nlu_injection_v1.jsonl \
    --max 1500 \
    --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# Ensure project root is importable when running from eval/*
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from support_classes import (  # noqa: E402
    ALLOWED_AVOID_ITEMS,
    ALLOWED_CALORIE_LEVELS,
    ALLOWED_DAYS,
    ALLOWED_TIME_LIMITS,
)


def _jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _join_natural(items: Sequence[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _surface_variants_for_item(token: str) -> List[str]:
    """Surface forms used in user utterances (gold stays canonical)."""
    token = token.strip().lower()
    mapping = {
        "egg": ["egg", "eggs"],
        "nuts": ["nuts", "nut"],
        "dairy": ["dairy", "milk", "cheese"],
        "gluten": ["gluten", "wheat", "bread"],
        "fish": ["fish", "seafood"],
        "shellfish": ["shellfish", "shell fish", "shrimp"],
        "meat": ["meat", "meats"],
        "sesame": ["sesame"],
        "soy": ["soy"],
    }
    return mapping.get(token, [token])


def _pick(rng: random.Random, xs: Sequence[str]) -> str:
    return xs[rng.randrange(0, len(xs))]


def _plan_prep_phrase(rng: random.Random, time_limit: str) -> str:
    tl = time_limit.upper()
    if tl == "FAST":
        return _pick(rng, ["quick prep", "fast meals", "short prep time", "as quick as possible"])
    return _pick(rng, ["normal prep", "regular prep", "standard prep time", "no rush"])


def _plan_cal_phrase(rng: random.Random, calorie_level: str) -> str:
    cl = calorie_level.upper()
    if cl == "LOW":
        return _pick(rng, ["low calories", "light meals", "lighter calories", "low-calorie"])
    if cl == "MED":
        return _pick(rng, ["balanced calories", "medium calories", "average calories", "moderate calories"])
    return _pick(rng, ["high calories", "hearty meals", "filling meals", "high-calorie"])


def _avoid_phrase(rng: random.Random, avoid_items: Sequence[str]) -> str:
    if not avoid_items:
        return _pick(rng, ["no restrictions", "no allergies", "nothing to avoid", "I eat everything"])

    surfaces: List[str] = []
    for tok in avoid_items:
        surfaces.append(_pick(rng, _surface_variants_for_item(tok)))

    joined = _join_natural(surfaces)
    return _pick(
        rng,
        [
            f"avoid {joined}",
            f"no {joined}",
            f"without {joined}",
            f"please exclude {joined}",
        ],
    )


def _mk_ex(
    idx: int,
    user_text: str,
    gold: List[Dict[str, Any]],
    awaiting_slot: Optional[str] = None,
    recent_turns: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "id": f"inj{idx:05d}",
        "user_text": user_text,
        "awaiting_slot": awaiting_slot,
        "recent_turns": recent_turns,
        "gold": gold,
    }


def _plan_mr(servings: int, time_limit: str, calorie_level: str, avoid_items: List[str]) -> Dict[str, Any]:
    return {
        "intent": "plan",
        "slots": {
            "servings": int(servings),
            "time_limit": time_limit,
            "calorie_level": calorie_level,
            "avoid_items": list(avoid_items),
        },
    }


def _generate_rows(rng: random.Random) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    idx = 1

    days = sorted(ALLOWED_DAYS)
    avoids = sorted(ALLOWED_AVOID_ITEMS)
    tls = sorted(ALLOWED_TIME_LIMITS)
    cals = sorted(ALLOWED_CALORIE_LEVELS)

    # ------------------------- PLAN (full combinations) -------------------------
    plan_templates = [
        "Plan meals for {servings} people, {prep}, {cal}, {avoid}.",
        "I need a weekly meal plan for {servings} servings — {prep}, {cal}, {avoid}.",
    ]

    # Avoid sets: empty + all singles + a deterministic sample of pairs
    avoid_sets: List[List[str]] = [[]]
    avoid_sets += [[a] for a in avoids]
    all_pairs = list(combinations(avoids, 2))
    rng.shuffle(all_pairs)
    avoid_sets += [list(p) for p in all_pairs[:12]]

    for servings in range(1, 7):
        for tl in tls:
            for cal in cals:
                for av in avoid_sets:
                    for tmpl in plan_templates:
                        utt = tmpl.format(
                            servings=servings,
                            prep=_plan_prep_phrase(rng, tl),
                            cal=_plan_cal_phrase(rng, cal),
                            avoid=_avoid_phrase(rng, av),
                        )
                        rows.append(_mk_ex(idx, utt, [_plan_mr(servings, tl, cal, av)]))
                        idx += 1

    # ------------------------- PLAN (single-slot / delta turns) -------------------------
    # servings
    for s in range(1, 7):
        for utt in [f"{s} servings please.", f"Make it for {s}.", f"Just {s} people."]:
            rows.append(_mk_ex(idx, utt, [{"intent": "plan", "slots": {"servings": s}}]))
            idx += 1

    # time_limit
    tl_map = {"FAST": ["quick", "fast", "asap"], "NORMAL": ["normal", "regular", "no rush"]}
    for tl, variants in tl_map.items():
        for v in variants:
            rows.append(_mk_ex(idx, v, [{"intent": "plan", "slots": {"time_limit": tl}}]))
            idx += 1

    # calorie_level
    cal_map = {"LOW": ["low", "light"], "MED": ["balanced", "medium"], "HIGH": ["high", "hearty"]}
    for cal, variants in cal_map.items():
        for v in variants:
            rows.append(_mk_ex(idx, v, [{"intent": "plan", "slots": {"calorie_level": cal}}]))
            idx += 1

    # avoid_items (plan)
    rows.append(_mk_ex(idx, "No allergies.", [{"intent": "plan", "slots": {"avoid_items": []}}])); idx += 1
    rows.append(_mk_ex(idx, "No restrictions.", [{"intent": "plan", "slots": {"avoid_items": []}}])); idx += 1
    for tok in avoids:
        surf = _pick(rng, _surface_variants_for_item(tok))
        rows.append(_mk_ex(idx, f"Please avoid {surf}.", [{"intent": "plan", "slots": {"avoid_items": [tok]}}])); idx += 1
    # a couple of multi-item avoids
    rows.append(_mk_ex(idx, "I can't eat dairy and gluten.", [{"intent": "plan", "slots": {"avoid_items": ["dairy", "gluten"]}}])); idx += 1
    rows.append(_mk_ex(idx, "avoid meat, egg.", [{"intent": "plan", "slots": {"avoid_items": ["meat", "egg"]}}])); idx += 1

    # ------------------------- SELECT_MENU -------------------------
    recent_menu_q = "assistant: Which option—1 or 2?\n"
    menu_templates = {
        1: ["1", "Option 1.", "I'll take menu 1.", "Go with the first one."],
        2: ["2", "Option 2.", "Let's do menu 2.", "The second one seems good."],
    }
    for mid, utters in menu_templates.items():
        for utt in utters:
            rows.append(_mk_ex(idx, utt, [{"intent": "select_menu", "slots": {"menu_id": mid}}], recent_turns=recent_menu_q)); idx += 1
            rows.append(_mk_ex(idx, utt, [{"intent": "select_menu", "slots": {"menu_id": mid}}], awaiting_slot="menu_id", recent_turns=recent_menu_q)); idx += 1

    # ------------------------- INSPECT -------------------------
    inspect_templates = [
        "What's planned for {day}?",
        "Show me {day}.",
        "What do we have on {day}?",
    ]
    for day in days:
        for tmpl in inspect_templates:
            rows.append(_mk_ex(idx, tmpl.format(day=day), [{"intent": "inspect", "slots": {"target_day": day}}])); idx += 1

    # ------------------------- SHOW_WEEK -------------------------
    for utt in [
        "Show me the weekly plan.",
        "Show week.",
        "Can you show the week again?",
        "Weekly menu, please.",
    ]:
        rows.append(_mk_ex(idx, utt, [{"intent": "show_week", "slots": {}}])); idx += 1

    # ------------------------- REFINE (swap) -------------------------
    for day in days:
        rows.append(
            _mk_ex(
                idx,
                f"Can you suggest an alternative for {day}?",
                [{"intent": "refine", "slots": {"refine_type": "SWAP_DAY", "target_day": day, "value": "BEST_FIT", "mode": "SUGGEST"}}],
            )
        )
        idx += 1
        rows.append(
            _mk_ex(
                idx,
                f"Swap {day} please.",
                [{"intent": "refine", "slots": {"refine_type": "SWAP_DAY", "target_day": day, "value": "BEST_FIT", "mode": "COMMIT"}}],
            )
        )
        idx += 1

    # ------------------------- REFINE (avoid list updates) -------------------------
    for tok in avoids:
        rows.append(
            _mk_ex(
                idx,
                f"Please avoid {tok} from now on.",
                [{"intent": "refine", "slots": {"refine_type": "ADD_AVOID_ITEM", "value": tok}}],
            )
        )
        idx += 1
        rows.append(
            _mk_ex(
                idx,
                f"I can eat {tok} again.",
                [{"intent": "refine", "slots": {"refine_type": "REMOVE_AVOID_ITEM", "value": tok}}],
            )
        )
        idx += 1

    # ------------------------- CONFIRM -------------------------
    for utt in [
        "Finalize and generate the shopping list.",
        "I'm done—make the list.",
        "Confirm and generate the shopping list.",
        "Generate my shopping list, please.",
    ]:
        rows.append(_mk_ex(idx, utt, [{"intent": "confirm", "slots": {}}])); idx += 1

    # ------------------------- HELP (schema-aligned: intent/slot) -------------------------
    rows.append(_mk_ex(idx, "help", [{"intent": "help", "slots": {"intent": "plan"}}])); idx += 1
    rows.append(_mk_ex(idx, "Help with calories", [{"intent": "help", "slots": {"intent": "plan", "slot": "calorie_level"}}])); idx += 1
    rows.append(_mk_ex(idx, "What time options do you support?", [{"intent": "help", "slots": {"intent": "plan", "slot": "time_limit"}}])); idx += 1

    # ------------------------- OUT OF DOMAIN -------------------------
    rows.append(_mk_ex(idx, "Tell me a joke.", [{"intent": "out_of_domain", "slots": {}}])); idx += 1
    rows.append(_mk_ex(idx, "Thanks!", [{"intent": "out_of_domain", "slots": {"ood_type": "ACK"}}])); idx += 1

    # ------------------------- Multi-intent turns -------------------------
    rows.append(
        _mk_ex(
            idx,
            "Avoid gluten and show Tue.",
            [
                {"intent": "refine", "slots": {"refine_type": "ADD_AVOID_ITEM", "value": "gluten"}},
                {"intent": "inspect", "slots": {"target_day": "Tue"}},
            ],
        )
    )
    idx += 1
    rows.append(
        _mk_ex(
            idx,
            "Show week and then finalize.",
            [
                {"intent": "show_week", "slots": {}},
                {"intent": "confirm", "slots": {}},
            ],
        )
    )
    idx += 1
    rows.append(
        _mk_ex(
            idx,
            "Option 2 and confirm.",
            [
                {"intent": "select_menu", "slots": {"menu_id": 2}},
                {"intent": "confirm", "slots": {}},
            ],
        )
    )
    idx += 1

    # ------------------------- Awaiting-slot examples (incl. INVALID_ANSWER) -------------------------
    rows.append(_mk_ex(idx, "normal", [{"intent": "plan", "slots": {"time_limit": "NORMAL"}}], awaiting_slot="time_limit", recent_turns="assistant: Quick meals or normal prep?\n")); idx += 1
    rows.append(_mk_ex(idx, "balanced", [{"intent": "plan", "slots": {"calorie_level": "MED"}}], awaiting_slot="calorie_level", recent_turns="assistant: Low/medium/high calories?\n")); idx += 1
    rows.append(_mk_ex(idx, "no meat or dairy", [{"intent": "plan", "slots": {"avoid_items": ["meat", "dairy"]}}], awaiting_slot="avoid_items", recent_turns="assistant: Any allergies?\n")); idx += 1
    rows.append(_mk_ex(idx, "maybe", [{"intent": "out_of_domain", "slots": {"ood_type": "INVALID_ANSWER"}}], awaiting_slot="servings", recent_turns="assistant: How many servings?\n")); idx += 1
    rows.append(_mk_ex(idx, "my sister eats a lot", [{"intent": "out_of_domain", "slots": {"ood_type": "INVALID_ANSWER"}}], awaiting_slot="calorie_level", recent_turns="assistant: Low/medium/high calories?\n")); idx += 1

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/data/nlu_injection_v1.jsonl")
    ap.add_argument("--max", type=int, default=1500, help="Maximum number of examples to write (0 = all)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--shuffle", action="store_true", help="Shuffle examples before truncation")
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    rows = _generate_rows(rng)
    if args.shuffle:
        rng.shuffle(rows)

    if args.max and args.max > 0:
        rows = rows[: int(args.max)]

    # Re-number ids after truncation for cleanliness
    for i, r in enumerate(rows, start=1):
        r["id"] = f"inj{i:05d}"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(_jdump(r) + "\n")

    print(_jdump({"wrote": str(out_path), "n": len(rows)}))


if __name__ == "__main__":
    main()
