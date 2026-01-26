"""NLU runner for automatic evaluation.

This runner intentionally does *not* import or instantiate your nlu.NLU class
so you can benchmark different prompts without touching your system files.

It reuses:
- utils.format_chat / utils.generate
- support_fn.parsing_json
- intents_schema.normalize_mr / validate_mr

Output contract: always returns list[dict] (possibly length 1).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from intents_schema import INTENT_SLOTS, validate_mr
from support_classes import (
    ALLOWED_DAYS,
    ALLOWED_TIME_LIMITS,
    ALLOWED_CALORIE_LEVELS,
    ALLOWED_AVOID_ITEMS,
)
from support_fn import parsing_json
from utils import format_chat, generate, infer_input_device


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def run_nlu(
    *,
    model,
    tokenizer,
    args,
    user_text: str,
    system_prompt: str,
    awaiting_slot: Optional[str] = None,
    recent_turns: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run an NLU prompt once and return normalized/validated MRs."""

    schema_hint = {
        "intents": list(INTENT_SLOTS.keys()),
        "slots_by_intent": INTENT_SLOTS,
        "controlled_values": {
            "servings": "int 1..6",
            "time_limit": sorted(list(ALLOWED_TIME_LIMITS)),
            "calorie_level": sorted(list(ALLOWED_CALORIE_LEVELS)),
            "avoid_items": sorted(list(ALLOWED_AVOID_ITEMS)),
            "target_day": sorted(list(ALLOWED_DAYS)),
            "refine_type": ["SWAP_DAY", "ADD_AVOID_ITEM", "REMOVE_AVOID_ITEM"],
            "swap_value": "BEST_FIT",
            "refine_mode": ["SUGGEST", "COMMIT"],
            "menu_id": [1, 2],
            "slot": [
                "servings", "time_limit", "calorie_level", "avoid_items",
                "menu_id", "target_day", "refine_type", "value", "all",
            ],
            "intent": ["plan", "select_menu", "inspect", "refine", "confirm", "show_week"],
        },
        "output_format": (
            "Return ONLY valid JSON: either a single MR object "
            "{\"intent\":\"...\",\"slots\":{...}} or an array of MR objects "
            "[{\"intent\":\"...\",\"slots\":{...}}, ...]."
        ),
    }

    payload = (
        "SCHEMA_HINT:\n"
        + _safe_json(schema_hint)
        + "\n\nAWAITING_SLOT:\n"
        + ((awaiting_slot or "").strip() if awaiting_slot else "(none)")
        + "\n\nRECENT_TURNS:\n"
        + (recent_turns if recent_turns else "(none)")
        + "\n\nUSER_INPUT:\n"
        + (user_text or "")
        + "\n\nReturn ONLY the JSON."
    )

    prompt = format_chat(args, system_prompt, payload, tokenizer=tokenizer)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = enc.to(infer_input_device(model))

    out = generate(model, inputs, tokenizer, args).strip()
    obj = parsing_json(out)

    # Freeze downstream contract: list[dict]
    if obj is None:
        raw_mrs: List[Dict[str, Any]] = []
    elif isinstance(obj, dict):
        raw_mrs = [obj]
    elif isinstance(obj, list):
        raw_mrs = [x for x in obj if isinstance(x, dict)]
    else:
        raw_mrs = []

    if not raw_mrs:
        raw_mrs = [{"intent": "out_of_domain", "slots": {}}]

    # Validate + normalize, mirroring main.py behavior
    mrs: List[Dict[str, Any]] = []
    for m in raw_mrs:
        v = validate_mr(m)
        if v.valid:
            mrs.append(v.normalized_mr)
        else:
            mrs.append({"intent": "out_of_domain", "slots": {"ood_type": "INVALID_MR"}})

    return mrs
