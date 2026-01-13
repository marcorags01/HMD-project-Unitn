# nlu.py
"""
Meal Kit Composer — NLU (LLM-based), Marina-style.

Responsibilities:
- Given user text, produce EXACTLY ONE MR in the flat JSON format:
    {"intent": "...", "slots": {...}}
- Do not invent values; if missing, output null
- Normalize and validate using intents_schema.validate_mr
- Return the normalized MR even if invalid (for robustness)

Compatible with:
- utils.generate + args.chat_template
- intents_schema.INTENT_SLOTS, intents_schema.validate_mr
- support_fn.parsing_json
- support_classes constants for controlled vocab (via schema_hint)
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from utils import generate, format_chat
from intents_schema import INTENT_SLOTS, validate_mr
from support_fn import parsing_json
from support_classes import (
    ALLOWED_DAYS,
    ALLOWED_TIME_LIMITS,
    ALLOWED_CALORIE_LEVELS,
    ALLOWED_AVOID_ITEMS,
)


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=str)


class NLU:
    """
    Minimal LLM-based NLU that outputs one MR per user turn.

    Design choices (aligned with your blueprint + current architecture):
    - single intent only (no multi-intent splitting)
    - controlled vocab guidance via schema_hint
    - validation/normalization via intents_schema.validate_mr
    """

    def __init__(self, history, model, tokenizer, args, logger):
        self.history = history
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.logger = logger

    def __call__(self, user_text: str) -> Dict[str, Any]:
        # A compact but explicit schema hint helps LLM stay grounded.
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
                "menu_id": [1, 2],
                "help_slot": [
                    "servings", "time_limit", "calorie_level", "avoid_items",
                    "menu_id", "target_day", "refine_type", "value",
                    "all",
                ],
                "help_intent": ["plan", "select_menu", "inspect", "refine", "confirm"],
            },
    
            "output_format": {"intent": "...", "slots": {"slot": "value_or_null"}},
            "rules": [
                "Extract EXACTLY ONE intent.",
                "Only fill slots that belong to that intent.",
                "If a slot is not explicitly stated, set it to null.",
                "Never invent defaults.",
                "Return ONLY JSON.",
            ],
        }

        system_prompt = (
            "You are the NLU component for a Meal Kit Composer assistant.\n"
            "Task: extract EXACTLY ONE intent and its slot-value pairs from the user's text.\n"
            "Rules:\n"
            "- If a slot value is not explicitly provided, output null.\n"
            "- Do not invent values. Do not assume defaults.\n"
            "- Only use the provided intents/slots and controlled values.\n"
            'Output format must be EXACTLY: {"intent":"...","slots":{...}}\n'
            "Return ONLY the JSON object. No extra text.\n"
            "Special intent rule:\n"
            "- If the user asks what values/options are allowed/supported for a slot, "
            "use intent='help' and set slots.slot to that slot name.\n"
            "- If the user asks general help (e.g., 'what can I do?'), use intent='help' with slots.slot='all'.\n"
            "- For help.intent, set it to the most relevant context among: plan/select_menu/inspect/refine/confirm; "
            "if unclear, set it to null.\n"
            'Output format must be EXACTLY: {"intent":"...","slots":{...}}\n'
            "Return ONLY the JSON object. No extra text.\n"
        )

        user_payload = (
            "SCHEMA_HINT:\n"
            + _safe_json(schema_hint)
            + "\n\nUSER_INPUT:\n"
            + (user_text or "")
            + "\n\nReturn ONLY the JSON."
        )

        nlu_text = format_chat(self.args, system_prompt, user_payload, tokenizer=self.tokenizer)

        self.logger.debug(f"NLU input:\n{nlu_text}")

        inputs = self.tokenizer(nlu_text, return_tensors="pt").to(self.model.device)
        out = generate(self.model, inputs, self.tokenizer, self.args).strip()

        self.logger.debug(f"NLU raw output: {out}")

        mr = parsing_json(out)
        if not mr or not isinstance(mr, dict):
            # Hard fallback: keep pipeline alive
            return {"intent": "out_of_domain", "slots": {}}

        vr = validate_mr(mr)
        if not vr.valid:
            self.logger.debug(f"NLU validation errors: {vr.errors}")

        return vr.normalized_mr
