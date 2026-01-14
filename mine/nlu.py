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
                "If a slot value is missing, set it to null.",
                "Do not invent new information, but DO normalize synonyms/typos into the controlled values.",
                "Use RECENT_TURNS to resolve short answers (e.g., '1', 'fast', 'yes').",
                "Return ONLY JSON.",
            ],
        }

        system_prompt = (
            "You are the NLU component for a Meal Kit Composer assistant.\n"
            "Task: extract EXACTLY ONE intent and its slot-value pairs from the user's text.\n"
            "\n"
            "Core rules:\n"
            "- Output MUST be valid JSON and MUST match EXACTLY: {\"intent\":\"...\",\"slots\":{...}}\n"
            "- Extract EXACTLY ONE intent.\n"
            "- Only fill slots that belong to that intent.\n"
            "- If a slot value is missing, output null.\n"
            "- Do not invent values or add defaults that the user did not state.\n"
            "- Return ONLY the JSON object. No extra text.\n"
            "\n"
            "Dialogue context rules (use RECENT_TURNS):\n"
            "- If the last assistant message asked for a specific detail (e.g., servings/time/calories/menu choice/day),\n"
            "  and the user replies with only a value (e.g., \"1\", \"two\", \"fast\", \"Tuesday\", \"menu 1\", \"yes\"),\n"
            "  interpret it as answering that question and fill the corresponding slot.\n"
            "- If the user replies \"yes/ok/fine\" after a confirmation request, interpret as intent='confirm'.\n"
            "\n"
            "Canonicalization (normalize user language into controlled values):\n"
            "- servings: output an integer 1..6 (map words one/two/three/four/five/six to 1..6).\n"
            "- time_limit: output FAST or NORMAL (map quick/short/asap to FAST; regular/standard to NORMAL).\n"
            "- calorie_level: output LOW, MED, or HIGH (map medium/balanced/average to MED).\n"
            "- target_day: output one of Mon,Tue,Wed,Thu,Fri (map full names like Monday->Mon).\n"
            "- menu_id: output 1 or 2.\n"
            "- If there is an obvious typo and confidence is high (e.g., \"fats\"->FAST), correct it.\n"
            "\n"
            "Special help intent rule:\n"
            "- If the user asks what values/options are allowed/supported for something, use intent='help'\n"
            "  and set slots.slot to the relevant item (e.g., 'servings', 'time_limit', 'calorie_level', 'avoid_items', or 'all').\n"
            "- For help.intent, set it to the most relevant context among: plan/select_menu/inspect/refine/confirm;\n"
            "  if unclear, set it to null.\n"
        )


        # Dialogue context to resolve short/elliptical answers (e.g., "1", "yes", "ok").
        recent = ""
        if self.history is not None:
            try:
                recent = self.history.last_iterations(last_n=6)
            except TypeError:
                # if last_iterations() doesn't accept last_n in your History implementation
                recent = self.history.last_iterations()

        user_payload = (
            "SCHEMA_HINT:\n"
            + _safe_json(schema_hint)
            + "\n\nRECENT_TURNS:\n"
            + (recent if recent else "(none)")
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
