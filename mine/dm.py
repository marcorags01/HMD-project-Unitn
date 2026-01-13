# dm.py
"""
Meal Kit Composer — Dialogue Manager (LLM-routed), Marina-style and lightweight.

Responsibilities:
- Build a DM prompt from utils.PROMPTS
- Provide the model with: normalized MR + tracker state + recent turns
- Ask the LLM to output exactly ONE compact action string: action(args)
- Parse and return (action, argument, debug_text)

Non-responsibilities (handled elsewhere, per compartmentalization plan):
- Deterministic guard rails (component/policy.py)
- Action execution (main loop / executor layer calling support_fn domain services)
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from utils import PROMPTS, generate, format_chat
from support_fn import extract_action_and_argument
from intents_schema import validate_mr


ALLOWED_DM_ACTIONS = {
    "request_info",
    "provide_info",
    "propose_menus",
    "set_active_menu",
    "show_day",
    "swap_day",
    "update_avoid",
    "confirm_plan",
    "fallback",
}


def _safe_json(obj: Any) -> str:
    """JSON dump that won't crash on occasional non-serializable objects."""
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=str)


class DM:
    def __init__(self, history, model, tokenizer, args, logger):
        self.history = history
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.logger = logger

    def __call__(
        self,
        tracker,
        mr: Dict[str, Any],
        last_action: str = "",
        last_n_turns: Optional[int] = None,
    ) -> Tuple[str, str, str]:
        """
        Returns:
          action: str
          argument: str (raw inside parentheses; may contain commas)
          debug_input: str (the full formatted prompt sent to the model)

        Notes:
        - This DM does not mutate tracker or call domain services.
        - Deterministic guard rails are intentionally not here (to keep DM small).
        """

        # 1) Normalize MR (schema-level). We do not enforce DM workflow here.
        vr = validate_mr(mr)
        nm = vr.normalized_mr  # always present

        # 2) Compose system prompt from utils (Marina-like)
        extra_rules = """ADDITIONAL DM RESPONSIBILITY (you are the primary controller):
        - If the PLAN is complete and tracker.phase is AWAITING_PLAN (menus not yet proposed), output propose_menus().
        - If menus are proposed but no active menu is selected, do not show/swap/update/confirm; request_info(menu_id).
        
        STRICT OUTPUT FORMAT (important):
        - Output exactly one line: action(arg1, arg2) or action() if no args.
        - Use POSITIONAL arguments only. Do NOT use key=value.
        - Do NOT add quotes, backticks, code fences, or any other text.
        """

        system_prompt = (
            PROMPTS["DM_START"]
            + "\n"
            + PROMPTS["DM_ACTIONS"]
            + "\n"
            + PROMPTS["DM_RULES"]
            + "\n"
            + extra_rules
            + "\n"
            + PROMPTS["DM_END"]
        )

        # 3) Compose user text: include action context + recent turns + DS snapshot
        recent = ""
        if self.history is not None:
            if last_n_turns is None:
                recent = self.history.last_iterations()
            else:
                recent = self.history.last_iterations(last_n=last_n_turns)

        ds = {
            "mr": nm,
            "mr_valid": bool(vr.valid),
            "mr_errors": vr.errors,
            "tracker": tracker.to_state_dict() if hasattr(tracker, "to_state_dict") else {},
            "last_action": last_action or "",
        }

        user_text = (
            "RECENT_TURNS:\n"
            + (recent if recent else "(none)")
            + "\n\nDIALOGUE_STATE_JSON:\n"
            + _safe_json(ds)
            + "\n\nReturn the next action now."
        )

        # 4) Format with chat template (Marina-style: args.chat_template.format(system, user))
        dm_text = format_chat(self.args, system_prompt, user_text, tokenizer=self.tokenizer)

        self.logger.debug(f"DM input:\n{dm_text}")

        # 5) Generate
        dm_inputs = self.tokenizer(dm_text, return_tensors="pt").to(self.model.device)
        dm_output = generate(self.model, dm_inputs, self.tokenizer, self.args).strip()

        # Many models may add trailing newlines; keep first non-empty line if present
        dm_output_line = ""
        for line in dm_output.splitlines():
            if line.strip():
                dm_output_line = line.strip()
                break
        if not dm_output_line:
            dm_output_line = dm_output.strip()

        self.logger.debug(f"DM raw output: {dm_output_line}")

        # 6) Parse action(args)
        parsed = extract_action_and_argument(dm_output_line)
        if not parsed:
            # If parsing fails, degrade gracefully.
            return "fallback", "", dm_text

        action, argument = parsed
        action = (action or "").strip().lower()

        if action not in ALLOWED_DM_ACTIONS:
            # If action is not in our closed set, degrade gracefully.
            return "fallback", "", dm_text

        # argument is allowed to be empty for propose_menus()/confirm_plan()/fallback()
        argument = (argument or "").strip()
        return action, argument, dm_text
