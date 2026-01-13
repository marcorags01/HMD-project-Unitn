# nlg.py
"""
Meal Kit Composer — NLG (LLM-based), Marina-style.

Responsibilities:
- Given final action + argument + payload (+ tracker snapshot), generate a user-facing message.
- Must NOT output the action string.
- Must be compatible with your closed action set:
  request_info, provide_info, propose_menus, set_active_menu, show_day,
  swap_day, update_avoid, confirm_plan, fallback

Non-responsibilities:
- Policy (hard rules) -> policy.py
- Execution + menu generation + shopping list -> main.py / support_fn.py
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from utils import PROMPTS, generate, format_chat


_ACTION_GUIDE = """ACTION SEMANTICS:
- request_info(slot): Ask a single clear question to obtain that missing slot value. Do not list options unless asked.
- provide_info(intent, slot): Provide allowed values for that slot only, concise.
- propose_menus(): Present two menu options (Mon–Fri). Ask user to choose menu_id 1 or 2.
- set_active_menu(menu_id): Confirm selection and explain next possible actions (inspect/refine/confirm).
- show_day(target_day): Show recipe title, time, calorie level, and ingredients (scaled).
- swap_day(target_day): Confirm the swap (or explain no alternative exists).
- update_avoid(op, value): Confirm updated avoid list and mention repaired days if any.
- confirm_plan(): Confirm and present the shopping list clearly.
- fallback(): Explain capabilities succinctly.

STYLE:
- Be concise and task-focused.
- No chitchat.
- Use bullet points for menus and shopping lists where helpful.
- If payload contains an "error", prioritize returning that error message clearly.
"""


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _strip_accidental_action_echo(text: str) -> str:
    """
    If the model mistakenly starts by echoing something like 'show_day(Tue)',
    drop that first line.
    """
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    if not lines:
        return text.strip()

    first = lines[0].strip()
    if re.match(r"^\w+\(.*\)$", first):
        # Drop first line, keep the rest (can be multi-line).
        return "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
    return text.strip()


class NLG:
    def __init__(self, history, model, tokenizer, args, logger):
        self.history = history
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.logger = logger

    def __call__(
        self,
        action: str,
        argument: str,
        tracker_state: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        last_n_turns: Optional[int] = None,
    ) -> str:
        payload = payload or {}

        # Build system prompt
        system_prompt = (
            PROMPTS["NLG_START"]
            + "\n"
            + _ACTION_GUIDE
            + "\n"
            + PROMPTS["NLG_END"]
        )

        # Recent turns (optional)
        recent = ""
        if self.history is not None:
            if last_n_turns is None:
                recent = self.history.last_iterations()
            else:
                recent = self.history.last_iterations(last_n=last_n_turns)

        # Provide the model everything needed deterministically
        bundle = {
            "action": action,
            "argument": argument,
            "tracker_state": tracker_state or {},
            "payload": payload,
            "recent_turns": recent if recent else "(none)",
        }

        user_text = (
            "INPUT_BUNDLE_JSON:\n"
            + _safe_json(bundle)
            + "\n\nGenerate the user-facing message now. Return ONLY the message."
        )

        nlg_text = format_chat(self.args, system_prompt, user_text, tokenizer=self.tokenizer)

        self.logger.debug(f"NLG input:\n{nlg_text}")

        inputs = self.tokenizer(nlg_text, return_tensors="pt").to(self.model.device)
        out = generate(self.model, inputs, self.tokenizer, self.args).strip()

        out = _strip_accidental_action_echo(out)
        if not out:
            # safety fallback
            return "Sorry—something went wrong while generating the response. Could you repeat that?"

        return out
