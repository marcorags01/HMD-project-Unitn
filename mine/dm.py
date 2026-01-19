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
from intents_schema import normalize_mr


ALLOWED_DM_ACTIONS = {
    "request_info",
    "provide_info",
    "propose_menus",
    "set_active_menu",
    "show_day",
    "suggest_swap_day",
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
        nm = normalize_mr(mr)  # always present

        # 2) Compose system prompt from utils (Marina-like)
        extra_rules = """ADDITIONAL DM RESPONSIBILITY (you are the primary controller):
        - Your goal is a smooth, human conversation. Do NOT mention intents, slots, enums, variable names, or JSON.
        - Ask for ONE piece of information at a time when you need something.
        - Prefer natural, short questions. Avoid listing options unless the user asks.
        - If the user already provided multiple preferences in one message, do NOT re-ask them—only ask what is still missing.
        - Use RECENT_TURNS to interpret short answers (e.g., "1", "yes", "menu 2", "Tue") in context.

        FLOW GUIDELINES:
        - If the PLAN is incomplete, use request_info(<one_missing_item>) to ask the next best question.
          Suggested order: servings -> time_limit -> calorie_level -> avoid_items.
        - If the PLAN is complete and menus are not yet proposed (no menus exist yet), output propose_menus().
        - If menus are proposed but no active menu is selected, do not show/swap/update/confirm; ask for the menu choice with request_info(menu_id).
        - If an active menu exists:
        * show_day(target_day) for "what's on Tue" / "show Tue"
        * suggest_swap_day(target_day) when the user asks to "suggest/propose an alternative" for a day
            (this must NOT commit the change; the user must explicitly confirm)
        * swap_day(target_day) only when the user explicitly asks to swap/change a day
        * If the tracker shows a pending_action of type SWAP_DAY and the user says "yes/ok/do it",
            output swap_day(<the pending day>) to commit it
        * update_avoid(ADD_AVOID_ITEM, item) or update_avoid(REMOVE_AVOID_ITEM, item) for avoid changes
        * confirm_plan() when the user asks to finalize / shopping list and there is no pending suggested change


        STYLE (for request_info arguments):
        - request_info(servings): ask "How many servings should I plan for?"
        - request_info(time_limit): ask "Do you want quick meals, or is normal prep time OK?"
        - request_info(calorie_level): ask "Are you aiming for lighter, balanced, or more filling meals?"
        - request_info(avoid_items): ask "Any allergies or foods you want to avoid?"
        - request_info(menu_id): ask "Do you prefer option 1 or option 2?"
        - request_info(target_day): ask "Which day should I focus on—Mon, Tue, Wed, Thu, or Fri?"

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

                
        pending_mrs = []
        if hasattr(tracker, "pending_mrs"):
            try:
                pending_mrs = list(tracker.pending_mrs or [])
            except Exception:
                pending_mrs = []

        pending_intents = []
        for pmr in pending_mrs:
            if isinstance(pmr, dict):
                pending_intents.append(str(pmr.get("intent", "")).strip() or "out_of_domain")

        tracker_state = tracker.to_state_dict() if hasattr(tracker, "to_state_dict") else {}
        if isinstance(tracker_state, dict):
            tracker_state.pop("pending_mrs", None)  # avoid duplicating / flooding

        ds = {
            # Selected MR (the one main.py chose to address now)
            "selected_mr": nm,                 # preferred name going forward
            "mr": nm,                          # keep for backward-compatibility (optional)

            # Queue summary (do NOT drown the model in JSON)
            "selected_intent": str(nm.get("intent", "")).strip() or "out_of_domain",
            "pending_count": len(pending_mrs),
            "pending_intents": pending_intents,

            # (Optional but useful) include only a tiny “head” sample, not full queue
            "pending_head": pending_mrs[:3],

            # Existing state signals
            "menus_exist": tracker.menus_exist() if hasattr(tracker, "menus_exist") else False,
            "tracker": tracker_state,
            "missing_plan": tracker.missing_plan_slots() if hasattr(tracker, "missing_plan_slots") else [],
            "has_active_menu": tracker.has_active_menu() if hasattr(tracker, "has_active_menu") else False,
            "last_action": last_action or "",
        }

         #  DM debug logging for queue context 
        if bool(getattr(self.args, "debug", False)):
            sel_intent = str(nm.get("intent", "")).strip() or "out_of_domain"
            self.logger.debug(
                "DM queue context | selected_intent=%s | pending_count=%d | pending_intents=%s",
                sel_intent,
                len(pending_mrs),
                pending_intents,
            )



        user_text = (
            "RECENT_TURNS:\n"
            + (recent if recent else "(none)")
            + "\n\nSELECTED_MR_AND_STATE_JSON:\n"
            + _safe_json(ds)
            + "\n\nReturn the next action now."
        )

        # 4) Format with chat template (Marina-style: args.chat_template.format(system, user))
        dm_text = format_chat(self.args, system_prompt, user_text, tokenizer=self.tokenizer)
        if not isinstance(dm_text, str):
            raise TypeError(f"format_chat() must return str, got {type(dm_text)}: {repr(dm_text)[:200]}")

        self.logger.debug(f"DM input:\n{dm_text}")

        # 5) Generate
        if dm_text is None:
            dm_text = ""
        elif not isinstance(dm_text, str):
            dm_text = str(dm_text)

        enc = self.tokenizer([dm_text], return_tensors="pt")
        inputs = enc.to(self.model.device)


        dm_output = generate(self.model, inputs, self.tokenizer, self.args).strip()

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
