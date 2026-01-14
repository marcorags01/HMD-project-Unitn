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
from collections import defaultdict


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

# ------------------------- Deterministic renderers -------------------------

_WEEK_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]

def _fmt_constraints(tracker_state: Dict[str, Any]) -> str:
    c = (tracker_state or {}).get("constraints") or {}
    servings = c.get("servings")
    time_limit = c.get("time_limit")
    calorie = c.get("calorie_level")
    avoids = c.get("avoid_items") or []
    avoids_txt = ", ".join(avoids) if avoids else "none"
    lines = []
    if servings is not None:
        lines.append(f"- Servings: {servings}")
    if time_limit is not None:
        lines.append(f"- Time limit: {time_limit}")
    if calorie is not None:
        lines.append(f"- Calorie level: {calorie}")
    lines.append(f"- Avoid items: {avoids_txt}")
    return "\n".join(lines)


def _render_menus(menu1_pretty: Dict[str, str], menu2_pretty: Dict[str, str]) -> str:
    def fmt(menu: Dict[str, str]) -> str:
        lines = []
        for d in _WEEK_DAYS:
            if d in menu:
                lines.append(f"- {d}: {menu[d]}")
        return "\n".join(lines) if lines else "(no items)"

    return (
        "Here are two menu options (Mon–Fri):\n\n"
        "**Menu 1**\n" + fmt(menu1_pretty) + "\n\n"
        "**Menu 2**\n" + fmt(menu2_pretty) + "\n\n"
        "Reply with the menu ID you want (1 or 2)."
    )


def _render_shopping_list(items: list[Dict[str, Any]]) -> str:
    # items: [{name, qty, unit, category}, ...]
    grouped: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for it in items or []:
        cat = str(it.get("category", "Other") or "Other").strip()
        grouped[cat].append(it)

    out_lines: list[str] = []
    for cat in sorted(grouped.keys(), key=lambda s: s.lower()):
        out_lines.append(f"**{cat}**")
        grouped[cat].sort(key=lambda x: str(x.get("name", "")).lower())
        for it in grouped[cat]:
            name = str(it.get("name", "")).strip()
            qty = it.get("qty", "")
            unit = str(it.get("unit", "")).strip()

            qty_txt = ""
            if qty != "" and qty is not None:
                # qty is often float; render cleanly
                try:
                    q = float(qty)
                    qty_txt = str(int(q)) if abs(q - int(q)) < 1e-9 else str(q)
                except Exception:
                    qty_txt = str(qty)

            left = " ".join([t for t in [qty_txt, unit] if t]).strip()
            out_lines.append(f"- {left} {name}".strip() if left else f"- {name}")
        out_lines.append("")

    return "\n".join(out_lines).strip() if out_lines else "(empty)"


def _render_day_details(details: Dict[str, Any]) -> str:
    # details: {day, title, time_min, calorie_level, ingredients:[...], avoid_check:bool}
    day = details.get("day", "")
    title = details.get("title", "")
    time_min = details.get("time_min", "")
    cal = details.get("calorie_level", "")
    avoid_check = details.get("avoid_check", False)
    ings = details.get("ingredients", []) or []

    lines = [
        f"**{day} — {title}**",
        f"- Time: {time_min} min",
        f"- Calorie level: {cal}",
    ]
    if avoid_check:
        lines.append("- Note: this recipe contains at least one avoided tag.")

    lines.append("\nIngredients (scaled):")
    for ing in ings:
        name = str(ing.get("name", "")).strip()
        qty = ing.get("qty", "")
        unit = str(ing.get("unit", "")).strip()
        lines.append(f"- {qty} {unit} {name}".strip())

    return "\n".join(lines).strip()

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
               
        tracker_state = tracker_state or {}
        phase = str(tracker_state.get("phase", "") or "")

        # ------------------------- Deterministic short-circuits -------------------------
        # If execution produced an error or a direct message, return it verbatim.
        if payload.get("error"):
            return str(payload["error"])

        if payload.get("message"):
            return str(payload["message"])

        # Render menus deterministically (executor provides menu*_pretty).
        if action == "propose_menus" and payload.get("menu1_pretty") and payload.get("menu2_pretty"):
            return _render_menus(payload["menu1_pretty"], payload["menu2_pretty"])

        # Render shopping list deterministically (executor provides shopping_list).
        if action == "confirm_plan" and payload.get("shopping_list") is not None:
            header = "Here is your shopping list (Mon–Fri):\n\n"
            header += _fmt_constraints(tracker_state) + "\n\n"
            body = _render_shopping_list(payload.get("shopping_list") or [])
            footer = "\n\nIf you are done, you can say 'finalize' or 'exit'."
            return header + body + footer

        # Deterministic inspect rendering
        if action == "show_day" and payload.get("details"):
            return _render_day_details(payload["details"])

        # Deterministic swap rendering
        if action == "swap_day":
            if payload.get("swapped"):
                return f"Done — I swapped **{argument}** to: **{payload.get('new_title', 'a new recipe')}**."
            return f"I couldn't find a feasible alternative for **{argument}** under the current constraints."

        # Deterministic avoid update rendering
        if action == "update_avoid":
            repaired = payload.get("repaired_days") or []
            msg = "Updated your avoid list.\n\n" + _fmt_constraints(tracker_state)
            if repaired:
                msg += "\n\nRepaired days: " + ", ".join(repaired)
            return msg

        # Deterministic menu selection acknowledgement
        if action == "set_active_menu":
            if payload.get("ok"):
                mid = tracker_state.get("active_menu_id")
                return (
                    f"Menu **{mid}** selected.\n"
                    "You can:\n"
                    "- inspect a day (e.g., 'inspect Tue')\n"
                    "- swap a day (e.g., 'swap Wed')\n"
                    "- update avoid items (e.g., 'avoid nuts')\n"
                    "- confirm to get the shopping list"
                )
            return "I couldn’t select that menu. Please reply with menu ID 1 or 2."

        # In CONFIRMED, fallback should not restart; provide closure guidance.
        if action == "fallback" and phase == "CONFIRMED":
            return "All set — your meal plan is finalized. Type 'exit' to end, or 'plan a meal' to start a new one."

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
