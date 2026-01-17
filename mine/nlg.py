# nlg.py
"""
Meal Kit Composer — NLG (LLM-based), Marina-style.

Responsibilities:
- Given final action + argument + payload (+ tracker snapshot), generate a user-facing message.
- Must NOT output the action string.
- Must be compatible with your closed action set:
  request_info, provide_info, propose_menus, set_active_menu, suggest_swap_day, show_day,
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


_ACTION_GUIDE = """YOU WRITE THE USER-FACING MESSAGE.

Hard requirements:
- Never mention internal system concepts (e.g., intents, slots, enums, IDs, variables, JSON).
- Do not output action names or action syntax.
- Keep the tone human, brief, and helpful.

If the input bundle contains any of these blocks:
- MENU_BLOCK
- DAY_BLOCK
- SHOPPING_BLOCK
- FACT_BLOCK

You MUST include each non-empty block EXACTLY as provided (verbatim).
Do not edit, reformat, reorder, paraphrase, or “improve” the block.
You may add a short sentence before and/or after the block if helpful.
Do not add additional markdown formatting around these blocks (no extra headings/bold beyond what is already inside).

General behavior:
- Ask for only one missing detail at a time.
- If the user confirms something (“yes”, “ok”, “that’s fine”), acknowledge and proceed naturally.
- Use bullet points when presenting options or lists, but do not alter the provided blocks.
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

_REQUEST_QUESTIONS = {
    "servings": "How many servings should I plan for? (1–6)",
    "time_limit": "Do you want quick meals, or is normal prep time OK?",
    "calorie_level": "Are you aiming for lighter meals, balanced, or more filling?",
    "avoid_items": "Any allergies or foods you want to avoid?",
    "menu_id": "Which option do you prefer—1 or 2?",
    "target_day": "Which day should I focus on—Mon, Tue, Wed, Thu, or Fri?",
    "refine_type": "Do you want to swap a day, or update foods to avoid?",
    "value": "What should I add or remove from foods to avoid? (e.g., nuts, dairy, gluten)",
    "all": "What would you like to do next?",
}

def _render_provide_info(payload: Dict[str, Any], tracker_state: Dict[str, Any]) -> str:
    # Prefer payload values (executor-supplied), fall back to tracker_state
    help_intent = str(payload.get("help_intent", "plan") or "plan").strip().lower() # (Currently unused; reserved for future intent-specific help variations.)
    help_slot = str(payload.get("help_slot", "all") or "all").strip().lower()

    phase = str(payload.get("phase") or tracker_state.get("phase") or "")
    has_active = bool(payload.get("has_active_menu", False))
    if "has_active_menu" not in payload:
        has_active = bool((tracker_state or {}).get("active_menu_id") in (1, 2))  # executor already computed it

    # Slot == all: phase-aware guidance (this is your old _provide_info_message logic)
    if help_slot in {"all", ""}:
        if phase == "AWAITING_PLAN":
            return (
                "I can help you plan weekday dinners (Mon–Fri) and generate a shopping list.\n"
                "To start: how many servings should I plan for?"
            )

        if phase == "AWAITING_MENU_SELECTION":
            return (
                "I’ve generated two options. Which do you prefer—1 or 2?"
            )

        if has_active or phase == "ACTIVE_MENU":
            return (
                "For your current plan, you can:\n"
                "- ask what’s planned on a day (e.g., “What’s on Tue?”)\n"
                "- swap a day (e.g., “Swap Wed”)\n"
                "- add/remove foods to avoid (e.g., “Avoid nuts”)\n"
                "- confirm to get the shopping list\n"
                "What would you like to do next?"
            )

        if phase == "CONFIRMED":
            return (
                "Your plan is already confirmed and the shopping list is ready.\n"
                "You can start a new plan (e.g., “Plan meals for 2 people, quick, balanced”), or type “exit”."
            )

        return "How can I help with your meal plan?"

    # Slot-specific help (same strings as you had in main.py)
    if help_slot == "servings":
        return _REQUEST_QUESTIONS["servings"]
    if help_slot == "time_limit":
        return _REQUEST_QUESTIONS["time_limit"]
    if help_slot == "calorie_level":
        return _REQUEST_QUESTIONS["calorie_level"]
    if help_slot == "avoid_items":
        return _REQUEST_QUESTIONS["avoid_items"]
    if help_slot == "menu_id":
        return _REQUEST_QUESTIONS["menu_id"]
    if help_slot == "target_day":
        return _REQUEST_QUESTIONS["target_day"]

    return _REQUEST_QUESTIONS["all"]


def _fmt_constraints(tracker_state: Dict[str, Any]) -> str:
    c = (tracker_state or {}).get("constraints") or {}
    servings = c.get("servings")
    time_limit = str(c.get("time_limit") or "").upper()
    calorie = str(c.get("calorie_level") or "").upper()
    avoids = c.get("avoid_items") or []

    time_map = {"FAST": "Quick", "NORMAL": "Normal"}
    cal_map = {"LOW": "Lighter", "MED": "Balanced", "HIGH": "More filling"}

    lines = []
    if servings is not None:
        lines.append(f"- Servings: {servings}")
    if time_limit:
        lines.append(f"- Prep time: {time_map.get(time_limit, time_limit.title())}")
    if calorie:
        lines.append(f"- Calories: {cal_map.get(calorie, calorie.title())}")

    avoids_txt = ", ".join(avoids) if avoids else "none"
    lines.append(f"- Foods to avoid: {avoids_txt}")
    return "\n".join(lines)



def _render_menus(menu1_pretty: Dict[str, str], menu2_pretty: Dict[str, str]) -> str:
    def fmt(menu: Dict[str, str]) -> str:
        lines = []
        for d in _WEEK_DAYS:
            if d in menu:
                lines.append(f"- {d}: {menu[d]}")
        return "\n".join(lines) if lines else "(no items)"

    return (
        "Two weekly options (Mon–Fri):\n\n"
        "Option 1\n" + fmt(menu1_pretty) + "\n\n"
        "Option 2\n" + fmt(menu2_pretty) + "\n\n"
        "Which option do you prefer—1 or 2?"
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
    day = details.get("day", "")
    title = details.get("title", "")
    time_min = details.get("time_min", "")
    cal = str(details.get("calorie_level", "") or "").upper()
    avoid_check = bool(details.get("avoid_check", False))
    ings = details.get("ingredients", []) or []

    cal_map = {"LOW": "Lighter", "MED": "Balanced", "HIGH": "More filling"}

    lines = [
        f"{day} — {title}",
        f"- Time: {time_min} min",
    ]
    if cal:
        lines.append(f"- Calories: {cal_map.get(cal, cal.title())}")

    if avoid_check:
        lines.append("- Note: this includes something you asked to avoid.")

    lines.append("")
    lines.append("Ingredients (scaled):")
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

        # ------------------------- Hard short-circuits -------------------------
        if payload.get("error"):
            return str(payload["error"])
        
        if action == "provide_info":
            return _render_provide_info(payload, tracker_state)


        if action == "fallback" and phase == "CONFIRMED":
            return "All set — your meal plan is finalized. Type 'exit' to end, or start a new plan anytime."

        # ------------------------- Build verbatim factual blocks -------------------------
        menu_block = ""
        day_block = ""
        shopping_block = ""
        fact_block = ""

        # If executor produced a direct message (e.g., help), treat it as factual content.
        if payload.get("message") and action != "provide_info" :
            fact_block = str(payload["message"]).strip()

        # Request questions are best kept deterministic (but wrapped by LLM).
        if action == "request_info":
            q = _REQUEST_QUESTIONS.get((argument or "").strip(), "")
            if q:
                return q if q else "What would you like to do next?"

        # Menus
        if action == "propose_menus" and payload.get("menu1_pretty") and payload.get("menu2_pretty"):
            menu_block = _render_menus(payload["menu1_pretty"], payload["menu2_pretty"])

        # Day details
        if action == "show_day" and payload.get("details"):
            day_block = _render_day_details(payload["details"])

        # Shopping list
        if action == "confirm_plan" and payload.get("shopping_list") is not None:
            header = "Shopping list (Mon–Fri):\n\n" + _fmt_constraints(tracker_state) + "\n\n"
            body = _render_shopping_list(payload.get("shopping_list") or [])
            shopping_block = (header + body).strip()
            return shopping_block  # direct return for shopping list

        # Menu selection acknowledgement (keep facts correct, let LLM add tone)
        if action == "set_active_menu":
            if payload.get("ok"):
                mid = tracker_state.get("active_menu_id") or argument
                fact_block = (
                    f"Great — we’ll go with option {mid}.\n"
                    "You can ask what’s planned on a day, swap a day, update foods to avoid, or confirm for the shopping list."
                )
            else:
                fact_block = "I couldn’t select that option. Please reply with 1 or 2."

        # Suggest swap (non-committing) acknowledgement (factual)
        if action == "suggest_swap_day":
            if payload.get("suggested"):
                # argument is the day (e.g., "Mon")
                fact_block = (
                    f"Sure! Here's an alternative meal for {argument}:\n\n"
                    f"- {payload.get('suggested_title', 'a new recipe')}\n\n"
                    f"Do you want me to swap {argument} to this?"
                )
            else:
                fact_block = "I couldn’t find a good alternative for that day with your current preferences."


        # Swap acknowledgement (factual)
        if action == "swap_day":
            if payload.get("swapped"):
                fact_block = f"Done — I swapped {argument} to: {payload.get('new_title', 'a new recipe')}."
            else:
                fact_block = "I couldn’t find a good alternative for that day with your current preferences."

        # Avoid update acknowledgement (factual)
        if action == "update_avoid":
            repaired = payload.get("repaired_days") or []
            base = "Got it — I updated your foods to avoid.\n\n" + _fmt_constraints(tracker_state)
            if repaired:
                base += "\n\nI also updated these days to keep everything compatible: " + ", ".join(repaired)
            fact_block = base
   

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
            "MENU_BLOCK": menu_block,
            "DAY_BLOCK": day_block,
            "SHOPPING_BLOCK": shopping_block,
            "FACT_BLOCK": fact_block,
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
