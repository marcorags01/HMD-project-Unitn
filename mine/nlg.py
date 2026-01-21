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
from typing import Any, Dict, Optional, List

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
- Do NOT repeat or summarize previous day details unless the action is show_day or the user explicitly asked to repeat them.
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





def _has_unfulfilled_user_intents(tracker_state: Dict[str, Any]) -> bool:
    """
    True if there are still pending (non out_of_domain) user intents to fulfill.
    Assumes tracker_state may contain pending_mrs as a list of MRs.
    """
    pending = (tracker_state or {}).get("pending_mrs") or []
    if not isinstance(pending, list):
        return False

    for mr in pending:
        if not isinstance(mr, dict):
            continue
        it = str(mr.get("intent", "") or "").strip()
        if it and it != "out_of_domain":
            return True

    return False


def _should_offer_next_steps(action: str, tracker_state: Dict[str, Any]) -> bool:
    """
    Offer next-steps prompt after fulfilling inspect/refine/show_week,
    only if nothing else is pending and plan isn't confirmed.
    """
    phase = str((tracker_state or {}).get("phase", "") or "")
    if phase == "CONFIRMED":
        return False

    # Eligible "completed" actions
    if action not in {"show_day", "swap_day", "update_avoid", "show_week"}:
        return False

    # If we're waiting for swap confirmation (after suggest_swap_day), do not offer finalize prompt
    pending_action = (tracker_state or {}).get("pending_action")
    if isinstance(pending_action, dict) and pending_action.get("type"):
        return False

    # If there are other pending intents, don't prompt yet
    if _has_unfulfilled_user_intents(tracker_state):
        return False

    return True


def _append_next_steps_prompt(text: str) -> str:
    """
    Append a single, consistent follow-up prompt, avoiding duplication.
    """
    if not isinstance(text, str):
        return str(text)

    low = text.lower()
    if "what else can i do for you" in low or "finalize the plan" in low:
        return text

    follow = (
        "What else can I do for you? If everything looks good, you can confirm to finalize the plan and get the shopping list."
    )
    return (text.rstrip() + "\n\n" + follow).strip()



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



def _render_request_info(slot: str, tracker_state: Dict[str, Any]) -> str:
    """
    Deterministic slot reprompt with escalation based on tracker_state:
    - reprompt_count = 0: base question
    - reprompt_count = 1: repair + examples / allowed values
    - reprompt_count >= 2: repair + help/restart hint + repeat question
    Escalation only applies when tracker_state.awaiting_slot matches `slot`.
    """
    slot = str(slot or "").strip()
    base = _REQUEST_QUESTIONS.get(slot, _REQUEST_QUESTIONS.get("all", "What would you like to do next?"))

    awaiting = str((tracker_state or {}).get("awaiting_slot") or "").strip()

    try:
        reprompt_count = int((tracker_state or {}).get("reprompt_count") or 0)
    except Exception:
        reprompt_count = 0

    # Only escalate when we are *actually* awaiting this same slot.
    if not awaiting or awaiting != slot:
        reprompt_count = 0

    examples = {
        "servings": "For example: 1, 2, 4, or 6.",
        "time_limit": "Reply “quick” or “normal”.",
        "calorie_level": "Reply “lighter”, “balanced”, or “more filling”.",
        "avoid_items": "For example: nuts, dairy, gluten — or “none”.",
        "menu_id": "Reply with 1 or 2.",
        "target_day": "Reply with Mon, Tue, Wed, Thu, or Fri.",
        "refine_type": "Reply “swap a day” or “update foods to avoid”.",
        "value": "For example: “add nuts” or “remove dairy”.",
    }

    ex = examples.get(slot, "")

    if reprompt_count <= 0:
        return base

    if reprompt_count == 1:
        # Second attempt: repair + examples + repeat the question (keeps it clear and deterministic)
        tail = f" {ex}" if ex else ""
        return f"Sorry — I didn’t catch that.{tail}\n\n{base}".strip()

    # Third+ attempt: help/restart hint + repeat question
    tail = f"\n\n{ex}" if ex else ""
    return (
        "I’m still missing that detail. "
        "Type “help” for examples, or “restart” to start over.\n\n"
        f"{base}{tail}"
    ).strip()

def _render_phase_aware_fallback(tracker_state: Dict[str, Any]) -> str:
    """
    Deterministic fallback copy that is phase-aware and not over-informative.
    Slot-aware behavior is handled by the caller (awaiting_slot check).
    """
    phase = str((tracker_state or {}).get("phase", "") or "")

    if phase == "ACTIVE_MENU":
        return (
            "I didn’t quite get that. You can:\n"
            "- ask what’s planned on a day (e.g., “What’s on Tue?”)\n"
            "- swap a day (e.g., “Swap Wed”)\n"
            "- add/remove foods to avoid (e.g., “Avoid nuts”)\n"
            "- confirm to get the shopping list\n"
            "What would you like to do?"
        )

    if phase == "AWAITING_MENU_SELECTION":
        return "Which option do you prefer—1 or 2?"

    if phase == "AWAITING_PLAN":
        # Deterministic “collapse back to slot filling” without needing extra state:
        # pick the first missing constraint in a fixed order.
        c = (tracker_state or {}).get("constraints") or {}
        if c.get("servings") in (None, ""):
            return _render_request_info("servings", tracker_state)
        if str(c.get("time_limit") or "").strip() == "":
            return _render_request_info("time_limit", tracker_state)
        if str(c.get("calorie_level") or "").strip() == "":
            return _render_request_info("calorie_level", tracker_state)
        avoid_items = c.get("avoid_items")
        if avoid_items is None or avoid_items == "":
            return _render_request_info("avoid_items", tracker_state)

        # If everything looks filled but we’re still in AWAITING_PLAN, be minimally helpful:
        return "Tell me your preferences for servings, prep time, calories, and any foods to avoid."

    # Default generic fallback (only when no better deterministic guidance exists)
    return "What would you like to do next?"



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

def _render_week_overview(payload: Dict[str, Any], tracker_state: Dict[str, Any]) -> str:
    """
    Compact week view (Mon–Fri), similar in spirit to menu options:
    day + recipe title + calorie level + prep time + avoid items (and optional conflicts).
    """
    rows = payload.get("week_overview") or []
    if not isinstance(rows, list) or not rows:
        return "I couldn’t show the weekly plan. Do you want option 1 or option 2?"

    cal_map = {"LOW": "Lighter", "MED": "Balanced", "HIGH": "More filling"}

    header = "Here’s your week plan (Mon–Fri):\n\n" + _fmt_constraints(tracker_state) + "\n\n"

    lines: list[str] = []
    for r in rows:
        if not isinstance(r, dict):
            continue

        day = str(r.get("day", "") or "").strip()
        title = str(r.get("title", "") or "").strip()

        time_min = r.get("time_min", None)
        cal = str(r.get("calorie_level", "") or "").upper()

        avoid_hits = r.get("avoid_hits") or []
        if not isinstance(avoid_hits, list):
            avoid_hits = []

        # Render a compact single-line summary per day
        meta_parts: list[str] = []
        if time_min is not None and str(time_min).strip() != "":
            meta_parts.append(f"{time_min} min")
        if cal:
            meta_parts.append(cal_map.get(cal, cal.title()))

        meta = f" ({', '.join(meta_parts)})" if meta_parts else ""
        line = f"- {day}: {title}{meta}"

        # Optional: surface conflicts if present
        if avoid_hits:
            line += f" — note: contains {', '.join(avoid_hits)}"

        lines.append(line)

    return (header + "\n".join(lines)).strip()

def _fmt_qty(q: Any) -> str:
    if q is None or q == "":
        return ""
    try:
        f = float(q)
        return str(int(f)) if abs(f - int(f)) < 1e-9 else str(f)
    except Exception:
        return str(q).strip()


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
    day = str(details.get("day", "") or "").strip()
    title = str(details.get("title", "") or "").strip()
    time_min = details.get("time_min", "")
    cal = str(details.get("calorie_level", "") or "").upper()
    avoid_check = bool(details.get("avoid_check", False))

    servings = details.get("servings", None)
    ings = details.get("ingredients", []) or []
    steps = details.get("steps", []) or []

    cal_map = {"LOW": "Lighter", "MED": "Balanced", "HIGH": "More filling"}

    lines: List[str] = []

    # Lead-in you requested
    if day:
        lines.append(f"Sure — here is all the information about {day}:")
        lines.append("")

    # Header (NO recipe_id)
    header = f"{day} — {title}".strip(" —")
    lines.append(header)

    if str(time_min).strip() != "":
        lines.append(f"- Time: {time_min} min")

    if cal:
        lines.append(f"- Calories: {cal_map.get(cal, cal.title())}")

    if avoid_check:
        lines.append("- Note: this includes something you asked to avoid.")

    lines.append("")

    # Ingredients section
    if servings is not None and str(servings).strip() != "":
        lines.append(f"Ingredients (scaled for {servings} servings):")
    else:
        lines.append("Ingredients (scaled):")

    for ing in ings:
        name = str(ing.get("name", "") or "").strip()
        qty_txt = _fmt_qty(ing.get("qty", ""))
        unit = str(ing.get("unit", "") or "").strip()

        left = " ".join([t for t in [qty_txt, unit] if t]).strip()
        if left and name:
            lines.append(f"- {left} {name}".strip())
        elif name:
            lines.append(f"- {name}")
        else:
            # Defensive: skip malformed ingredient items
            continue

    # Steps section
    steps_clean = [str(s).strip() for s in steps if str(s).strip()]
    if steps_clean:
        lines.append("")
        lines.append("Steps:")
        for i, s in enumerate(steps_clean, 1):
            lines.append(f"{i}) {s}")

    return "\n".join(lines).strip()


def _render_confirm_plan(payload: Dict[str, Any], tracker_state: Dict[str, Any]) -> str:
    header = "Shopping list (Mon–Fri):\n\n" + _fmt_constraints(tracker_state) + "\n\n"
    body = _render_shopping_list(payload.get("shopping_list") or [])
    return (header + body).strip()



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

        # ------------------------- Swap-rejection acknowledgement (state-driven) -------------------------
        # Tracker sets last_denied_action when NLU emits out_of_domain with ood_type=REFUSE_PENDING
        if action == "fallback" and phase != "CONFIRMED":
            denied = (tracker_state or {}).get("last_denied_action")
            if isinstance(denied, dict) and str(denied.get("type", "")).strip().upper() == "SWAP_DAY":
                day = str(denied.get("day") or "").strip() or "that day"
                if day == "that day":
                    return "Okay — I won’t make the swap. What would you like to do next?"
                return f"Okay — I’ll keep {day} as-is. What would you like to do next?"

            last_mr = (tracker_state or {}).get("last_user_mr") or {}
            if isinstance(last_mr, dict) and str(last_mr.get("intent") or "") == "out_of_domain":
                ood_type = str((last_mr.get("slots") or {}).get("ood_type") or "").strip().upper()
                if ood_type == "REFUSE_PENDING":
                    day = str((tracker_state or {}).get("last_referenced_day") or "").strip()
                    day = day if day in {"Mon", "Tue", "Wed", "Thu", "Fri"} else "that day"
                    return (
                        f"No problem — we’ll keep {day} as-is. "
                        "If you’d like, I can still suggest another alternative or swap a different day."
                    )


        # ------------------------- Build verbatim factual blocks -------------------------
        menu_block = ""
        day_block = ""
        shopping_block = ""
        fact_block = ""

        # If executor produced a direct message (e.g., help), treat it as factual content.
        if payload.get("message") and action not in {
            "provide_info", "suggest_swap_day", "swap_day", "show_day", "propose_menus", "confirm_plan"
        }:
            fact_block = str(payload["message"]).strip()

        # Request questions are best kept deterministic (but wrapped by LLM).
        if action == "request_info":
            return _render_request_info(argument, tracker_state)
            
        # Deterministic high-risk actions 
        if action == "propose_menus":
            m1 = payload.get("menu1_pretty")
            m2 = payload.get("menu2_pretty")
            if isinstance(m1, dict) and isinstance(m2, dict) and m1 and m2:
                return _render_menus(m1, m2)
            # Defensive fallback if menus not provided
            return (
                "I couldn’t generate the two menu options. "
                "Could you adjust your preferences (prep time, calories, or avoids) and try again?"
            )

        if action == "show_day":
            details = payload.get("details")
            if isinstance(details, dict) and details:
                msg = _render_day_details(details)
                if _should_offer_next_steps(action, tracker_state):
                    msg = _append_next_steps_prompt(msg)
                return msg
            return "I couldn’t show that day. Which day should I focus on—Mon, Tue, Wed, Thu, or Fri?"
            

        if action == "show_week":
            if payload.get("week_overview") is not None:
                msg = _render_week_overview(payload, tracker_state)
                if _should_offer_next_steps(action, tracker_state):
                    msg = _append_next_steps_prompt(msg)
                return msg
            return "I couldn’t show the weekly plan right now. Try: “show the week plan again”."



        # Shopping list
        if action == "confirm_plan" and payload.get("shopping_list") is not None:
            return _render_confirm_plan(payload, tracker_state)


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
   
        if action in {"set_active_menu", "suggest_swap_day", "swap_day", "update_avoid"}:
            if fact_block:
                msg = fact_block.strip()
                if _should_offer_next_steps(action, tracker_state):
                    msg = _append_next_steps_prompt(msg)
                return msg

        if action == "fallback" and phase != "CONFIRMED":
            # swap rejection special-case already handled above

            awaiting = str((tracker_state or {}).get("awaiting_slot") or "").strip()
            if awaiting:
                return _render_request_info(awaiting, tracker_state)

            return _render_phase_aware_fallback(tracker_state)

        
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
            if action in {"suggest_swap_day", "swap_day"}:
                recent = "(none)"
            else:
                recent = self.history.last_iterations()if last_n_turns is None else self.history.last_iterations(last_n=last_n_turns)

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
        if not isinstance(nlg_text, str):
            raise TypeError(f"format_chat() must return str, got {type(nlg_text)}: {repr(nlg_text)[:200]}")

        self.logger.debug(f"NLG input:\n{nlg_text}")

        if nlg_text is None:
            nlg_text = ""
        elif not isinstance(nlg_text, str):
            nlg_text = str(nlg_text)

        enc = self.tokenizer(nlg_text, return_tensors="pt")
        inputs = enc.to(self.model.device)


        out = generate(self.model, inputs, self.tokenizer, self.args).strip()

        out = _strip_accidental_action_echo(out)
        if not out:
            return "Sorry—something went wrong while generating the response. Could you repeat that?"

        if _should_offer_next_steps(action, tracker_state):
            out = _append_next_steps_prompt(out)

        return out
    
    def render_steps(
        self,
        executed_steps: List[Any],
        tracker_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Deterministically render a single assistant message from multiple executed steps.
        NO LLM calls here.
        """
        tracker_state = tracker_state or {}

        chunks: List[str] = []
        last_rendered_action = ""
        
        # Defensive: ensure confirm_plan is rendered last if present
        def is_confirm(step: Any) -> bool:
            return getattr(step, "final_action", "") == "confirm_plan"

        non_confirm = [s for s in executed_steps if not is_confirm(s)]
        confirm = [s for s in executed_steps if is_confirm(s)]
        ordered = non_confirm + confirm

        for s in ordered:
            action = getattr(s, "final_action", "")
            argument = getattr(s, "final_argument", "")
            payload = getattr(s, "payload", {}) or {}

            # If any step produced an execution error, surface it and stop.
            if payload.get("error"):
                return str(payload["error"]).strip()

            # Hard stop: request_info should be the final output immediately
            if action == "request_info":
                return _render_request_info(argument, tracker_state)
            
            if action == "provide_info":
                chunks.append(_render_provide_info(payload, tracker_state))
                last_rendered_action = action
                continue

            if action == "propose_menus" and payload.get("menu1_pretty") and payload.get("menu2_pretty"):
                chunks.append(_render_menus(payload["menu1_pretty"], payload["menu2_pretty"]))
                # After menus, user must select next -> stop here
                break

            if action == "set_active_menu":
                if payload.get("ok"):
                    mid = (tracker_state.get("active_menu_id") or argument)
                    chunks.append(
                        f"Great — we’ll go with option {mid}.\n"
                        "You can ask what’s planned on a day, swap a day, update foods to avoid, or confirm for the shopping list."
                    )
                else:
                    chunks.append("I couldn’t select that option. Please reply with 1 or 2.")
                continue

            if action == "update_avoid":
                repaired = payload.get("repaired_days") or []
                base = "Got it — I updated your foods to avoid.\n\n" + _fmt_constraints(tracker_state)
                if repaired:
                    base += "\n\nI also updated these days to keep everything compatible: " + ", ".join(repaired)
                chunks.append(base)
                last_rendered_action = action
                continue

            if action == "suggest_swap_day":
                if payload.get("suggested"):
                    chunks.append(
                        f"Sure! Here's an alternative meal for {argument}:\n\n"
                        f"- {payload.get('suggested_title', 'a new recipe')}\n\n"
                        f"Do you want me to swap {argument} to this?"
                    )
                else:
                    chunks.append("I couldn’t find a good alternative for that day with your current preferences.")
                continue

            if action == "swap_day":
                if payload.get("swapped"):
                    chunks.append(f"Done — I swapped {argument} to: {payload.get('new_title', 'a new recipe')}.")
                else:
                    chunks.append("I couldn’t find a good alternative for that day with your current preferences.")
                last_rendered_action = action
                continue

            if action == "show_day" and payload.get("details"):
                chunks.append(_render_day_details(payload["details"]))
                last_rendered_action = action
                continue

            if action == "show_week" and payload.get("week_overview") is not None:
                chunks.append(_render_week_overview(payload, tracker_state))
                last_rendered_action = action
                continue

            if action == "confirm_plan" and payload.get("shopping_list") is not None:
                chunks.append(_render_confirm_plan(payload, tracker_state))
                continue

            if action == "fallback":
                phase = str((tracker_state or {}).get("phase", "") or "")
                if phase == "CONFIRMED":
                    chunks.append("All set — your meal plan is finalized. Type 'exit' to end, or start a new plan anytime.")
                else:
                    denied = (tracker_state or {}).get("last_denied_action")
                    if isinstance(denied, dict) and str(denied.get("type", "")).strip().upper() == "SWAP_DAY":
                        day = str(denied.get("day") or "").strip() or "that day"
                        if day == "that day":
                            chunks.append("Okay — I won’t make the swap. What would you like to do next?")
                        else:
                            chunks.append(f"Okay — I’ll keep {day} as-is. What would you like to do next?")
                        continue
                
                    awaiting = str((tracker_state or {}).get("awaiting_slot") or "").strip()
                    if awaiting:
                        chunks.append(_render_request_info(awaiting, tracker_state))
                    else:
                        chunks.append(_render_phase_aware_fallback(tracker_state))
                continue
             

        final = "\n\n".join([c for c in chunks if c]).strip()

        if final and _should_offer_next_steps(last_rendered_action, tracker_state):
            final = _append_next_steps_prompt(final)

        return final

