# nlu.py
"""
Meal Kit Composer — NLU (LLM-based).

Responsibilities:
- Given user text, produce one or more MR in the flat JSON format:
    {"intent": "...", "slots": {...}}
- Do not invent values; if missing, output null
- Normalize using intents_schema.normalize_mr
- Return the normalized MR even if invalid (for robustness)

Compatible with:
- utils.generate + args.chat_template
- intents_schema.INTENT_SLOTS, intents_schema.validate_mr
- support_fn.parsing_json
- support_classes constants for controlled vocab (via schema_hint)
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Union, List, Optional
from utils import generate, format_chat
from intents_schema import INTENT_SLOTS, normalize_mr
from support_fn import parsing_json
from support_classes import (
    ALLOWED_DAYS,
    ALLOWED_TIME_LIMITS,
    ALLOWED_CALORIE_LEVELS,
    ALLOWED_AVOID_ITEMS,
)


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=str)


# --- Evidence-gate helpers (Option B) ----------------------------------------

_WORD_NUMS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}

_YES_TOKENS = ("yes", "yep", "yeah", "ok", "okay", "sure", "do it", "go ahead", "swap it")
_NO_TOKENS  = ("no", "nope", "nah", "no thanks", "don't", "do not", "keep it", "leave it", "cancel", "never mind")

def _last_assistant_line(recent: str) -> str:
    """Extract the last assistant message from RECENT_TURNS (best-effort)."""
    if not recent:
        return ""
    lines = [ln.strip() for ln in recent.splitlines() if ln.strip()]
    # try common prefixes
    for i in range(len(lines) - 1, -1, -1):
        ln = lines[i]
        if ln.lower().startswith("assistant:"):
            return ln[len("assistant:"):].strip()
    # fallback: just return last line
    return lines[-1] if lines else ""

def _infer_awaited_slot(recent: str) -> str:
    """
    Heuristic: infer which controlled slot is being asked based on the last assistant utterance.
    Returns: servings|time_limit|calorie_level|avoid_items|menu_id|yes_no_swap|""
    """
    s = (recent or "").lower()

    if "how many servings" in s:
        return "servings"
    if "quick meals" in s or "prep time" in s:
        return "time_limit"
    if "lighter meals" in s or "balanced" in s or "more filling" in s:
        return "calorie_level"
    if "allerg" in s or "foods you want to avoid" in s:
        return "avoid_items"
    if ("which option do you prefer" in s) or (("option 1" in s) and ("option 2" in s)):
        return "menu_id"
    if "do you want me to swap" in s:
        return "yes_no_swap"

    return ""

def _text_has_servings_evidence(user_text: str) -> bool:
    t = (user_text or "").lower()
    if re.search(r"\b[1-6]\b", t):
        return True
    return any(re.search(rf"\b{w}\b", t) for w in _WORD_NUMS.keys())

def _text_has_time_limit_evidence(user_text: str, val: str) -> bool:
    t = (user_text or "").lower()
    if val == "FAST":
        return any(k in t for k in ("fast", "quick", "short", "asap"))
    if val == "NORMAL":
        return any(k in t for k in ("normal", "regular", "standard", "ok", "okay"))
    return False

def _text_has_calorie_level_evidence(user_text: str, val: str) -> bool:
    t = (user_text or "").lower()
    if val == "LOW":
        return any(k in t for k in ("low", "light", "lighter"))
    if val == "MED":
        return any(k in t for k in ("med", "medium", "balanced", "average"))
    if val == "HIGH":
        return any(k in t for k in ("high", "filling", "hearty", "more"))
    return False

def _text_has_menu_id_evidence(user_text: str, mid: int) -> bool:
    t = (user_text or "").lower()
    if mid == 1:
        return bool(re.search(r"\b1\b", t) or re.search(r"\bone\b", t) or "option 1" in t or "menu 1" in t)
    if mid == 2:
        return bool(re.search(r"\b2\b", t) or re.search(r"\btwo\b", t) or "option 2" in t or "menu 2" in t)
    return False

def _text_has_yes_no_evidence(user_text: str) -> Optional[bool]:
    """Return True for yes, False for no, None for neither."""
    t = (user_text or "").strip().lower()
    if not t:
        return None
    # allow exact/starts-with matches for short replies
    if any(t == y or t.startswith(y + " ") for y in _YES_TOKENS):
        return True
    if any(t == n or t.startswith(n + " ") for n in _NO_TOKENS):
        return False
    return None

def _text_has_avoid_items_evidence(user_text: str, items: List[str]) -> bool:
    """
    Minimal evidence check:
    - require that each controlled token (or a close synonym) appears somewhere in the user text.
    This avoids accepting hallucinated ["nuts"] for 'kuku'.
    """
    t = (user_text or "").lower()
    syn = {
        "dairy": ("dairy", "milk", "cheese", "butter", "yogurt"),
        "egg": ("egg", "eggs", "omelette"),
        "fish": ("fish", "salmon", "tuna", "cod", "seafood"),
        "gluten": ("gluten", "wheat", "flour", "bread", "pasta"),
        "meat": ("meat", "beef", "pork", "chicken", "steak"),
        "nuts": ("nuts", "peanut", "almond", "walnut", "hazelnut"),
        "sesame": ("sesame",),
        "shellfish": ("shellfish", "shrimp", "crab", "lobster"),
        "soy": ("soy", "tofu", "soy sauce"),
    }
    for it in items:
        keys = syn.get(it, (it,))
        if not any(k in t for k in keys):
            return False
    return True


class NLU:
    """
    Minimal LLM-based NLU that outputs one or more MR per user turn.

    Design choices (aligned with your blueprint + current architecture):
    - controlled vocab guidance via schema_hint
    - normalization via intents_schema.normalize_mr
    """

    def __init__(self, history, model, tokenizer, args, logger):
        self.history = history
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.logger = logger

    def __call__(self, user_text: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
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
                "refine_mode": ["SUGGEST", "COMMIT"],
                "menu_id": [1, 2],
                "help_slot": [
                    "servings", "time_limit", "calorie_level", "avoid_items",
                    "menu_id", "target_day", "refine_type", "value", 
                    "all",
                ],
                "help_intent": ["plan", "select_menu", "inspect", "refine", "confirm", "show_week"],
            },
    
            "output_format": (
                    "Either a single MR object: {\"intent\":\"...\",\"slots\":{...}} "
                    "OR a JSON array of MR objects: [{\"intent\":\"...\",\"slots\":{...}}, ...]"
                ),

            "rules": [
                "Output MUST be valid JSON.",
                "Output MUST be either a single MR object OR a JSON array of MR objects.",
                "If the user expresses multiple distinct requests that map to different intents, output multiple MRs in user order.",
                "Only fill slots that belong to each MR's intent.",
                "Only include slots that are relevant to what the user provided in this turn (no extra null slots).",
                "Do not invent new information, but DO normalize synonyms/typos into the controlled values.",
                "Use RECENT_TURNS to resolve short answers (e.g., '1', 'fast', 'yes').",
                "When extracting avoid_items, output a JSON list when confident; otherwise output null. Never output a single string.",
                "Return ONLY JSON.",
            ],
        }

        system_prompt = (
            "You are the NLU component for a Meal Kit Composer assistant.\n"
            "Task: extract ONE OR MORE intents and their slot-value pairs from the user's text.\n"
            "\n"
            "Core rules:\n"
            "- Output MUST be valid JSON.\n"
            "- Output MUST be either:\n"
            "  (A) a single MR object: {\"intent\":\"...\",\"slots\":{...}}\n"
            "  (B) a JSON array of MR objects: [{\"intent\":\"...\",\"slots\":{...}}, ...]\n"
            "- If the user expresses multiple distinct requests that map to different intents, output multiple MRs in user order.\n"
            "- Only fill slots that belong to each MR's intent.\n"
            "- Only include slots that are relevant to what the user provided in this turn.\n"
            "  Exception: when outputting a full plan MR in one turn, include all plan slots.\n"
            "- Do not invent values or add defaults that the user did not state.\n"
            "- For intent=\"out_of_domain\", you MAY include an optional slot \"ood_type\" to specify the reason (e.g., \"REFUSE_PENDING\").\n"
            "- Return ONLY JSON. No extra text.\n"
            "\n"
            "Dialogue context rules (use RECENT_TURNS):\n"
            "- If the last assistant message asked for a specific detail (e.g., servings/time/calories/menu choice/day),\n"
            "  and the user replies with only a value (e.g., \"1\", \"two\", \"fast\", \"Tuesday\", \"menu 1\", \"yes\"),\n"
            "  interpret it as answering that question and fill the corresponding slot.\n"
            "- If the user asks to see the whole weekly plan/menu again (e.g., \"show the week plan\", \"show the weekly plan\", \n"
            "  \"show the week again\", \"show me the plan again\", \"weekly menu\", \"week overview\"), interpret as:\n"
            "  {\"intent\":\"show_week\",\"slots\":{}}\n"
            "- If RECENT_TURNS shows the assistant is collecting PLAN details (servings, time_limit, calorie_level, avoid_items),\n"
            "  then interpret the user’s reply as intent=plan that fills ONLY the asked slot(s), even if the text contains verbs like \"avoid\".\n"
            "- When replying to a plan question (RECENT_TURNS indicates which slot was asked):\n"
            "  - Output ONLY that plan slot in slots (plus avoid_items as [] if explicitly none).\n"
            "  - Do NOT include other plan plan slots with null in that MR.\n"
            "- If the user expresses dislike/preference about a weekday meal (e.g., \"don’t like Friday\", \"change Friday\", \"replace Friday meal\"), \n"
            "  interpret as intent='refine' with refine_type='SWAP_DAY' and target_day set to that weekday; set value='BEST_FIT'. \n"
            "  Set mode='SUGGEST' unless they explicitly say change/swap/replace now (then mode='COMMIT'). \n"
            "  Do NOT treat weekdays as avoid_items. \n"
            "  Do NOT output multiple inspect intents for Mon–Fri in this case.\n"
            "- If the last assistant message asked to confirm a suggested swap (e.g., contains \"Do you want me to swap\"),\n"
            "  then interpret short replies as follows:\n"
            " - If the user replies with acceptance (e.g., \"yes\", \"ok\", \"do it\", \"swap it\", \"go ahead\") -> {\"intent\":\"confirm\",\"slots\":{}}\n"
            " - If the user replies with refusal (e.g., \"no\", \"no thanks\", \"nope\", \"nah\", \"don't\", \"don't swap\", \"keep it\", \"leave it\", \"never mind\", \"cancel\") -> {\"intent\":\"out_of_domain\",\"slots\":{\"ood_type\":\"REFUSE_PENDING\"}}\n"
            "- Interpret intent=\"confirm\" ONLY if the user explicitly requests confirmation/finalization (e.g., \"confirm\", \"finalize\", \"generate shopping list\")\n"
            "  OR if the last assistant message asked an explicit yes/no confirmation question and the user replies \"yes\".\n"
            "  Do NOT treat generic acknowledgements (\"ok\", \"fine\", \"thanks\") as confirm.\n"
            "- If the user reply does NOT contain evidence for a controlled value (servings/time_limit/calorie_level/avoid_items/menu_id/yes-no), output intent=\"out_of_domain\" with slots={\"ood_type\":\"INVALID_ANSWER\"} (do NOT guess).\n"
            "\n"
            "Canonicalization (normalize user language into controlled values):\n"
            "- servings: output an integer 1..6 (map words one/two/three/four/five/six to 1..6).\n"
            "- time_limit: output FAST or NORMAL (map quick/short/asap to FAST; regular/standard to NORMAL).\n"
            "- calorie_level: output LOW, MED, or HIGH (map medium/balanced/average to MED).\n"
            "- target_day: output one of Mon,Tue,Wed,Thu,Fri (map full names like Monday->Mon).\n"
            "- menu_id: output 1 or 2.\n"
            "- show_week: output {\"intent\":\"show_week\",\"slots\":{}} (slots must be an empty object).\n"
            "- If there is an obvious typo and confidence is high (e.g., \"fats\"->FAST), correct it.\n"
            "- avoid_items: output a list of strings from the controlled vocabulary.\n"
            "  If the user explicitly indicates no restrictions (e.g., 'none', 'no allergies', 'nothing to avoid', 'I eat everything'),\n"
            "  output an EMPTY LIST: [] (do NOT output null).\n"
            "  If the user lists multiple avoid items in one turn (e.g., \"meat and nuts\", \"meat, nuts, and dairy\"), output ALL items as a list in the same order.\n"
            "avoid_items strict rules:\n"
            "- avoid_items MUST ONLY contain tokens from this exact set:\n"
            "  [\"dairy\",\"egg\",\"fish\",\"gluten\",\"meat\",\"nuts\",\"sesame\",\"shellfish\",\"soy\"].\n"
            "- If the user says a plural, output the singular form used in the set (e.g., \"eggs\" -> \"egg\").\n"
            "- If the user uses a synonym, map it to the closest controlled token:\n"
            " - \"seafood\" -> \"fish\" (or \"shellfish\" if explicitly shrimp/crab/lobster; if unclear prefer \"fish\")\n"
            " - \"milk/cheese/butter\" -> \"dairy\"\n"
            " - \"bread/pasta/flour/wheat\" -> \"gluten\"\n"
            "- If the user gives an avoid item that does not map confidently to the controlled set, DO NOT guess.\n"
            "  Output null for avoid_items (if the slot is required by context) and let the DM ask again.\n"           
            "  If avoid_items is present, it MUST be either a JSON list or null.\n"
            "  Prefer a JSON list whenever mapping is confident.\n"
            "  Avoid mapping table (examples):\n"
            "  - eggs, omelette -> egg\n"
            "  - peanuts, almonds, hazelnuts, walnuts -> nuts\n"
            "  - steak, beef, pork, chicken -> meat\n"
            "  - shrimp, crab, lobster -> shellfish\n"
            "  - salmon, tuna, cod -> fish\n"
            "  - soy sauce, tofu -> soy\n"
            "  avoid_items output rules:\n"
            "  - If the user provides at least one avoid item: output avoid_items as a non-empty list.\n"
            "  - If the user explicitly says no restrictions: output [].\n"
            "  - If the user response is ambiguous or contains items that do not map confidently to the controlled set: output null.\n"
            "\n"
            "Refine mode rule (for refine_type=SWAP_DAY):\n"
            "- If the user asks to suggest/propose an alternative, set slots.mode to \"SUGGEST\".\n"
            "- If the user explicitly asks to swap/change/replace, set slots.mode to \"COMMIT\".\n"
            "\n"
            "Examples (illustrative, not exhaustive):\n"
            "- User: \"Plan my meals for two people, fast, medium calories, no allergies.\" -> "
            "{\"intent\":\"plan\",\"slots\":{\"servings\":2,\"time_limit\":\"FAST\",\"calorie_level\":\"MED\",\"avoid_items\":[]}}\n"
            "- Assistant previously asked about avoid items. User: \"none\" -> "
            "{\"intent\":\"plan\",\"slots\":{\"avoid_items\":[]}}\n"
            "- User: \"Show me Tuesday.\" -> "
            "{\"intent\":\"inspect\",\"slots\":{\"target_day\":\"Tue\"}}\n"
            "- Assistant asked about avoid items. User: \"avoid meat and nuts\" -> "
            "{\"intent\":\"plan\",\"slots\":{\"avoid_items\":[\"meat\",\"nuts\"]}}\n"
            "- User: \"Also avoid nuts and show me Tuesday.\" -> "
            "["
            "{\"intent\":\"refine\",\"slots\":{\"refine_type\":\"ADD_AVOID_ITEM\",\"target_day\":null,\"value\":\"nuts\",\"mode\":null}},"
            "{\"intent\":\"inspect\",\"slots\":{\"target_day\":\"Tue\"}}"
            "]\n"
            "- User: \"Can you show me the week plan again?\" -> {\"intent\":\"show_week\",\"slots\":{}}\n"
            "- User: \"Show the weekly menu\" -> {\"intent\":\"show_week\",\"slots\":{}}\n"
            "- User: \"Show the week plan and swap Tue\" -> [\n"
            "  {\"intent\":\"show_week\",\"slots\":{}},\n"
            "  {\"intent\":\"refine\",\"slots\":{\"refine_type\":\"SWAP_DAY\",\"target_day\":\"Tue\",\"value\":\"BEST_FIT\",\"mode\":\"COMMIT\"}}\n"
            "]\n"
            "- Assistant asked about avoid items. User: \"avoid eggs\" -> "
            "{\"intent\":\"plan\",\"slots\":{\"avoid_items\":[\"egg\"]}}\n"
            "\n"
            "- Assistant asked about avoid items. User: \"avoid seafood\" -> "
            "{\"intent\":\"plan\",\"slots\":{\"avoid_items\":[\"fish\"]}}\n"
            "\n"
            "- Assistant asked about avoid items. User: \"avoid mango\" -> "
            "{\"intent\":\"plan\",\"slots\":{\"avoid_items\":null}}\n"
            "\n"
            "- User: \"I don't like Friday\" -> "
            "{\"intent\":\"refine\",\"slots\":{\"refine_type\":\"SWAP_DAY\",\"target_day\":\"Fri\",\"value\":\"BEST_FIT\",\"mode\":\"SUGGEST\"}}\n"
            "- User: \"Change Friday\" -> "
            "{\"intent\":\"refine\",\"slots\":{\"refine_type\":\"SWAP_DAY\",\"target_day\":\"Fri\",\"value\":\"BEST_FIT\",\"mode\":\"COMMIT\"}}\n"
            "- Assistant: \"Do you want me to swap Mon to this?\" User: \"no thanks\" -> "
            "{\"intent\":\"out_of_domain\",\"slots\":{\"ood_type\":\"REFUSE_PENDING\"}}\n"
            "- User: \"Option 1 and confirm.\" -> "
            "["
            "{\"intent\":\"select_menu\",\"slots\":{\"menu_id\":1}},"
            "{\"intent\":\"confirm\",\"slots\":{}}"
            "]\n"
            "- User: \"Suggest an alternative for Tuesday and show Wednesday.\" -> "
            "["
            "{\"intent\":\"refine\",\"slots\":{\"refine_type\":\"SWAP_DAY\",\"target_day\":\"Tue\",\"value\":\"BEST_FIT\",\"mode\":\"SUGGEST\"}},"
            "{\"intent\":\"inspect\",\"slots\":{\"target_day\":\"Wed\"}}"
            "]\n"
            "\n"
            "Special help intent rule:\n"
            "- If the user asks what values/options are allowed/supported for something, use intent='help'\n"
            "  and set slots.slot to the relevant item (e.g., 'servings', 'time_limit', 'calorie_level', 'avoid_items', or 'all').\n"
            "- For help.intent, set it to the most relevant context among: plan/select_menu/inspect/refine/confirm;\n"
            "  if unclear, set it to null.\n"
        )


        DEBUG = bool(getattr(self.args, "debug", False))

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

        # ---- Hard normalize to a tokenizer-safe text input ----
        if nlu_text is None:
            nlu_text = ""

        # If something upstream ever returns chat "messages" (list[dict]), render it now.
        if isinstance(nlu_text, list) and nlu_text and isinstance(nlu_text[0], dict) and "role" in nlu_text[0]:
            nlu_text = self.tokenizer.apply_chat_template(
                nlu_text, tokenize=False, add_generation_prompt=True
            )

        # Any remaining non-str becomes str (last-resort safety)
        if not isinstance(nlu_text, str):
            nlu_text = str(nlu_text)

        self.logger.debug(f"NLU input:\n{nlu_text}")
        self.logger.debug(f"NLU prompt type={type(nlu_text)}")

        if DEBUG:
            print("DEBUG nlu_text type:", type(nlu_text))
            print("DEBUG nlu_text head:", repr(nlu_text)[:200])

        # IMPORTANT: tokenize as a *batch of 1* (your comment said you wanted this, but you weren't doing it)
        enc = self.tokenizer([nlu_text], return_tensors="pt")
        inputs = enc.to(self.model.device)


        if DEBUG:
            print("DEBUG tokenizer input type:", type(nlu_text), "len:", len(nlu_text))
            print("DEBUG tokenizer input first 80:", repr(nlu_text[:80]))


        out = generate(self.model, inputs, self.tokenizer, self.args).strip()

        self.logger.debug(f"NLU raw output: {out}")

        obj = parsing_json(out)

        # --- Evidence gate (prevents "uga" -> FAST, etc.) ---------------------
        awaited = _infer_awaited_slot(recent)

        def _invalid_answer():
            return {"intent": "out_of_domain", "slots": {"ood_type": "INVALID_ANSWER"}}

        # If we are in a controlled-slot Q/A moment, require evidence in USER_INPUT
        if awaited == "servings":
            # accept only if user text looks like a servings answer
            if not _text_has_servings_evidence(user_text):
                return _invalid_answer()

        elif awaited == "time_limit":
            # If LLM filled time_limit, require evidence for that specific value
            if isinstance(obj, dict):
                slots = obj.get("slots") or {}
                if obj.get("intent") == "plan" and "time_limit" in slots:
                    val = slots.get("time_limit")
                    if isinstance(val, str) and val in ALLOWED_TIME_LIMITS:
                        if not _text_has_time_limit_evidence(user_text, val):
                            return _invalid_answer()

        elif awaited == "calorie_level":
            if isinstance(obj, dict):
                slots = obj.get("slots") or {}
                if obj.get("intent") == "plan" and "calorie_level" in slots:
                    val = slots.get("calorie_level")
                    if isinstance(val, str) and val in ALLOWED_CALORIE_LEVELS:
                        if not _text_has_calorie_level_evidence(user_text, val):
                            return _invalid_answer()

        elif awaited == "avoid_items":
            if isinstance(obj, dict):
                slots = obj.get("slots") or {}
                if obj.get("intent") == "plan" and "avoid_items" in slots:
                    val = slots.get("avoid_items")
                    # if non-empty list, require evidence for each item
                    if isinstance(val, list) and val:
                        if not _text_has_avoid_items_evidence(user_text, [str(x) for x in val]):
                            return _invalid_answer()
                    # if [] (explicit none) you can accept; if null/None accept (DM will re-ask)

        elif awaited == "menu_id":
            if isinstance(obj, dict):
                slots = obj.get("slots") or {}
                if obj.get("intent") == "select_menu" and "menu_id" in slots:
                    mid = slots.get("menu_id")
                    if isinstance(mid, int) and mid in (1, 2):
                        if not _text_has_menu_id_evidence(user_text, mid):
                            return _invalid_answer()

        elif awaited == "yes_no_swap":
            yn = _text_has_yes_no_evidence(user_text)
            if yn is True:
                # force confirm regardless of LLM output
                obj = {"intent": "confirm", "slots": {}}
            elif yn is False:
                # force REFUSE_PENDING regardless of LLM output
                obj = {"intent": "out_of_domain", "slots": {"ood_type": "REFUSE_PENDING"}}
            else:
                return _invalid_answer()


        if isinstance(obj, dict):
            nm = normalize_mr(obj)
            self.logger.debug(f"NLU normalized MR: {nm}")
            return nm

        if isinstance(obj, list):
            mrs = [normalize_mr(x) for x in obj if isinstance(x, dict)]
            if mrs:
                self.logger.debug(f"NLU normalized MRs: {mrs}")
                return mrs

        return {"intent": "out_of_domain", "slots": {}}
        

