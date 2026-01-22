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

_ACK_TOKENS = (
    "great", "thanks", "thank you", "ok", "okay", "cool", "nice", "sounds good", "perfect", "awesome"
)

_CLOSE_TOKENS = (
    "all set", "i am all set", "i'm all set", "that's all", "that is all",
    "nothing else", "no more", "we're done", "we are done", "done for now",
    "i am satisfied", "i'm satisfied", "i am finished", "i'm finished"
)

def _is_ack(user_text: str) -> bool:
    t = (user_text or "").strip().lower()
    return bool(t) and any(t == a or a in t for a in _ACK_TOKENS)

def _is_close(user_text: str) -> bool:
    t = (user_text or "").strip().lower()
    return bool(t) and any(c in t for c in _CLOSE_TOKENS)






def _text_has_servings_evidence(user_text: str) -> bool:
    t = (user_text or "").lower()
    if re.search(r"\b[1-6]\b", t):
        return True
    return any(re.search(rf"\b{w}\b", t) for w in _WORD_NUMS.keys())

def _text_has_time_limit_evidence(user_text: str, val: str) -> bool:
    t = (user_text or "").lower()
    if val == "FAST":
        return any(k in t for k in ("fast", "quick", "short", "asap", "in a hurry", "busy", "no time"))
    if val == "NORMAL":
        return any(k in t for k in ("normal", "regular", "standard", "ok", "okay", "plenty of time", "a lot of time", "lots of time", "free time", "no rush"))
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

    def __call__(self, user_text: str, awaiting_slot: Optional[str] = None,) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
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
                "Return ONLY valid JSON: either a single MR object "
                "{\"intent\":\"...\",\"slots\":{...}} or an array of MR objects "
                "[{\"intent\":\"...\",\"slots\":{...}}, ...]."
            ),

            "rules": [
                "Return ONLY valid JSON (no extra text).",
                "Output must be either ONE MR object or an ARRAY of MR objects.",
                "If the user makes multiple distinct requests mapping to different intents, output multiple MRs in user order.",
                "For each MR: include only slots allowed for that intent, and only slots supported by the current user turn (no extra nulls).",
                "Do not invent information; normalize clear synonyms/typos into controlled values.",
                "Use RECENT_TURNS for ellipsis ONLY when AWAITING_SLOT is not (none) or when resolving day/menu references (e.g., “that one”, “option 2”).",
                "If AWAITING_SLOT is (none), do not treat the user message as an answer to a prior slot question.",
                "avoid_items must be a JSON list (possibly empty) when confident, otherwise null; never output a single string.",
            ],

        }

        system_prompt = (
            "You are the NLU component for a Meal Kit Composer assistant.\n"
            "Task: extract ONE OR MORE Meaning Representations (MRs) from USER_INPUT.\n"
            "Each MR is: {\"intent\":\"...\",\"slots\":{...}}.\n\n"

            "Output constraints:\n"
            "- Return ONLY valid JSON (no extra text).\n"
            "- Output must be either ONE MR object or an ARRAY of MR objects.\n"
            "- If multiple distinct requests map to different intents, output multiple MRs in user order.\n"
            "- For each MR: include only slots allowed for that intent, and only slots supported by the current user turn (no extra nulls).\n"
            "- Do not invent values or defaults; normalize clear synonyms/typos into controlled values.\n"
            "- For intent=\"out_of_domain\", you may include slots.ood_type (e.g., REFUSE_PENDING, INVALID_ANSWER, ACK).\n\n"

            "RECENT_TURNS handling (context resolution):\n"
            "- Use RECENT_TURNS to interpret short/elliptical replies.\n"
            "- If multiple assistant questions appear in RECENT_TURNS, treat the MOST RECENT assistant question as the one being answered.\n"
            "If AWAITING_SLOT is one of servings/time_limit/calorie_level/avoid_items, interpret the user reply as intent=plan and fill ONLY that slot.\n"
            "- When replying to a plan question, output ONLY that plan slot in slots (plus avoid_items=[] if explicitly none).\n\n"
            "- If the user provides multiple plan constraints in one message, output intent=plan and include all provided plan slots.\n"

            "Intent rules:\n"
            "- show_week: if the user asks to see the weekly plan/menu again (e.g., show the week/weekly plan/menu/overview/again), output:\n"
            "  {\"intent\":\"show_week\",\"slots\":{}}.\n"
            "- refine (weekday preference/dislike): if the user asks to change/replace a weekday meal, output intent=refine with:\n"
            "  refine_type=SWAP_DAY, target_day=<Mon..Fri>, value=BEST_FIT, mode=SUGGEST unless they explicitly request swapping now (then COMMIT).\n"
            "  Do not treat weekdays as avoid_items; do not emit multiple inspect intents for Mon–Fri.\n"
            "- refine (avoid list updates): if the user asks to avoid a food from now on (allergy/intolerance/preference), output intent=refine with:\n"
            "  refine_type=ADD_AVOID_ITEM and value=<one allowed token>.\n"
            "  Examples of ADD_AVOID_ITEM cues: \"avoid X\", \"no X\", \"can't eat X\", \"allergic to X\", \"remove X from the meals\", \"take out X\".\n"
            "- refine (remove from avoid list): ONLY use refine_type=REMOVE_AVOID_ITEM if the user explicitly wants to STOP avoiding something.\n"
            "  Examples: \"I can eat X again\", \"don't avoid X\", \"remove X from my avoid list\", \"X is OK\".\n"
            "- If the user says \"remove X\" or \"take out X\" without explicitly mentioning the avoid list, interpret it as ADD_AVOID_ITEM (remove X from meals).\n"
            "- swap confirmation: if the last assistant message asked \"Do you want me to swap ...?\",\n"
            "  then interpret short replies as (this applies only in that swap-confirmation context):\n"
            "  * accept -> {\"intent\":\"confirm\",\"slots\":{}}\n"
            "  * refuse -> {\"intent\":\"out_of_domain\",\"slots\":{\"ood_type\":\"REFUSE_PENDING\"}}\n"
            "- confirm: output intent=confirm if the user explicitly requests finalization (confirm/finalize/generate shopping list),\n"
            "  or if they clearly close the session (I’m all set/that’s all/nothing else/we’re done/etc.),\n"
            "  BUT NOT if RECENT_TURNS shows the assistant is currently asking for a required slot value.\n"
            "- acknowledgements: generic acknowledgements (ok/fine/thanks/great/etc.) are NOT confirm; output out_of_domain with ood_type=ACK.\n"
            "- invalid answers: ONLY if AWAITING_SLOT is set to a controlled slot AND the user reply lacks evidence for that slot, output INVALID_ANSWER (do not guess). If AWAITING_SLOT is (none), do NOT output INVALID_ANSWER.\n"

            "Canonicalization (controlled values):\n"
            "- servings: integer 1..6 (map one/two/three/four/five/six).\n"
            "- time_limit: FAST or NORMAL (quick/fast/short/asap -> FAST; normal/regular/standard/no rush -> NORMAL).\n"
            "- calorie_level: LOW, MED, HIGH (medium/balanced/average -> MED).\n"
            "- target_day: Mon/Tue/Wed/Thu/Fri (map full day names).\n"
            "- menu_id: 1 or 2.\n"
            "- If an obvious typo is high-confidence (e.g., fats->FAST), correct it.\n\n"

            "avoid_items rules:\n"
            "- avoid_items must be either a JSON list, an empty list [], or null (never a single string).\n"
            "- Allowed tokens only: [\"dairy\",\"egg\",\"fish\",\"gluten\",\"meat\",\"nuts\",\"sesame\",\"shellfish\",\"soy\"].\n"
            "- Plurals -> singular token (eggs->egg).\n"
            "- Synonyms (confident mappings):\n"
            "  * seafood -> fish (unless explicitly shrimp/crab/lobster -> shellfish)\n"
            "  * milk/cheese/butter -> dairy\n"
            "  * bread/pasta/flour/wheat -> gluten\n"
            "- If the user explicitly indicates no restrictions (none/no allergies/nothing to avoid/I eat everything), output [].\n"
            "- If the user mentions items that do not map confidently to the set, output null.\n\n"

            "Help intent:\n"
            "- If the user asks what values/options are allowed/supported, use intent=help and set slots.help_slot accordingly\n"
            "  (servings/time_limit/calorie_level/avoid_items/menu_id/target_day/refine_type/value/all). Set help_intent if clear, else null.\n"

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
            "{\"intent\":\"refine\",\"slots\":{\"refine_type\":\"ADD_AVOID_ITEM\",\"value\":\"nuts\"}},"
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
            "- User (after menu is chosen): \"My sister doesn't eat eggs\" -> "
            "{\"intent\":\"refine\",\"slots\":{\"refine_type\":\"ADD_AVOID_ITEM\",\"value\":\"egg\"}}\n"
            "- User: \"Remove eggs from the meals\" -> "
            "{\"intent\":\"refine\",\"slots\":{\"refine_type\":\"ADD_AVOID_ITEM\",\"value\":\"egg\"}}\n"
            "- User: \"Eggs are fine now, stop avoiding them\" -> "
            "{\"intent\":\"refine\",\"slots\":{\"refine_type\":\"REMOVE_AVOID_ITEM\",\"value\":\"egg\"}}\n"
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
            "- User: \"I'm all set, thanks.\" -> {\"intent\":\"confirm\",\"slots\":{}}\n"
            "- User: \"Great, thanks!\" -> {\"intent\":\"out_of_domain\",\"slots\":{\"ood_type\":\"ACK\"}}\n"   
            "\n"
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
            + "\n\nAWAITING_SLOT:\n"
            + ((awaiting_slot or "").strip() if awaiting_slot else "(none)")
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

        # Tokenize a single string (still yields batch dim = 1 with return_tensors="pt").
        # This avoids intermittent fast-tokenizer encode_batch() type errors.
        try:
            enc = self.tokenizer(nlu_text, return_tensors="pt")
        except TypeError:
            # Last-resort hardening: coerce to a plain Python str and retry.
            enc = self.tokenizer(str(nlu_text), return_tensors="pt")
        inputs = enc.to(self.model.device)


        if DEBUG:
            print("DEBUG tokenizer input type:", type(nlu_text), "len:", len(nlu_text))
            print("DEBUG tokenizer input first 80:", repr(nlu_text[:80]))


        out = generate(self.model, inputs, self.tokenizer, self.args).strip()

        self.logger.debug(f"NLU raw output: {out}")

        obj = parsing_json(out)

        # --- Evidence gate (prevents "uga" -> FAST, etc.) ---------------------
        awaited = (awaiting_slot or "").strip()
        
        # Closing intent: user indicates they are done / want to finalize
        if awaited == "" and _is_close(user_text):
            return {"intent": "confirm", "slots": {}}

        # Simple acknowledgement: keep conversation open without fallback spam
        if awaited == "" and _is_ack(user_text):
            return {"intent": "out_of_domain", "slots": {"ood_type": "ACK"}}


        # --- Deterministic repair: user answered time_limit but LLM may output out_of_domain ---
        if awaited == "time_limit":
            t = (user_text or "").lower()
            if any(k in t for k in (
                "normal", "regular", "standard", "plenty of time", "a lot of time",
                "lots of time", "free time", "no rush", "take your time"
            )):
                obj = {"intent": "plan", "slots": {"time_limit": "NORMAL"}}
            elif any(k in t for k in (
                "fast", "quick", "short", "asap", "in a hurry", "busy", "no time"
            )):
                obj = {"intent": "plan", "slots": {"time_limit": "FAST"}}


        def _is_interrupting_intent(obj: Any) -> bool:
            if not isinstance(obj, (dict, list)):
                return False

            objs = obj if isinstance(obj, list) else [obj]
            for o in objs:
                if not isinstance(o, dict):
                    continue
                intent = str(o.get("intent") or "")
                slots = o.get("slots") or {}
                ood_type = str(slots.get("ood_type") or "")

                if intent in {"inspect", "show_week", "refine", "help", "confirm"}:
                    return True
                if intent == "out_of_domain" and ood_type == "REFUSE_PENDING":
                    return True

            return False


        def _invalid_answer():
            return {"intent": "out_of_domain", "slots": {"ood_type": "INVALID_ANSWER"}}

        # If we're awaiting a slot but the user is clearly issuing a different valid request
        # (e.g., "show Wed"), do not force INVALID_ANSWER.
        if awaited and _is_interrupting_intent(obj):
            if isinstance(obj, dict):
                return normalize_mr(obj)
            if isinstance(obj, list):
                mrs = [normalize_mr(x) for x in obj if isinstance(x, dict)]
                return mrs if mrs else {"intent": "out_of_domain", "slots": {}}


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
        

