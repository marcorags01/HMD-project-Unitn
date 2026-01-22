# support_fn.py
"""
Meal Kit Composer — minimal support functions (domain + parsing helpers).

This file is designed to be coherent with:
- the minimal spec in Meal Kit Composer Summary (filter -> score -> choose, deterministic),
- our support_classes.py (Tracker holds state; support_fn performs domain operations).

Core responsibilities (minimal, fully functional):
1) Parse LLM outputs into structured objects (optional but useful for pipeline):
   - parsing_json(text) -> dict | None
   - extract_action_and_argument(text) -> (action, argument) | None

2) Domain logic over a local recipes dataset:
   - load_recipes(path) -> (recipes_list, recipes_by_id)
   - can_generate_menus(tracker) -> (bool, missing_plan_slots)
   - generate_two_menus(recipes, constraints) -> (menu1, menu2)
   - get_day_details(day, menu, recipes_by_id, servings, avoid_items) -> dict
   - swap_day_in_menu(menu, day, recipes, constraints) -> (updated_menu, new_recipe_id|None)
   - repair_menu(menu, recipes, constraints) -> (updated_menu, repaired_days)
   - generate_shopping_list(menu, recipes_by_id, servings) -> list[dict]

Menu representation (minimal internal):
- menu is a dict: {"Mon": "R12", "Tue": "R03", "Wed": "R44", "Thu": "R08", "Fri": "R19"}
"""

from __future__ import annotations

import json
import re
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from support_classes import (
    ALLOWED_DAYS,
    ALLOWED_CALORIE_LEVELS,
    ALLOWED_AVOID_ITEMS,
)

# ------------------------- Constants -------------------------

WEEK_DAYS: List[str] = ["Mon", "Tue", "Wed", "Thu", "Fri"]

TIME_THRESHOLDS_MIN: Dict[str, int] = {
    "FAST": 25,    # <= 25 minutes
    "NORMAL": 40,  # <= 40 minutes
}

REQUIRED_RECIPE_FIELDS = {
    "recipe_id", "title", "time_min", "calorie_level", "servings_base", "contains_tags", "ingredients", "steps"
}
REQUIRED_ING_FIELDS = {"name", "qty", "unit", "category"}

class MenuGenerationError(ValueError):
    """
    User-facing error for menu infeasibility.
    Keep the message safe to show directly to the user.
    """
    pass


def _menu_infeasibility_message(constraints: Dict[str, Any], feasible_count: int) -> str:
    """
    Build a friendly, actionable message without internal jargon like:
    - "constraints"
    - "need at least 5"
    """
    time_limit = str((constraints or {}).get("time_limit") or "").upper()
    calorie = str((constraints or {}).get("calorie_level") or "").upper()
    avoids = (constraints or {}).get("avoid_items") or []

    suggestions: list[str] = []

    # 1–2 concrete relaxations, based on what the user chose
    if time_limit == "FAST":
        suggestions.append("switching to normal prep time")
    if calorie in {"LOW", "HIGH"}:
        suggestions.append("choosing balanced calories")
    if avoids:
        suggestions.append("removing one item to avoid")

    suggestions = suggestions[:2]

    if feasible_count <= 0:
        base = "I couldn’t find meals that match those preferences."
    else:
        base = "I couldn’t put together a full Mon–Fri plan with those preferences."

    if suggestions:
        if len(suggestions) == 1:
            return f"{base} Try {suggestions[0]}, and I can generate options."
        return f"{base} Try {suggestions[0]} or {suggestions[1]}, and I can generate options."

    # If we have no specific suggestion, fall back to a generic gentle prompt
    return f"{base} If you adjust your preferences a bit, I can try again."



def _extract_first_balanced_json_span(text: str) -> Optional[Tuple[int, int]]:
    """
    Return (start_idx, end_idx_exclusive) for the first balanced JSON object/array found in `text`.
    Balanced means:
      - supports {...} and [...]
      - ignores braces/brackets inside JSON strings
      - handles escaped quotes inside strings
    If nothing is found, returns None.
    """
    if not text:
        return None

    # Find first opening of either object or array
    start = None
    opener = None
    for i, ch in enumerate(text):
        if ch == "{" or ch == "[":
            start = i
            opener = ch
            break
    if start is None:
        return None

   

    depth_obj = 0
    depth_arr = 0
    in_string = False
    escape = False

    for j in range(start, len(text)):
        c = text[j]

        if in_string:
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = False
            continue

        # Not in string
        if c == '"':
            in_string = True
            continue

        if c == "{":
            depth_obj += 1
        elif c == "}":
            depth_obj -= 1
            if depth_obj < 0:
                # malformed
                return None
        elif c == "[":
            depth_arr += 1
        elif c == "]":
            depth_arr -= 1
            if depth_arr < 0:
                # malformed
                return None

        # We started at either { or [, so we consider completion when BOTH depths are zero
        if (depth_obj == 0 and depth_arr == 0) and j >= start:
            return (start, j + 1)

    return None

# ------------------------- Parsing helpers -------------------------

def parsing_json(text: str) -> Any:
    """
    Robust JSON parser for LLM outputs.

    Strategy:
      1) Try direct json.loads on the whole text (works if model outputs pure JSON).
      2) Extract the first balanced JSON object/array via brace/bracket balancing and parse it.
      3) If that fails, try to parse the *largest* balanced span found by scanning forward.
    Returns:
      - Parsed Python object (dict or list) on success
      - {} on failure (keeps downstream robust)
    """
    if text is None:
        return {}

    s = str(text).strip()
    if not s:
        return {}

    # Common case: model returned raw JSON only
    try:
        return json.loads(s)
    except Exception:
        pass

    # Remove common fencing without relying on regex extraction of JSON itself
    # (does not harm normal text; helps when model wraps output in ```json ... ```)
    s = s.replace("```json", "```").replace("```JSON", "```").strip()

    # 1) Parse first balanced JSON span
    span = _extract_first_balanced_json_span(s)
    if span is not None:
        start, end = span
        candidate = s[start:end].strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 2) More robust fallback: scan for any later balanced span and keep the longest successfully parsed
    best_obj = None
    best_len = 0
    idx = 0
    while idx < len(s):
        # find next possible opener
        next_open = None
        for k in range(idx, len(s)):
            if s[k] == "{" or s[k] == "[":
                next_open = k
                break
        if next_open is None:
            break

        sub = s[next_open:]
        sub_span = _extract_first_balanced_json_span(sub)
        if sub_span is None:
            idx = next_open + 1
            continue

        a, b = sub_span
        start = next_open + a
        end = next_open + b
        cand = s[start:end].strip()
        try:
            obj = json.loads(cand)
            if (end - start) > best_len:
                best_obj = obj
                best_len = (end - start)
        except Exception:
            pass

        idx = end  # move past this span

    return best_obj if best_obj is not None else {}


# ------------------------- Dataset loading & validation -------------------------

def load_recipes(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Load recipes from a JSON file that contains a list of recipe objects.
    Returns:
      - recipes: list of dicts
      - by_id: dict recipe_id -> recipe_dict
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("recipes dataset must be a JSON list of recipe objects")

    by_id: Dict[str, Dict[str, Any]] = {}
    for r in data:
        _validate_recipe(r)
        rid = str(r["recipe_id"])
        if rid in by_id:
            raise ValueError(f"Duplicate recipe_id: {rid}")
        by_id[rid] = r

    return data, by_id


def _validate_recipe(r: Dict[str, Any]) -> None:
    if not isinstance(r, dict):
        raise ValueError("Each recipe must be a JSON object (dict).")

    missing = REQUIRED_RECIPE_FIELDS - set(r.keys())
    if missing:
        raise ValueError(f"Recipe missing required fields: {sorted(missing)}")

    # Basic type checks (minimal)
    if str(r["calorie_level"]).upper() not in ALLOWED_CALORIE_LEVELS:
        raise ValueError(f"Invalid calorie_level for recipe {r.get('recipe_id')}: {r['calorie_level']}")

    if not isinstance(r["contains_tags"], list):
        raise ValueError(f"contains_tags must be a list for recipe {r.get('recipe_id')}")

    if not isinstance(r["ingredients"], list) or len(r["ingredients"]) == 0:
        raise ValueError(f"ingredients must be a non-empty list for recipe {r.get('recipe_id')}")

    if not isinstance(r["steps"], list) or len(r["steps"]) == 0:
        raise ValueError(f"steps must be a non-empty list for recipe {r.get('recipe_id')}")

    for ing in r["ingredients"]:
        if not isinstance(ing, dict):
            raise ValueError(f"Ingredient must be a dict in recipe {r.get('recipe_id')}")
        missing_ing = REQUIRED_ING_FIELDS - set(ing.keys())
        if missing_ing:
            raise ValueError(f"Ingredient missing fields {sorted(missing_ing)} in recipe {r.get('recipe_id')}")




# ------------------------- Filtering / feasibility -------------------------

def is_feasible(recipe: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
    """
    Feasibility per minimal spec:
    - time_min <= threshold for time_limit
    - calorie_level matches user calorie_level (exact match)
    - contains_tags has no overlap with avoid_items
    """
    # time
    time_limit = (constraints.get("time_limit") or "").upper()
    if time_limit in TIME_THRESHOLDS_MIN:
        if int(recipe["time_min"]) > TIME_THRESHOLDS_MIN[time_limit]:
            return False

    # calorie
    cal = (constraints.get("calorie_level") or "").upper()
    if cal:
        if str(recipe["calorie_level"]).upper() != cal:
            return False

    # avoids
    avoid_items = constraints.get("avoid_items") or []
    avoid_set = {a for a in avoid_items if a in ALLOWED_AVOID_ITEMS}

    tags = recipe.get("contains_tags") or []
    tags_set = {str(t).lower() for t in tags}
    if avoid_set.intersection(tags_set):
        return False

    return True


def filter_recipes(recipes: List[Dict[str, Any]], constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return feasible recipes only (deterministic order preserved by stable sorting)."""
    feasible = [r for r in recipes if is_feasible(r, constraints)]
    # Deterministic tie-break baseline: sort by recipe_id
    return sorted(feasible, key=lambda r: str(r["recipe_id"]))


# ------------------------- Menu generation (deterministic) -------------------------

def generate_two_menus(
    recipes: List[Dict[str, Any]],
    constraints: Dict[str, Any],
    seed: Optional[int] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Generate two menus under the same constraints but different selection priorities:
    - Option 1: promote variety (min overlap of ingredient categories across the week)
    - Option 2: minimize time (fastest recipes)

    Returns:
      menu1, menu2  (each is a dict day->recipe_id)
    """
    rng = random.Random(seed)

    feasible = filter_recipes(recipes, constraints)

    # You need at least 5 feasible recipes to build a 5-day menu
    if len(feasible) < 5:
        raise MenuGenerationError(_menu_infeasibility_message(constraints, len(feasible)))

    menu1 = _build_menu_option1(feasible, rng)
    menu2 = _build_menu_option2(feasible, rng)

    return menu1, menu2

def _recipe_id(r: Dict[str, Any]) -> str:
    return str(r["recipe_id"])


def _ingredient_categories(r: Dict[str, Any]) -> List[str]:
    cats = []
    for ing in r.get("ingredients", []):
        c = str(ing.get("category", "")).strip().lower()
        if c:
            cats.append(c)
    # deterministic unique order
    seen = set()
    out = []
    for c in cats:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _build_menu_option2(feasible: List[Dict[str, Any]], rng: random.Random) -> Dict[str, str]:
    """
    Option 2: random sample 5 distinct feasible recipes.
    """
    if len(feasible) < 5:
        raise MenuGenerationError("Not enough feasible recipes to build a 5-day menu (need at least 5).")

    chosen = rng.sample(feasible, k=5)
    return {day: _recipe_id(chosen[i]) for i, day in enumerate(WEEK_DAYS)}


def _build_menu_option1(feasible: List[Dict[str, Any]], rng: random.Random) -> Dict[str, str]:
    """
    Option 1: greedy variety selection with randomized candidate order to avoid
    always picking the same tie-break winners.
    """
    if len(feasible) < 5:
        raise MenuGenerationError("Not enough feasible recipes to build a 5-day menu (need at least 5).")

    used_ids = set()
    used_cats = set()
    menu: Dict[str, str] = {}

    for day in WEEK_DAYS:
        best = None
        best_key = None

        # randomized iteration order each day
        candidates = feasible[:]     # shallow copy
        rng.shuffle(candidates)

        for r in candidates:
            rid = _recipe_id(r)
            if rid in used_ids:
                continue

            cats = set(_ingredient_categories(r))
            overlap = len(cats.intersection(used_cats))
            unique_cats = len(cats)

            key = (
                overlap,              # minimize overlap first
                -unique_cats,         # then maximize diversity inside recipe
                int(r["time_min"]),   # then prefer shorter time
                rid,                  # still deterministic tie-break after shuffle
            )

            if best is None or key < best_key:
                best = r
                best_key = key

        assert best is not None
        rid = _recipe_id(best)
        menu[day] = rid
        used_ids.add(rid)
        used_cats.update(_ingredient_categories(best))

    return menu



# ------------------------- Inspect helpers -------------------------

def get_day_details(
    day: str,
    menu: Dict[str, str],
    recipes_by_id: Dict[str, Dict[str, Any]],
    servings: int,
    avoid_items: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Return the standard inspect payload for a given day in a menu:
    - title, time_min, calorie_level
    - ingredients scaled to servings
    - avoid_check (True if recipe contains any avoid tags)
    """
    if day not in ALLOWED_DAYS:
        raise ValueError(f"Invalid day: {day}")

    if day not in menu:
        raise ValueError(f"Day {day} not present in menu.")

    rid = str(menu[day])
    recipe = recipes_by_id.get(rid)
    if recipe is None:
        raise KeyError(f"Recipe id not found in dataset: {rid}")

    scaled_ings = scale_ingredients(recipe, servings)

    avoid_items = avoid_items or []
    avoid_set = {a for a in avoid_items if a in ALLOWED_AVOID_ITEMS}
    tags = {str(t).lower() for t in (recipe.get("contains_tags") or [])}
    avoid_check = bool(avoid_set.intersection(tags))
    steps = recipe.get("steps") or []
    steps = [str(s).strip() for s in steps if str(s).strip()]

    return {
        "day": day,
        "servings": int(servings),
        "recipe_id": rid,
        "title": recipe["title"],
        "time_min": int(recipe["time_min"]),
        "calorie_level": str(recipe["calorie_level"]).upper(),
        "ingredients": scaled_ings,
        "steps": steps,
        "avoid_check": avoid_check,
    }


# ------------------------- Refine helpers (swap / repair / avoid update) -------------------------

def swap_day_in_menu(
    menu: Dict[str, str],
    day: str,
    recipes: List[Dict[str, Any]],
    constraints: Dict[str, Any],
) -> Tuple[Dict[str, str], Optional[str]]:
    """
    Deterministic SWAP_DAY(day, value=BEST_FIT).

    Behavior:
    - remove current day's recipe id
    - choose best feasible alternative not already used in the week if possible
    - tie-break: minimum time_min, then recipe_id
    - returns (updated_menu, new_recipe_id or None if swap not possible)
    """
    if day not in ALLOWED_DAYS:
        raise ValueError(f"Invalid day: {day}")
    if day not in menu:
        raise ValueError(f"Day {day} not present in menu.")

    current_id = str(menu[day])
    used_ids = {str(v) for v in menu.values()}

    feasible = filter_recipes(recipes, constraints)

    def rank_key(r: Dict[str, Any]) -> Tuple[int, str]:
        return (int(r["time_min"]), _recipe_id(r))

    # Pass 1: prefer recipes not used in the current week
    candidates_1 = [
        r for r in feasible
        if _recipe_id(r) != current_id and _recipe_id(r) not in used_ids
    ]
    candidates_1.sort(key=rank_key)

    # Pass 2: allow reuse (still avoid keeping the same recipe)
    candidates_2 = [
        r for r in feasible
        if _recipe_id(r) != current_id
    ]
    candidates_2.sort(key=rank_key)

    chosen = candidates_1[0] if candidates_1 else (candidates_2[0] if candidates_2 else None)
    if chosen is None:
        return dict(menu), None

    new_id = _recipe_id(chosen)
    updated = dict(menu)
    updated[day] = new_id
    return updated, new_id

def suggest_swap_day_in_menu(
    menu: Dict[str, str],
    day: str,
    recipes: List[Dict[str, Any]],
    constraints: Dict[str, Any],
) -> Optional[str]:
    """
    Deterministic suggestion for an alternative on `day` without mutating the menu.

    Returns:
      - new_recipe_id (str) if a feasible alternative exists
      - None if no alternative exists

    Selection logic mirrors swap_day_in_menu():
    - choose best feasible alternative not already used in the week if possible
    - tie-break: minimum time_min, then recipe_id
    """
    if day not in ALLOWED_DAYS:
        raise ValueError(f"Invalid day: {day}")
    if day not in menu:
        raise ValueError(f"Day {day} not present in menu.")

    current_id = str(menu[day])
    used_ids = {str(v) for v in menu.values()}

    feasible = filter_recipes(recipes, constraints)

    def rank_key(r: Dict[str, Any]) -> Tuple[int, str]:
        return (int(r["time_min"]), _recipe_id(r))

    # Pass 1: prefer recipes not used in the current week
    candidates_1 = [
        r for r in feasible
        if _recipe_id(r) != current_id and _recipe_id(r) not in used_ids
    ]
    candidates_1.sort(key=rank_key)

    # Pass 2: allow reuse (still avoid keeping the same recipe)
    candidates_2 = [
        r for r in feasible
        if _recipe_id(r) != current_id
    ]
    candidates_2.sort(key=rank_key)

    chosen = candidates_1[0] if candidates_1 else (candidates_2[0] if candidates_2 else None)
    if chosen is None:
        return None

    return _recipe_id(chosen)


def repair_menu(
    menu: Dict[str, str],
    recipes_by_id: Dict[str, Dict[str, Any]],
    recipes: List[Dict[str, Any]],
    constraints: Dict[str, Any],
) -> Tuple[Dict[str, str], List[str]]:
    """
    After avoid constraint changes:
    - iterate Mon–Fri
    - if a day violates constraints, swap that day (BEST_FIT)
    Returns (updated_menu, repaired_days)
    """
    updated = dict(menu)
    repaired_days: List[str] = []

    for day in WEEK_DAYS:
        rid = str(updated.get(day, ""))
        recipe = recipes_by_id.get(rid)
        if recipe is None:
            continue

        if not is_feasible(recipe, constraints):
            updated, new_id = swap_day_in_menu(updated, day, recipes, constraints)
            if new_id is not None:
                repaired_days.append(day)

    return updated, repaired_days


def update_avoid_items(
    constraints: Dict[str, Any],
    op: str,
    item: str,
) -> Tuple[bool, Optional[str]]:
    """
    Update constraints['avoid_items'] by adding/removing one item.
    Returns (ok, error_message).
    """
    if not item:
        return False, "Missing avoid item value."

    it = str(item).lower().strip()
    if it not in ALLOWED_AVOID_ITEMS:
        return False, f"Unknown avoid item: {it}"

    avoid = constraints.get("avoid_items") or []
    avoid_set = set([a for a in avoid if a in ALLOWED_AVOID_ITEMS])

    op_up = str(op).upper().strip()
    if op_up == "ADD_AVOID_ITEM":
        avoid_set.add(it)
    elif op_up == "REMOVE_AVOID_ITEM":
        avoid_set.discard(it)
    else:
        return False, f"Unknown avoid operation: {op}"

    # Keep deterministic ordering
    constraints["avoid_items"] = sorted(avoid_set)
    return True, None


# ------------------------- Shopping list (scaling + aggregation) -------------------------

_WS_RE = re.compile(r"\s+")

def _normalize_unit(unit: str) -> str:
    u = (unit or "").strip().lower()
    unit_map = {
        "pc": "pc",
        "piece": "pc",
        "pieces": "pc",
        "pcs": "pc",
        "g": "g",
        "gram": "g",
        "grams": "g",
        "ml": "ml",
        "milliliter": "ml",
        "milliliters": "ml",
    }
    return unit_map.get(u, u)

def _normalize_category(category: str) -> str:
    c = (category or "").strip().lower()
    if not c:
        return "other"
    cat_map = {
        "produce": "produce",
        "pantry": "pantry",
        "dairy": "dairy",
        "protein": "protein",
        "spices": "spices",
    }
    return cat_map.get(c, c)

def _singularize_token(tok: str) -> str:
    # Conservative English singularization for simple plurals.
    # Only apply to alphabetic tokens.
    if not tok.isalpha():
        return tok

    # Common “do not touch” cases (mass nouns / tricky endings)
    exceptions = {
        "rice", "pasta", "glass", "asparagus", "couscous",
    }
    if tok in exceptions:
        return tok

    if tok.endswith("ies") and len(tok) > 3:
        return tok[:-3] + "y"
    if tok.endswith("oes") and len(tok) > 3:
        return tok[:-2]  # tomatoes -> tomato, potatoes -> potato
    if tok.endswith("ches") or tok.endswith("shes") or tok.endswith("xes") or tok.endswith("zes"):
        return tok[:-2]  # drop 'es'
    if tok.endswith("s") and not tok.endswith("ss") and len(tok) > 1:
        return tok[:-1]

    return tok

def _normalize_ingredient_name(name: str) -> str:
    s = (name or "").strip().lower()
    s = s.strip(" ,;")
    s = re.sub(r"[,:;]", " ", s)
    s = s.replace("-", " ")
    s = _WS_RE.sub(" ", s).strip()

    # If there is a parenthetical suffix, singularize only the "base" part.
    if "(" in s:
        base, rest = s.split("(", 1)
        base = base.strip()
        rest = "(" + rest  # add back '('
    else:
        base, rest = s, ""

    toks = base.split()
    if toks:
        toks[-1] = _singularize_token(toks[-1])
    base_norm = " ".join(toks).strip()

    return (base_norm + (" " + rest if rest else "")).strip()



def scale_ingredients(recipe: Dict[str, Any], servings: int) -> List[Dict[str, Any]]:
    """
    Scale ingredient quantities by:
      scale = servings / servings_base
      scaled_qty = qty * scale

    Rounding (minimal recommended):
      - g/ml: round to nearest 1 (small) or 5 (larger) for readability
      - pc: round to nearest int, minimum 1 if nonzero amount is required
    """
    base = int(recipe.get("servings_base", 2))
    if base <= 0:
        base = 2

    scale = float(servings) / float(base)

    scaled: List[Dict[str, Any]] = []
    for ing in recipe.get("ingredients", []):
        raw_name = ing.get("name", "")
        name = _normalize_ingredient_name(raw_name)
      
        if not name:
            # Optional during debugging: raise to find the offending ingredient precisely
            # raise ValueError(f"Ingredient missing/blank name in recipe {recipe.get('recipe_id')}: {ing}")
            continue

        raw_unit = ing.get("unit", "")
        unit = _normalize_unit(raw_unit)

        raw_category = ing.get("category", "")
        category = _normalize_category(raw_category)

        qty = ing.get("qty", 0)

        try:
            q = float(qty) * scale
        except Exception:
            # If qty is malformed, skip rather than injecting junk
            continue

        q_rounded = _round_qty(q, unit)
        
        scaled.append({
            "name": name,
            "qty": q_rounded,
            "unit": unit,
            "category": category,
        })

    return scaled


def _round_qty(qty: float, unit: str) -> float:
    u = (unit or "").lower().strip()
    if qty <= 0:
        return 0.0

    if u in ("pc"):
        # nearest int, minimum 1
        return float(max(1, int(round(qty))))

    if u in ("g", "ml"):
        # minimal readable rounding: nearest 1 if small, else nearest 5
        if qty < 20:
            return float(int(round(qty)))
        return float(int(round(qty / 5.0) * 5))

    # fallback: 1 decimal place
    return round(qty, 1)


def generate_shopping_list(
    menu: Dict[str, str],
    recipes_by_id: Dict[str, Dict[str, Any]],
    servings: int,
) -> List[Dict[str, Any]]:
    """
    Consolidated shopping list:
    - aggregate ingredient quantities across Mon–Fri
    - quantities are scaled to servings
    Output is a list of {name, qty, unit, category} sorted by category then name.
    """
    agg: Dict[Tuple[str, str, str], float] = defaultdict(float)

    for day in WEEK_DAYS:
        rid = str(menu.get(day, ""))
        recipe = recipes_by_id.get(rid)
        if recipe is None:
            continue

        for ing in scale_ingredients(recipe, servings):
            key = (str(ing["name"]), str(ing["unit"]), str(ing["category"]))
            agg[key] += float(ing["qty"])

    out: List[Dict[str, Any]] = []
    for (name_lc, unit, category), qty in agg.items():
        out.append({
            "name": name_lc,  # keep lowercase for consistent aggregation
            "qty": _round_qty(qty, unit),
            "unit": unit,
            "category": category,
        })

    out.sort(key=lambda x: (str(x["category"]).lower(), str(x["name"]).lower(), str(x["unit"]).lower()))
    return out
