"""
Meal Kit Composer — shared utilities.

This module centralizes:
- PROMPTS for DM/NLG (and optionally NLU later)
- model registry + chat templates
- CLI args + model loading
- generate() wrapper

It is designed to be dependency-light and compatible with:
- support_classes.py (History/Tracker)
- intents_schema.py (MR schema/validation)
- support_fn.py (domain functions + parsing helpers)
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from typing import Tuple, Callable, Dict, Optional
from functools import partial
import re

from models import qwen3 

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizer,
    PreTrainedModel,
    BitsAndBytesConfig,
)


# ------------------------- Prompts -------------------------

PROMPTS = {
    "START": "Hello — I’m the Meal Kit Composer.\nMy job is to turn your preferences into a practical weekly dinner plan: balanced recipes, clear constraints, and easy iteration.\nI’ll start by asking a few quick questions (servings, prep time, calorie target, avoid-list), then I’ll generate two menu options.\nAfter you choose, you can inspect any day and request changes until your week plan is exactly right.\nLet me know when you’re ready to begin!",

    # DM prompt parts (LLM must output exactly ONE compact action)
    "DM_START": """You are the Dialogue Manager (DM) for a Meal Kit Composer assistant.

You will be given:
- SELECTED_MR: the single NLU Meaning Representation (MR) to address NOW (JSON)
- tracker_state: the current tracker state (JSON), which may include pending_mrs (a backlog)
- recent dialogue turns

Your job:
- Decide the next best action for SELECTED_MR only.
- Output EXACTLY ONE next action in the specified compact format.
- Use pending_mrs only as context (do NOT try to satisfy multiple pending requests in one turn).

Return ONLY the action. No explanations. No extra text.
""",


"DM_ACTIONS": """ALLOWED ACTIONS (choose exactly one):
- request_info(slot)
- provide_info(intent, slot)
- propose_menus()
- set_active_menu(menu_id)
- show_day(target_day)
- show_week()
- suggest_swap_day(target_day)
- swap_day(target_day)
- update_avoid(op, value)
- confirm_plan()
- fallback()
""",

# We keep the “hard rules” in prompt for LLM guidance;
# deterministic guard rails will still be implemented outside the DM (policy.py).
"DM_RULES": """HARD WORKFLOW RULES (never violate):
0) You must choose ONE action for SELECTED_MR only. pending_mrs is context only.
0b) If SELECTED_MR is a synthetic/empty MR (e.g., {"intent":"plan","slots":{}}), ask for the next missing slot using request_info(...).
1) If PLAN is incomplete (missing servings/time_limit/calorie_level/avoid_items), choose request_info(one missing slot).
2) After menus are proposed, the user must select menu_id before inspect/refine/confirm.
3) If the user asks to "suggest/propose an alternative" for a day, choose suggest_swap_day(target_day).
   Only use swap_day(target_day) if the user explicitly asks to swap/change, or confirms a suggestion.
4) If inspect is requested and target_day is missing, choose request_info(target_day).
5) If refine is requested and refine_type is missing, choose request_info(refine_type).
   - If refine_type=SWAP_DAY and target_day missing -> request_info(target_day).
   - If refine_type=ADD_AVOID_ITEM or REMOVE_AVOID_ITEM and value missing -> request_info(value).
6) If out_of_domain, choose fallback().

SLOT/VALUE NOTES:
- menu_id must be 1 or 2
- target_day must be one of Mon Tue Wed Thu Fri
- suggest_swap_day uses BEST_FIT and does NOT commit the change; the user must confirm to apply it
- swap_day always uses BEST_FIT (do not ask for other values)
- update_avoid op is ADD_AVOID_ITEM or REMOVE_AVOID_ITEM
- avoid_items can be an empty list if the user has no restrictions (explicit “none”)

""",

    "DM_END": """OUTPUT FORMAT:
Return only one action in this exact format, e.g.:
request_info(servings)
set_active_menu(1)
show_day(Tue)
suggest_swap_day(Mon)
swap_day(Tue)
update_avoid(ADD_AVOID_ITEM, nuts)
confirm_plan()
fallback()

Return ONE action for SELECTED_MR only.
""",

    # NLG prompt parts
    "NLG_START": """You are the NLG component for a Meal Kit Composer assistant.
You will be given:
- the DM action in compact format (e.g., request_info(servings))
- the relevant tracker state and/or payload
- recent dialogue turns

Write a concise, helpful user-facing message consistent with the action.
Do not output the action string. Output ONLY the message to the user.
""",

    "NLG_END": """Return ONLY the user-facing message. No extra text.""",
}



def extract_action_and_argument(input_string: str) -> Optional[Tuple[str, str]]:
    """
    Parse a DM “function-call string” like:
      request_info(servings)
      set_active_menu(1)
      show_day(Tue)
      suggest_swap_day(Mon)
      update_avoid(ADD_AVOID_ITEM, nuts)
      confirm_plan()

    Returns (action, argument_string). If there's no argument, returns ("CONFIRM", "").

    """
    if not input_string:
        return None

    s = input_string.strip()
    # Remove common quoting artifacts
    s = s.replace("`", "").replace("\"", "").replace("'", "")

    match = re.match(r"(\w+)\((.*?)\)\s*$", s)
    if not match:
        return None

    action = match.group(1)
    arg = match.group(2).strip()

    # If it's "key=value", keep only the value 
    if "=" in arg:
        # Keep right-hand side of the first "="
        arg = arg.split("=", 1)[1].strip()

    return action, arg




# ------------------------- Model registry + templates -------------------------
# You can extend these later. For now, keep it minimal and stable.

MODELS: Dict[str, str] = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama31": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3": "Qwen/Qwen3-4B-Instruct-2507",  # <-- adjust if your course uses a different id
}

# Per-model loader (Qwen often needs trust_remote_code=True in course repos)
MODEL_LOADERS: Dict[str, Callable[..., PreTrainedModel]] = {
    "llama3": AutoModelForCausalLM.from_pretrained,
    "llama31": AutoModelForCausalLM.from_pretrained,
    "qwen3": partial(AutoModelForCausalLM.from_pretrained, trust_remote_code=True),
}

# Per-model tokenizer kwargs
TOKENIZER_KWARGS: Dict[str, Dict[str, object]] = {
    "llama3": {},
    "llama31": {},
    "qwen3": {"trust_remote_code": True},
}

# If prepare_text is callable, we'll use tokenizer.apply_chat_template formatting (Qwen style).
PREPARE_TEXT: Dict[str, Optional[Callable[..., str]]] = {
    "llama3": None,
     "llama31": None,
    "qwen3": qwen3.prepare_text,
}

TEMPLATES = {
    # System + user -> assistant
    "llama3": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>"
    ),
    "llama31": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>"
    ),
}

def _flatten_token_ids(x: object) -> list[int]:
    """
    Convert token id containers into a flat list[int].
    Accepts:
      - torch.Tensor (1D/2D)
      - list/tuple (possibly nested)
      - int-like scalars
    """
    # torch tensor
    if hasattr(x, "detach") and hasattr(x, "tolist"):
        x = x.detach().cpu().tolist()

    # batched [[...]] -> take first row
    if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple)):
        x = x[0]

    if isinstance(x, (list, tuple)):
        out: list[int] = []
        for t in x:
            try:
                out.append(int(t))
            except Exception:
                # skip non-int-like entries
                return []
        return out

    # scalar
    try:
        return [int(x)]
    except Exception:
        return []



def _as_str_prompt(
    out: object,
    tokenizer: PreTrainedTokenizer,
) -> str:
    """
    Normalize any chat-template output to a plain string prompt.

    Handles:
    - str
    - BatchEncoding / dict with input_ids (decode deterministically)
    - list/tuple of token ids (including int-like types) possibly nested (batch)
    """
    if isinstance(out, str):
        return out

    # BatchEncoding or dict-like pretokenized output
    if isinstance(out, BatchEncoding):
        data = out.data
        ids = data.get("input_ids", None)
        if ids is not None:
            # ids might be tensor/list/nested list
            return tokenizer.decode(_flatten_token_ids(ids), skip_special_tokens=False)
        raise TypeError("prepare_text returned BatchEncoding without input_ids")

    if isinstance(out, dict) and "input_ids" in out:
        return tokenizer.decode(_flatten_token_ids(out["input_ids"]), skip_special_tokens=False)

    # List/tuple tokens: could be [ids] or [[ids]] (batched)
    if isinstance(out, (list, tuple)):
        flat = _flatten_token_ids(out)
        if flat:
            return tokenizer.decode(flat, skip_special_tokens=False)

    # Last resort: fail loudly rather than passing garbage downstream
    raise TypeError(f"prepare_text returned unsupported type for prompt: {type(out)}")


def format_chat(
    args: Namespace,
    system_prompt: str,
    user_text: str,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> str:
    """
    Format a system+user exchange.
    - LLaMA: string template in args.chat_template
    - Qwen: tokenizer.apply_chat_template via models.qwen3.prepare_text
    """
    prepare_fn = getattr(args, "prepare_text", None)

    if callable(prepare_fn):
        if tokenizer is None:
            raise ValueError("tokenizer is required when using apply_chat_template formatting")

        # qwen3.prepare_text appends the user message; we inject system here
        messages = [{"role": "system", "content": system_prompt}]
        out = prepare_fn(user_text, tokenizer, messages=messages, n_exchanges=1)

        return _as_str_prompt(out, tokenizer)

    return args.chat_template.format(system_prompt, user_text)


# ------------------------- CLI args / loading -------------------------

def get_args() -> Namespace:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
    "--dtype",
    type=str,
    choices=["f32", "fp16", "bf16"],
    default="bf16",
    help="Model dtype.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (e.g., cuda, cuda:0, cpu).",
    )

    parser.add_argument(
        "model_name",
        type=str,
        choices=list(MODELS.keys()),
        help="Which model key to use (e.g., llama3, qwen3).",
    )

    
   
    parser.add_argument(
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens per generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 for deterministic; >0 enables sampling.",
    )
    parser.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=1.0,
        help="Top-p sampling (only used if temperature > 0).",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Split model across devices with device_map='auto'.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug prints (for development; keep off for evaluation).",
    )

    args = parser.parse_args()

    # Keep the model key and resolve HF id
    args.model_key = args.model_name
    args.model_name = MODELS[args.model_key]

    # Attach prepare_text callable if the model uses apply_chat_template (Qwen)
    args.prepare_text = PREPARE_TEXT.get(args.model_key, None)

    # Only set chat_template for non-apply_chat_template models (e.g., llama3)
    if args.prepare_text is None:
        args.chat_template = TEMPLATES[args.model_key]

    return args



def load_model(args: Namespace) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    use_auto_map = bool(getattr(args, "parallel", False))
    model_key = getattr(args, "model_key", None) or "llama3"

    # ---- dtype selection (handles f32/fp16/bf16 + safe bf16 fallback) ----
    if args.dtype == "f32":
        tdtype = torch.float32
    elif args.dtype == "fp16":
        tdtype = torch.float16
    else:  # "bf16"
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            tdtype = torch.float16  # Azure T4 etc.
        else:
            tdtype = torch.bfloat16

    loader_fn = MODEL_LOADERS[model_key]

    # Quantize llama31 to fit in ~16GB VRAM
    quant_config = None
    if model_key == "llama31" and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    model_kwargs = dict(
        device_map="auto" if (use_auto_map or quant_config is not None) else None,
        quantization_config=quant_config,
    )

    if quant_config is None:
        model_kwargs["torch_dtype"] = tdtype

    model = loader_fn(args.model_name, **model_kwargs)




    # Only manually move if we're NOT using device_map and NOT quantizing
    if not (use_auto_map or quant_config is not None):
        model = model.to(args.device)

    tok_kwargs = dict(TOKENIZER_KWARGS.get(model_key, {}))

    # Fast tokenizers occasionally throw TextEncodeInput errors on some setups.
    # Use the slow tokenizer for stability.
    if model_key in {"llama3", "llama31"}:
        tok_kwargs["use_fast"] = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tok_kwargs)

    

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    
    return model, tokenizer  # type: ignore


def infer_input_device(model) -> torch.device:
    # Best signal: where embeddings live (first op in most decoder LMs)
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight") and emb.weight is not None:
            return emb.weight.device
    except Exception:
        pass

    # Next: explicit model.device if present
    dev = getattr(model, "device", None)
    if dev is not None:
        return torch.device(dev)

    # Fallback: hf_device_map (pick the first real device)
    dm = getattr(model, "hf_device_map", None)
    if isinstance(dm, dict):
        for d in dm.values():
            if isinstance(d, str) and d not in {"disk"}:
                return torch.device(d)
    return torch.device("cpu")




# ------------------------- Generation wrapper -------------------------

def _eos_ids_for_model(tokenizer: PreTrainedTokenizer, args: Namespace):
    eos = tokenizer.eos_token_id
    model_key = getattr(args, "model_key", "") or ""
    if model_key in {"llama3", "llama31"}:
        try:
            eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot is not None and eot != eos:
                return [eos, eot]
        except Exception:
            pass
    return eos


def generate(
    model: PreTrainedModel,
    inputs: BatchEncoding,
    tokenizer: PreTrainedTokenizer,
    args: Namespace,
) -> str:
    """
    Minimal generation wrapper, with optional sampling.
    """
    do_sample = getattr(args, "temperature", 0.0) > 0.0

    gen_kwargs = dict(
        max_new_tokens=getattr(args, "max_new_tokens", 64),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=_eos_ids_for_model(tokenizer, args),
        do_sample=do_sample,
    )

    if do_sample:
        gen_kwargs["temperature"] = float(getattr(args, "temperature", 0.7))
        gen_kwargs["top_p"] = float(getattr(args, "top_p", 0.9))

    

    with torch.inference_mode():
        output = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            **gen_kwargs,
        )

    return tokenizer.decode(output[0][len(inputs.input_ids[0]) :], skip_special_tokens=True).strip()
