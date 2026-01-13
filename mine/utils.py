# utils.py
"""
Meal Kit Composer — shared utilities (Marina-like).

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


from models import qwen3  # ensure models/__init__.py exists

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizer,
    PreTrainedModel,
)


# ------------------------- Prompts -------------------------

PROMPTS = {
    "START": "Hi, I am your Meal Kit Composer assistant. How can I help you?",

    # DM prompt parts (LLM must output exactly ONE compact action)
    "DM_START": """You are the Dialogue Manager (DM) for a Meal Kit Composer assistant.
You will be given:
- the latest NLU Meaning Representation (MR) in JSON
- the current tracker state in JSON
- recent dialogue turns

Your job: output EXACTLY ONE next action in the specified compact format.
Return ONLY the action. No explanations. No extra text.
""",

    "DM_ACTIONS": """ALLOWED ACTIONS (choose exactly one):
- request_info(slot)
- provide_info(intent, slot)
- propose_menus()
- set_active_menu(menu_id)
- show_day(target_day)
- swap_day(target_day)
- update_avoid(op, value)
- confirm_plan()
- fallback()
""",

    # We keep the “hard rules” in prompt for LLM guidance;
    # deterministic guard rails will still be implemented outside the DM (policy.py).
    "DM_RULES": """HARD WORKFLOW RULES (never violate):
1) If PLAN is incomplete (missing servings/time_limit/calorie_level), choose request_info(one missing slot).
2) After menus are proposed, the user must select menu_id before inspect/refine/confirm.
3) If inspect is requested and target_day is missing, choose request_info(target_day).
4) If refine is requested and refine_type is missing, choose request_info(refine_type).
   - If refine_type=SWAP_DAY and target_day missing -> request_info(target_day).
   - If refine_type=ADD_AVOID_ITEM or REMOVE_AVOID_ITEM and value missing -> request_info(value).
5) If out_of_domain, choose fallback().

SLOT/VALUE NOTES:
- menu_id must be 1 or 2
- target_day must be one of Mon Tue Wed Thu Fri
- swap_day always uses BEST_FIT (do not ask for other values)
- update_avoid op is ADD_AVOID_ITEM or REMOVE_AVOID_ITEM
""",

    "DM_END": """OUTPUT FORMAT:
Return only one action in this exact format, e.g.:
request_info(servings)
set_active_menu(1)
swap_day(Tue)
update_avoid(ADD_AVOID_ITEM, nuts)
confirm_plan()
""",

    # NLG prompt parts (will be used by component/nlg.py)
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


# ------------------------- Model registry + templates -------------------------
# You can extend these later. For now, keep it minimal and stable.

MODELS: Dict[str, str] = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen3": "Qwen/Qwen3-4B-Instruct-2507",  # <-- adjust if your course uses a different id
}

# Per-model loader (Qwen often needs trust_remote_code=True in course repos)
MODEL_LOADERS: Dict[str, Callable[..., PreTrainedModel]] = {
    "llama3": AutoModelForCausalLM.from_pretrained,
    "qwen3": partial(AutoModelForCausalLM.from_pretrained, trust_remote_code=True),
}

# Per-model tokenizer kwargs
TOKENIZER_KWARGS: Dict[str, Dict[str, object]] = {
    "llama3": {},
    "qwen3": {"trust_remote_code": True},
}

# If prepare_text is callable, we'll use tokenizer.apply_chat_template formatting (Qwen style).
PREPARE_TEXT: Dict[str, Optional[Callable[..., str]]] = {
    "llama3": None,
    "qwen3": qwen3.prepare_text,
}

TEMPLATES = {
    # System + user -> assistant
    "llama3": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>"
    ),
}

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
        return prepare_fn(user_text, tokenizer, messages=messages, n_exchanges=1)

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

    model = loader_fn(
        args.model_name,
        device_map="auto" if use_auto_map else None,
        torch_dtype=tdtype,
    )

    if not use_auto_map:
        model = model.to(args.device)

    tok_kwargs = TOKENIZER_KWARGS.get(model_key, {})
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tok_kwargs)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer  # type: ignore



# ------------------------- Generation wrapper -------------------------

def generate(
    model: PreTrainedModel,
    inputs: BatchEncoding,
    tokenizer: PreTrainedTokenizer,
    args: Namespace,
) -> str:
    """
    Minimal generation wrapper consistent with Marina’s style, with optional sampling.
    """
    do_sample = getattr(args, "temperature", 0.0) > 0.0

    gen_kwargs = dict(
        max_new_tokens=getattr(args, "max_new_tokens", 64),
        pad_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
    )

    if do_sample:
        gen_kwargs["temperature"] = float(getattr(args, "temperature", 0.7))
        gen_kwargs["top_p"] = float(getattr(args, "top_p", 0.9))

    output = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        **gen_kwargs,
    )

    return tokenizer.decode(output[0][len(inputs.input_ids[0]) :], skip_special_tokens=True).strip()
