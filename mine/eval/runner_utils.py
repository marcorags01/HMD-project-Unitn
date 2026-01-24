"""Evaluation harness utilities for Meal Kit Composer.

Design goals
- Zero edits to existing system files.
- Reuse the project's model loading, chat formatting, and generation wrappers.
- Provide a stable CLI-friendly way to load models and run prompts.

These helpers mirror the argument post-processing done in utils.get_args():
- args.model_key is the short key (llama3/llama31/qwen3)
- args.model_name is the resolved Hugging Face repo id
- args.prepare_text and args.chat_template are set appropriately
"""

from __future__ import annotations

import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional

import torch

# Ensure project root is importable when running from eval/*
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import MODELS, PREPARE_TEXT, TEMPLATES, load_model  # noqa: E402


def build_args(
    model_key: str = "qwen3",
    device: Optional[str] = None,
    dtype: str = "bf16",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    parallel: bool = False,
    debug: bool = False,
) -> Namespace:
    """Create a utils-compatible Namespace for loading and generation."""

    if model_key not in MODELS:
        raise ValueError(f"Unknown model_key={model_key}. Choices: {sorted(MODELS.keys())}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    args = Namespace()
    args.dtype = dtype
    args.device = device
    args.max_new_tokens = int(max_new_tokens)
    args.temperature = float(temperature)
    args.top_p = float(top_p)
    args.parallel = bool(parallel)
    args.debug = bool(debug)

    # Mirror utils.get_args post-processing
    args.model_key = model_key
    args.model_name = MODELS[model_key]
    args.prepare_text = PREPARE_TEXT.get(model_key, None)
    if args.prepare_text is None:
        args.chat_template = TEMPLATES[model_key]

    return args


def load_llm(args: Namespace):
    """Load model+tokenizer using the project's loader."""
    return load_model(args)


def get_logger(name: str = "eval", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.setLevel(level)
    return logger
