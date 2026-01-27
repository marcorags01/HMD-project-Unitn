# HMD-pro — Meal Kit Composer (LLM-driven)

A minimal, modular Meal Kit Composer pipeline that demonstrates an LLM-routed conversational assistant for building weekly dinner plans. The system uses small domain modules (NLU, DM, NLG, policy, executor/support functions) plus a local recipe dataset to propose and refine menus interactively.

Key ideas
- LLM-based NLU and DM for flexible, schema-guided interpretation and action selection.
- Deterministic policy + tracker for safe workflow and slot-filling.
- Domain functions to generate menus, swap days, and build shopping lists from a local recipe dataset.

Features
- Intent/slot schema and MR normalization (mine/intents_schema.py)
- LLM-based NLU (mine/nlu.py), Dialogue Manager (mine/dm.py), and NLG (mine/nlg.py)
- Deterministic policy & state tracker (mine/policy.py, mine/support_classes.py)
- Menu generation, swap/repair logic, and shopping-list helpers (mine/support_fn.py)
- Sample recipes dataset: `mine/recipes_30.json`
- Minimal CLI entrypoint: `mine/main.py`

Quickstart (short)
1. Install Python (3.10+) and dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Prepare model weights / environment:
   - The code expects a causal LM via `transformers` (see `mine/utils.py` — it references a `qwen3` wrapper).
   - Adjust model loading or provide model checkpoint as needed for your setup.
3. Run the main loop (simple CLI/demo):
   ```bash
   python -m mine.main
   ```
   or
   ```bash
   python mine/main.py
   ```

Repository layout (important files)
- mine/main.py — orchestration / main controller loop
- mine/nlu.py — convert user text to MR (meaning representation)
- mine/dm.py — dialogue manager (LLM routes to compact action tokens)
- mine/nlg.py — user-facing text generation
- mine/policy.py — deterministic workflow guardrails
- mine/support_fn.py — domain logic (menu generation, swap, shopping list)
- mine/support_classes.py — Tracker, History, normalization helpers
- mine/recipes_30.json — small recipes dataset
- requirements.txt — Python dependencies
- LICENSE — MIT

Notes & caveats
- This repo is designed as a demonstration/prototype: the DM and NLU rely on LLM prompting and may require careful model selection and prompting to be reliable.
- Ensure you supply compatible model weights and enough compute (or use a smaller/open LLM) before running.
- No automated tests are included (add tests for NLU/DM/policy as needed).

License
- MIT (see LICENSE file)

Contributing
- Open an issue or a PR with proposed changes. For improvements, consider:
  - Adding example prompts and model-loading configs
  - A small demo script showcasing a full conversation
  - Unit tests for policy and support functions

Contact
- Repo owner: marcorags01 (https://github.com/marcorags01)
