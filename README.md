# HMD-pro — Hybrid LLM + Rule-based Dialogue System for Meal Planning

## Overview

HMD-pro is a task-oriented conversational system that helps users build a 5-day dinner plan (Monday–Friday) under explicit constraints such as dietary restrictions, time, and calorie level.

The system follows a hybrid architecture, combining:
- LLM-based components for Natural Language Understanding (NLU), Dialogue Management (DM), and fallback Natural Language Generation (NLG)
- deterministic policy rules and domain logic to enforce workflow consistency and reliability

The interaction supports constraint specification, menu proposal (two alternatives), iterative refinement, and final plan confirmation with an automatically generated shopping list.

---

## Key Features

- Hybrid LLM + rule-based dialogue architecture
- Structured Meaning Representation (MR) with schema validation
- Deterministic policy layer for workflow enforcement
- Support for:
  - constraint-based planning (servings, time, calories, avoid items)
  - menu selection (2 alternatives)
  - day-level inspection
  - iterative refinement (swap meals, update constraints)
  - shopping list generation
- Built-in evaluation pipeline (NLU, DM, NLG, policy)

---

## Architecture

```text
User → NLU (LLM) → MR → DM (LLM) → Policy (rules) → Executor → NLG

Core components
mine/nlu.py → MR extraction (JSON-based, schema-guided)
mine/dm.py → next action prediction (closed action set)
mine/policy.py → deterministic guardrails and workflow control
mine/support_fn.py → domain logic (menu generation, swaps, repairs)
mine/nlg.py → response generation (deterministic + LLM fallback)
mine/main.py → orchestration and dialogue control
Key Design Choices
Evidence-gated NLU: slot values are only extracted when explicitly supported by user input (no guessing)
One-action-per-turn execution: ensures predictable behavior
Deferred intent handling: multi-intent inputs are processed sequentially via a “continue?” mechanism
Suggest-then-confirm pattern: critical operations (e.g., swaps) require explicit confirmation
Deterministic policy layer: enforces phase constraints and corrects LLM errors

Repository Structure
mine/
  ├── nlu.py
  ├── dm.py
  ├── policy.py
  ├── nlg.py
  ├── main.py
  ├── support_fn.py
  ├── utils.py
  ├── models/
  └── eval/
        ├── run_nlu_eval.py
        ├── run_pipeline_eval.py
        ├── run_policy_tests.py
        ├── metrics.py
        ├── data/
        ├── prompts/
        └── scenarios/

EvalResults/
  └── saved evaluation outputs (NLU, DM, NLG, human eval)

  Evaluation

The project includes intrinsic evaluation for all main components.

NLU (Injection Dataset, n = 1500)
Model	Prompt	Intent Acc	Slot F1	Exact Match
Llama 3.1	v3	0.970	0.903	0.619
Qwen3	v3	0.837	0.848	0.669
Qwen3	v2	0.833	0.823	0.586
Dialogue Management (n = 232 decisions)
Pre-policy accuracy: 0.543
Macro-F1: 0.556
Policy override rate: 0.457

The deterministic policy layer improves reliability by correcting a large portion of raw DM errors.

NLG (n = 30)
BLEU: 0.724
ROUGE-1 (F1): 0.808
ROUGE-2 (F1): 0.771
ROUGE-L (F1): 0.803
Action leak rate: 0.000
Human Evaluation (n = 8 participants)
Overall score: 3.66 / 5

Strengths:

constraint satisfaction
interaction guidance
response clarity

Limitations:

lower trust in final output
moderate performance on clarification handling
Example Interaction
User: I want a quick meal plan for 2 people, low calories, no dairy
System: [collects constraints]

System: Here are two weekly menu options...
User: I choose option 1

User: What’s on Tuesday?
System: [detailed meal description]

User: Can you replace Tuesday?
System: [suggests alternative meal]

User: Yes
System: [applies change]

User: Confirm
System: [final plan + shopping list]
How to Run
Requirements
pip install -r requirements.txt
Run Demo
python mine/main.py

The system requires a HuggingFace-compatible causal language model.
Model loading can be configured in mine/models/.

Running Evaluation
python mine/eval/run_nlu_eval.py
python mine/eval/run_pipeline_eval.py
python mine/eval/run_policy_tests.py

Evaluation outputs are stored in EvalResults/.

Limitations
Requires local LLM inference (no lightweight demo mode)
Dialogue Manager alone has limited accuracy and relies on policy corrections
Dataset is relatively small and domain-specific
Prototype system, not production-ready
Future Work
Improved clarification strategies for underspecified inputs
Increased robustness of dialogue management
Better transparency and explanation of system decisions
Improved reproducibility and packaging
