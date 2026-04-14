# Hybrid LLM + Rule-Based Conversational System for Meal Planning

> Course project for *Human-Machine Dialogue (2025)* — University of Trento  
> A hybrid conversational AI system combining LLMs and deterministic policies to generate and refine weekly meal plans under user constraints.

## Overview

HMD-pro is a task-oriented conversational system that helps users generate a 5-day dinner plan (Monday–Friday) under explicit constraints such as dietary restrictions, preparation time, and calorie level.

The system follows a **hybrid architecture**, combining:
- Large Language Models (LLMs) for Natural Language Understanding (NLU) and Dialogue Management (DM)
- Deterministic policy rules for enforcing workflow consistency, safety, and reliability

---

## Key Features

- Hybrid LLM + rule-based dialogue architecture  
- Structured Meaning Representation (MR) with schema validation  
- Deterministic policy layer for workflow enforcement  
- Constraint-based planning:
  - servings
  - time constraints
  - calorie level
  - dietary restrictions  
- Menu generation (2 alternatives) with user selection  
- Interactive refinement:
  - day-level inspection  
  - meal replacement (swap)  
  - constraint updates  
- Automatic shopping list generation  
- Full evaluation pipeline (NLU, DM, NLG, policy)  

---

## System Architecture

```
User → NLU (LLM) → Meaning Representation (JSON)
     → Dialogue Manager (LLM)
     → Policy Layer (deterministic rules)
     → Executor (domain logic)
     → NLG (deterministic + LLM fallback)
```

### Design Highlights

- **Evidence-gated NLU**  
  Slot values are extracted only when explicitly supported by user input (no hallucination)

- **One-action-per-turn execution**  
  Ensures predictable and controllable dialogue flow

- **Deferred intent handling**  
  Multi-intent inputs are processed sequentially via a “continue?” mechanism

- **Suggest-then-confirm pattern**  
  Critical operations (e.g., swaps) require explicit user confirmation

- **Policy override mechanism**  
  A deterministic layer corrects Dialogue Manager errors

---

## Tech Stack

- Python  
- HuggingFace Transformers  
- LLMs (Llama 3.1, Qwen3)  
- JSON-based Meaning Representation schema  
- Rule-based policy engine  

---
## Repository Structure

```
mine/
├── main.py                 # Dialogue orchestration and control loop
├── nlu.py                  # Meaning Representation extraction (LLM-based)
├── dm.py                   # Dialogue Manager (next-action prediction)
├── policy.py               # Deterministic policy and guardrails
├── nlg.py                  # Response generation (template + LLM fallback)
├── support_fn.py           # Domain logic (menu generation, swaps, repairs)
├── support_classes.py      # Dialogue state tracker and core data structures
├── intents_schema.py       # Definition and validation of intents and slots
├── utils.py                # Utility functions
├── models/                 # Model loading and configuration
├── eval/                   # Evaluation pipeline
│   ├── run_nlu_eval.py
│   ├── run_pipeline_eval.py
│   ├── run_policy_tests.py
│   ├── metrics.py
│   ├── data/
│   ├── prompts/
│   └── scenarios/
├── data/
│   └── recipes_30.json     # Recipe dataset used for menu generation

EvalResults/
└── evaluation outputs
```



---

## Evaluation (Summary)

### NLU (n = 1500)
- Intent Accuracy: **0.97**
- Slot F1: **0.90**
- Exact Match: **0.62**

### Dialogue Management (n = 232)
- Accuracy (pre-policy): **0.54**
- Macro-F1: **0.56**
- Policy override rate: **45.7%**

### NLG
- BLEU: **0.72**
- ROUGE-L: **0.80**
- Action leak rate: **0.00**

### Human Evaluation (n = 8)
- Overall score: **3.66 / 5**

**Key insight:**  
The deterministic policy layer significantly improves system reliability by correcting a large portion of Dialogue Manager errors.

---

For a deeper technical analysis, refer to the full report:

## Project Report

📄 **Technical Report**  
[Download PDF](./docs/MarcoRagusa_HMDreport.pdf)

A detailed report covering system design, conversation modeling, and evaluation:

- Hybrid LLM + rule-based architecture
- Meaning Representation schema (intents and slots)
- Intrinsic evaluation (NLU, DM, NLG)
- Policy effectiveness and error correction
- Human evaluation study (n = 8 participants)

---


## Example Interaction

```
User: I want a quick meal plan for 2 people, low calories, no dairy
System: [collects constraints]

System: Here are two weekly menu options...
User: I choose option 1

User: What’s on Tuesday?
System: [meal details]

User: Replace Tuesday
System: [suggests alternative]

User: Yes
System: [applies change]

User: Confirm
System: [final plan + shopping list]
```

---

## How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

---

### 2. Set up a language model

The system requires a HuggingFace-compatible causal language model for NLU and Dialogue Management.

Recommended options:

- `meta-llama/Meta-Llama-3.1-8B-Instruct` (best performance, requires GPU)
- `Qwen/Qwen3-4B-Instruct` (lighter alternative)

You can configure the model in:

```
mine/models/
```

Make sure you have access to the model and have authenticated with HuggingFace if required:

```
huggingface-cli login
```

---

### 3. Run the system

```
python mine/main.py
```

The system will start an interactive dialogue in the terminal.

---

### 4. Run evaluation

```
python mine/eval/run_nlu_eval.py
python mine/eval/run_pipeline_eval.py
python mine/eval/run_policy_tests.py
```

Evaluation results will be stored in:

```
EvalResults/
```

---

## Requirements

- Python 3.10+
- 8GB+ RAM (minimum)
- GPU recommended for large models (Llama 3.1)

---

## Notes

- The system relies on local LLM inference via HuggingFace Transformers  
- Performance and latency depend on the selected model  
- Smaller models can be used for testing, but may reduce accuracy  

## Limitations

- Requires local LLM inference (no lightweight demo mode)  
- Dialogue Manager accuracy depends on policy correction  
- Dataset is relatively small and domain-specific  
- Prototype system (not production-ready)  

---

## Future Work

- Improved clarification strategies for underspecified inputs  
- Increased robustness of Dialogue Management  
- Better transparency and explainability of system decisions  
- Improved reproducibility and packaging  
