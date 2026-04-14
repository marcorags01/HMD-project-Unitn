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
├── nlu.py              # Meaning Representation extraction
├── dm.py               # Dialogue Manager
├── policy.py           # Rule-based control layer
├── nlg.py              # Response generation
├── support_fn.py       # Domain logic (menus, swaps, repairs)
├── main.py             # Dialogue orchestration
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
[Download PDF](./MarcoRagusa_HMDreport.pdf)

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

### Install dependencies

```
pip install -r requirements.txt
```

### Run the system

```
python mine/main.py
```

### Run evaluation

```
python mine/eval/run_nlu_eval.py
python mine/eval/run_pipeline_eval.py
python mine/eval/run_policy_tests.py
```

---

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
