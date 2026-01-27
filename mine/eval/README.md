# Meal Kit Composer — Automatic Evaluation Pack

## What this pack evaluates

### 1) NLU (prompt-based)
- **Intent accuracy**
- **Slot micro-F1** over (slot,value) pairs
- **Exact match** (intent + all slots)

You can benchmark **different prompts** and **different models** (llama3/llama31/qwen3) as required by your assignment.

### 2) DM/Policy (deterministic unit tests)
- Sanity-check the **policy guardrails** (slot-filling gate, menu selection gate, swap confirmation handling).

### 3) End-to-end (oracle) pipeline
- Runs the pipeline using **gold MRs** provided in the scenario file (no NLU/DM model needed).
- Validates that the workflow reaches expected phases and invariants.






