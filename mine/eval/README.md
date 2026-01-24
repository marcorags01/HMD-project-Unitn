# Meal Kit Composer — Automatic Evaluation Pack (Add-on)

This folder is designed to be dropped next to your existing project files **without modifying any of them**.

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

## Quickstart

From your project root (same directory as `main.py`):

### NLU evaluation

```bash
python -m eval.run_nlu_eval \
  --model qwen3 \
  --prompt eval/prompts/nlu_v1_concise.txt \
  --data eval/data/nlu_synth.jsonl \
  --out eval_outputs/nlu_report.json
```

Try additional prompts:
- `eval/prompts/nlu_v2_strict_context.txt`
- `eval/prompts/nlu_v3_prod_aligned.txt` (adds menu_id + yes/no swap awaiting-slot behavior)

Try the scalable string-injection dataset (recommended):

```bash
python -m eval.run_nlu_eval \
  --model qwen3 \
  --prompt eval/prompts/nlu_v3_prod_aligned.txt \
  --data eval/data/nlu_injection_v1.jsonl \
  --out eval_outputs/nlu_report_injection.json
```

### Policy unit tests

```bash
python -m eval.run_policy_tests
```

### Pipeline (oracle) evaluation

```bash
python -m eval.run_pipeline_eval \
  --scenarios eval/scenarios/scenarios.json \
  --recipes recipes_30.json \
  --out eval_outputs/pipeline_report.json
```

You can also run a multi-turn scenario set that is schema-aligned with the pipeline runner:

```bash
python -m eval.run_pipeline_eval \
  --scenarios eval/scenarios/scenarios_v1.json \
  --recipes recipes_30.json \
  --out eval_outputs/pipeline_report_v1.json
```

## Extending the datasets

- Add more NLU test cases to `eval/data/nlu_synth.jsonl`.
- Or regenerate a larger injection dataset with:

```bash
python -m eval.generate_nlu_injection --out eval/data/nlu_injection_v1.jsonl --max 3000 --seed 7 --shuffle
```
- Add more end-to-end scenarios to `eval/scenarios/scenarios.json`.

Recommended strategy (per course instructions):
1. Start with **template-based generation** for coverage (slot/value combinations).
2. Use an LLM to **generate paraphrases** or **new templates**, validate them, then inject values.

## Notes on fairness and comparability

- The `eval.nlu_runner` is *prompt-only* (it does not apply the extra deterministic repair logic found in your production `nlu.py`).
  This is deliberate: it isolates prompt/model behavior.
- If you want to evaluate the full production NLU module, you can add a second benchmark script that instantiates `nlu.NLU` directly.
  Keep those results separate from prompt-only results.
