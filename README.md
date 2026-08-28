# On-Demand Distillation Harness

Post-train a small open-weight student, wrap it in a task harness, and call a
larger open-weight teacher only when the student's own outputs signal it
needs one. Distillation becomes a standing capability of the running system,
not a one-time offline pass.

## Model pairing

| Role | Model | Notes |
|---|---|---|
| Student | `Qwen/Qwen2.5-7B-Instruct` | always-on, served warm |
| Teacher | `Qwen/Qwen2.5-72B-Instruct` | scaled-to-zero, invoked only on escalation |

Same family as the student to minimize chat-template / format drift in
distilled data.

## Layout

```
harness/          runtime: student calls, escalation gate, JSONL logging
distillation/     turns the escalation log into an SFT/DPO training set
training/         SFT + optional DPO post-training, regression eval
modal_app/        Modal deployment: student endpoint, teacher endpoint, train job
config.yaml       models, thresholds, budgets, cycle cadence
tests/            unit tests for the gate and the dataset builder
```

## Loops

1. **Inference loop** (per request) — student answers; harness checks a
   confidence/failure signal and returns the answer if it passes.
2. **Escalation loop** (gated) — on failure, harness calls the teacher, logs
   `(input, student_attempt, teacher_output, verdict)`, returns the teacher's
   answer.
3. **Post-training loop** (batched, e.g. weekly or every N escalations) —
   `distillation/build_dataset.py` filters/dedupes/formats the log,
   `training/sft.py` (+ optional `training/dpo.py`) produces a new
   checkpoint, `training/eval.py` gates promotion on no regression against
   held-out slices from every prior cycle.

See `config.yaml` for the escalation gate thresholds and budgets, and the
docstrings in `harness/gate.py` for the three signal types (structural,
self-reported, sampling-disagreement).

## Running on Modal

```
modal deploy modal_app/student_app.py
modal deploy modal_app/teacher_app.py
modal run harness/run.py --input "..."
modal run modal_app/train_app.py   # scheduled / manual post-training cycle
```

## Status

Working skeleton: harness, gate, logging, dataset build, and SFT/DPO training
are implemented and runnable end-to-end on Modal. Escalation thresholds,
task-class taxonomy, and cycle cadence are still tuning knobs — see
"Open questions" below.

## Open questions

- Escalation threshold calibration — start conservative and tighten once the
  log shows which signals correlate with real failures?
- Cycle cadence — fixed schedule vs. triggered by escalation-count threshold?
- Teacher cost ceiling — what caps the escalation rate or the model size?
- Eval ownership — who defines task classes and their held-out sets?
