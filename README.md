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
harness/
  gate.py             escalation gate: structural / self-reported / sampling-disagreement signals + budgets
  verifiers.py         structural checkers (python exec, json-schema, tool-call result)
  task_classifier.py   assigns task_class (code/math/extraction/default) — human-owned taxonomy, not learned
  logger.py            append-only JSONL escalation log
  harness.py           wires student -> gate -> teacher -> logger
  run.py                CLI entrypoint calling the deployed Modal endpoints
distillation/
  build_dataset.py     filter -> dedupe -> rebalance -> format -> holdout, writes per-cycle holdout registry
training/
  sft.py                LoRA/full SFT with replay mixing
  dpo.py                optional preference stage: (student_attempt, teacher_output) = (rejected, chosen)
  eval.py               regression eval against every prior cycle's held-out slice
  promote.py            promotes a checkpoint to the serving slot, or rolls back
modal_app/
  student_app.py        always-on student endpoint; loads the promoted LoRA adapter, if any
  teacher_app.py        scale-to-zero teacher endpoint
  train_app.py           scheduled cycle: build dataset -> SFT (+DPO) -> eval -> promote/hold
  common.py              shared vLLM serving helpers
config.yaml             models, thresholds, budgets, task classes, cycle cadence
tests/                  unit tests for the gate, verifiers, classifier, dataset build, and promotion
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
modal deploy modal_app/train_app.py       # installs the weekly cron
modal run harness/run.py --input "..."
modal run modal_app/train_app.py::run_cycle    # manual trigger, same logic as the cron
modal run modal_app/train_app.py::rollback     # escape hatch if a promoted checkpoint regresses in the wild
```

`student_app.py` and `train_app.py` share a Modal volume (`on-demand-distill-data`)
that holds the escalation log, the per-cycle holdout registry, and the
serving pointer (`serving/current_checkpoint.txt`). Because the student
endpoint is kept warm (`min_containers=1`), a promotion doesn't take effect
until the container restarts — redeploy or scale it down after a promotion
if you want it live immediately; otherwise it picks up the new checkpoint
on its next natural cold start.

## Closing the loop

1. Traffic hits `harness/run.py` → student answers, `harness/task_classifier.py`
   assigns a task class, `harness/verifiers.py` runs any structural check for
   that class, and the gate in `harness/gate.py` decides whether to escalate.
2. Escalations go to the teacher and get appended to the JSONL log via
   `harness/logger.py`.
3. `modal_app/train_app.py` runs weekly (or on demand): `distillation/build_dataset.py`
   turns the log into an SFT (+ optional DPO) set and snapshots this cycle's
   holdout into the registry; `training/sft.py` (+ `training/dpo.py`) trains a
   candidate checkpoint; `training/eval.py` scores the candidate and the
   currently-served baseline against *every* prior cycle's holdout;
   `training/promote.py` only flips the serving pointer if the candidate
   doesn't regress.

## Status

End-to-end: escalation signals (structural/self-reported/sampling-disagreement),
task classification, logging, dataset build with per-cycle holdout tracking,
LoRA SFT + optional DPO, regression eval, and checkpoint promotion/rollback
are all implemented and exercised by the test suite (`pytest`, 31 tests).
What's still a placeholder: `training/eval.py`'s scorer is exact-match
against the recorded target — swap in a task-specific verifier or judge
model before relying on it to gate real promotions. Escalation thresholds,
the task-class taxonomy in `harness/task_classifier.py`, and cycle cadence
are tuning knobs meant to be adjusted from what the escalation log shows —
see "Open questions" below.

## Open questions

- Escalation threshold calibration — start conservative and tighten once the
  log shows which signals correlate with real failures?
- Cycle cadence — fixed schedule vs. triggered by escalation-count threshold?
- Teacher cost ceiling — what caps the escalation rate or the model size?
- Eval ownership — who defines task classes and their held-out sets?
