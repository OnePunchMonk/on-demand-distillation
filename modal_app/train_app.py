"""Scheduled post-training cycle: pulls the escalation log, builds the
dataset, trains, evals against the currently-served checkpoint, and
promotes only if it doesn't regress on any prior cycle's holdout. Runs as a
Modal scheduled function sharing a volume with the harness's log and the
student's serving slot.

Deploy: modal deploy modal_app/train_app.py
Manual trigger: modal run modal_app/train_app.py::run_cycle
"""
import datetime

import modal

app = modal.App("on-demand-distill-train")

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0", "transformers>=4.45", "trl>=0.11", "peft>=0.13",
        "datasets>=2.20", "sentence-transformers>=3.0", "pyyaml",
    )
    .add_local_dir(".", remote_path="/app")
)

volume = modal.Volume.from_name("on-demand-distill-data", create_if_missing=True)


@app.function(
    image=train_image,
    gpu="A100-80GB:2",
    volumes={"/data": volume},
    timeout=6 * 60 * 60,
    schedule=modal.Cron("0 6 * * 1"),  # weekly, Monday 06:00 UTC — matches config.yaml cycle.cadence
)
def run_cycle():
    import subprocess
    import sys
    from pathlib import Path

    sys.path.insert(0, "/app")
    from training.eval import evaluate_all_cycles, gate_promotion
    from training.promote import current_checkpoint_path, promote

    cycle_id = datetime.date.today().isoformat()

    build = subprocess.run(
        [
            "python", "-m", "distillation.build_dataset",
            "--config", "/app/config.yaml",
            "--out-dir", "/data/dataset",
            "--holdout-dir", "/data/holdouts",
            "--cycle-id", cycle_id,
        ],
        capture_output=True, text=True,
    )
    print(build.stdout)
    if build.returncode != 0 or not Path("/data/dataset/sft_train.jsonl").exists():
        print("no dataset produced this cycle (below min_batch_size, or build failed) — skipping training")
        print(build.stderr)
        volume.commit()
        return

    baseline_checkpoint = current_checkpoint_path("/data") or "Qwen/Qwen2.5-7B-Instruct"

    subprocess.run(
        [
            "python", "-m", "training.sft",
            "--dataset", "/data/dataset/sft_train.jsonl",
            "--config", "/app/config.yaml",
        ],
        check=True,
    )
    candidate_checkpoint = "checkpoints/sft-latest"  # matches training.output_dir/sft-latest in config.yaml

    import yaml
    with open("/app/config.yaml") as f:
        cfg = yaml.safe_load(f)

    if cfg["training"]["dpo"]["enabled"] and Path("/data/dataset/dpo_train.jsonl").exists():
        subprocess.run(
            [
                "python", "-m", "training.dpo",
                "--dataset", "/data/dataset/dpo_train.jsonl",
                "--model-path", candidate_checkpoint,
                "--config", "/app/config.yaml",
            ],
            check=True,
        )
        candidate_checkpoint = "checkpoints/dpo-latest"

    new_scores = evaluate_all_cycles(candidate_checkpoint, "/data/holdouts")
    baseline_scores = evaluate_all_cycles(baseline_checkpoint, "/data/holdouts")
    should_promote = gate_promotion(new_scores, baseline_scores)

    print(f"cycle={cycle_id} new_scores={new_scores} baseline_scores={baseline_scores} promote={should_promote}")

    if should_promote:
        served_path = promote("/data", cycle_id, candidate_checkpoint)
        print(f"promoted cycle {cycle_id} -> {served_path}. Redeploy/restart student_app to pick it up.")
    else:
        print(f"cycle {cycle_id} regressed on at least one prior holdout — not promoted, staying on {baseline_checkpoint}")

    volume.commit()


@app.function(image=train_image, volumes={"/data": volume})
def rollback():
    """Manual escape hatch: modal run modal_app/train_app.py::rollback"""
    import sys

    sys.path.insert(0, "/app")
    from training.promote import rollback as do_rollback

    path = do_rollback("/data")
    volume.commit()
    print(f"rolled back to {path or 'base'}")
