"""Scheduled post-training cycle: pulls the escalation log, builds the
dataset, trains, evals, and (if it passes) promotes a checkpoint. Runs as a
Modal scheduled function sharing a volume with the harness's log.

Deploy: modal deploy modal_app/train_app.py
Manual trigger: modal run modal_app/train_app.py::run_cycle
"""
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

    sys.path.insert(0, "/app")

    subprocess.run(
        ["python", "-m", "distillation.build_dataset", "--config", "/app/config.yaml", "--out-dir", "/data/dataset"],
        check=True,
    )

    from pathlib import Path
    if not Path("/data/dataset/sft_train.jsonl").exists():
        print("no dataset produced this cycle (likely below min_batch_size) — skipping training")
        return

    subprocess.run(
        [
            "python", "-m", "training.sft",
            "--dataset", "/data/dataset/sft_train.jsonl",
            "--config", "/app/config.yaml",
        ],
        check=True,
    )

    volume.commit()
    print("cycle complete: new checkpoint written to volume, gate promotion manually via training/eval.py")
