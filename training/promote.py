"""Promotion: the seam between a trained checkpoint and the serving student.

A checkpoint only becomes what `modal_app/student_app.py` loads if
`training/eval.py` says it doesn't regress on every prior cycle's holdout.
Promotion means: copy the checkpoint into the serving slot and update a
pointer file the student endpoint re-reads on its next cold start.

Layout under the shared volume root (e.g. /data on Modal):
    serving/current_checkpoint.txt   -> path to the LoRA adapter dir (or "base")
    serving/<cycle_id>/              -> promoted adapter weights for that cycle
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def current_checkpoint_path(volume_root: str) -> str | None:
    pointer = Path(volume_root) / "serving" / "current_checkpoint.txt"
    if not pointer.exists():
        return None
    value = pointer.read_text().strip()
    return None if value in ("", "base") else value


def promote(volume_root: str, cycle_id: str, candidate_checkpoint_dir: str) -> str:
    """Copies the candidate checkpoint into the serving slot for this cycle
    and repoints current_checkpoint.txt at it. Returns the new served path."""
    serving_dir = Path(volume_root) / "serving"
    serving_dir.mkdir(parents=True, exist_ok=True)

    dest = serving_dir / cycle_id
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(candidate_checkpoint_dir, dest)

    (serving_dir / "current_checkpoint.txt").write_text(str(dest))

    history_path = serving_dir / "promotion_history.jsonl"
    with history_path.open("a") as f:
        f.write(json.dumps({"cycle_id": cycle_id, "checkpoint": str(dest)}) + "\n")

    return str(dest)


def rollback(volume_root: str) -> str | None:
    """Repoints current_checkpoint.txt at the previous entry in promotion
    history. Returns the path rolled back to, or None if there's nothing to
    roll back to (stays on base)."""
    serving_dir = Path(volume_root) / "serving"
    history_path = serving_dir / "promotion_history.jsonl"
    if not history_path.exists():
        return None

    entries = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    if len(entries) < 2:
        (serving_dir / "current_checkpoint.txt").write_text("base")
        return None

    previous = entries[-2]["checkpoint"]
    (serving_dir / "current_checkpoint.txt").write_text(previous)
    return previous


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_promote = sub.add_parser("promote")
    p_promote.add_argument("--volume-root", default="/data")
    p_promote.add_argument("--cycle-id", required=True)
    p_promote.add_argument("--checkpoint-dir", required=True)

    p_rollback = sub.add_parser("rollback")
    p_rollback.add_argument("--volume-root", default="/data")

    args = parser.parse_args()
    if args.command == "promote":
        path = promote(args.volume_root, args.cycle_id, args.checkpoint_dir)
        print(f"promoted {args.cycle_id} -> {path}")
    elif args.command == "rollback":
        path = rollback(args.volume_root)
        print(f"rolled back to {path or 'base'}")


if __name__ == "__main__":
    main()
