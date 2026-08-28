"""Turn the escalation log into an SFT (and optionally DPO) training set.

Pipeline, matching the design doc:
  filter -> dedupe -> rebalance -> format -> holdout

Run standalone:
    python -m distillation.build_dataset --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml


def load_records(log_path: str) -> list[dict]:
    path = Path(log_path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def filter_records(records: list[dict], near_identical_threshold: float) -> list[dict]:
    """Drop escalations where the teacher also failed verification, or where
    student and teacher outputs are near-identical (nothing to learn)."""
    from difflib import SequenceMatcher

    kept = []
    for r in records:
        if r.get("verifier_verdict") == "fail":
            continue
        similarity = SequenceMatcher(None, r["student_attempt"], r["teacher_output"]).ratio()
        if similarity >= near_identical_threshold:
            continue
        kept.append(r)
    return kept


def dedupe_records(records: list[dict], embedding_model: str, threshold: float) -> list[dict]:
    """Embedding-similarity dedup against each other (and, in a real
    deployment, against the current train set too)."""
    if not records:
        return records
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        # sentence-transformers not installed in this environment: skip dedup
        # rather than fail the whole build.
        return records

    model = SentenceTransformer(embedding_model)
    texts = [r["input"] for r in records]
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

    keep_mask = [True] * len(records)
    for i in range(len(records)):
        if not keep_mask[i]:
            continue
        sims = util.cos_sim(embeddings[i], embeddings[i + 1 :])[0]
        for offset, sim in enumerate(sims):
            if sim.item() >= threshold:
                keep_mask[i + 1 + offset] = False
    return [r for r, keep in zip(records, keep_mask) if keep]


def rebalance_records(records: list[dict], max_per_class: int) -> list[dict]:
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_class[r.get("task_class", "default")].append(r)

    rebalanced = []
    for task_class, items in by_class.items():
        random.shuffle(items)
        rebalanced.extend(items[:max_per_class])
    return rebalanced


def format_sft_example(record: dict) -> dict:
    """Recast into a chat-format SFT example targeting the teacher output,
    in the student's own chat template (tokenized later, in training/sft.py)."""
    return {
        "messages": [
            {"role": "user", "content": record["input"]},
            {"role": "assistant", "content": record["teacher_output"]},
        ],
        "task_class": record.get("task_class", "default"),
    }


def format_dpo_example(record: dict) -> dict:
    """(student attempt, teacher output) as (rejected, chosen) — close to
    free since the escalation log already contains matched pairs."""
    return {
        "prompt": record["input"],
        "chosen": record["teacher_output"],
        "rejected": record["student_attempt"],
        "task_class": record.get("task_class", "default"),
    }


def split_holdout(examples: list[dict], holdout_fraction: float) -> tuple[list[dict], list[dict]]:
    """Reserve a slice per task_class as eval-only, never trained on."""
    by_class: dict[str, list[dict]] = defaultdict(list)
    for e in examples:
        by_class[e["task_class"]].append(e)

    train, holdout = [], []
    for items in by_class.values():
        random.shuffle(items)
        n_holdout = max(1, int(len(items) * holdout_fraction)) if items else 0
        holdout.extend(items[:n_holdout])
        train.extend(items[n_holdout:])
    return train, holdout


def build(cfg: dict) -> dict:
    d = cfg["distillation"]
    records = load_records(cfg["logging"]["log_path"])

    if len(records) < d["min_batch_size"]:
        return {"status": "skipped", "reason": f"only {len(records)} escalations, need {d['min_batch_size']}"}

    records = filter_records(records, d["near_identical_similarity_threshold"])
    records = dedupe_records(records, d["embedding_model"], d["dedupe_similarity_threshold"])
    records = rebalance_records(records, d["max_examples_per_task_class"])

    sft_examples = [format_sft_example(r) for r in records]
    sft_train, sft_holdout = split_holdout(sft_examples, d["holdout_fraction"])

    result = {
        "status": "ok",
        "n_input_records": len(records),
        "sft_train": sft_train,
        "sft_holdout": sft_holdout,
    }

    if cfg["training"]["dpo"]["enabled"]:
        dpo_examples = [format_dpo_example(r) for r in records]
        dpo_train, dpo_holdout = split_holdout(dpo_examples, d["holdout_fraction"])
        result["dpo_train"] = dpo_train
        result["dpo_holdout"] = dpo_holdout

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out-dir", default="data/dataset")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    result = build(cfg)
    if result["status"] == "skipped":
        print(f"skipped: {result['reason']}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in ("sft_train", "sft_holdout", "dpo_train", "dpo_holdout"):
        if key not in result:
            continue
        with (out_dir / f"{key}.jsonl").open("w") as f:
            for ex in result[key]:
                f.write(json.dumps(ex) + "\n")

    print(f"built dataset from {result['n_input_records']} filtered records -> {out_dir}")


if __name__ == "__main__":
    main()
