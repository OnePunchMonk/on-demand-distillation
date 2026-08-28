"""Optional preference stage: (student_attempt, teacher_output) as
(rejected, chosen) directly from the escalation log. Close to free since the
pairs already exist — targets the exact failure the student exhibited.

    python -m training.dpo --dataset data/dataset/dpo_train.jsonl --model-path checkpoints/sft-latest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_dpo(cfg: dict, dataset_path: str, model_path: str) -> str:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    examples = load_jsonl(dataset_path)
    dataset = Dataset.from_list(examples)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    output_dir = str(Path(cfg["training"]["output_dir"]) / "dpo-latest")
    dpo_config = DPOConfig(
        output_dir=output_dir,
        beta=cfg["training"]["dpo"]["beta"],
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=5e-6,
        logging_steps=10,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = run_dpo(cfg, args.dataset, args.model_path)
    print(f"saved checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
