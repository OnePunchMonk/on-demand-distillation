"""SFT on the accumulated escalation set, mixed with a replay slice of the
original instruction-tuning data to prevent regression.

LoRA by default for fast iteration; set training.method=full in config.yaml
to fold to a full fine-tune once a batch has proven itself on eval.

Run on Modal via modal_app/train_app.py, or locally with a GPU:
    python -m training.sft --dataset data/dataset/sft_train.jsonl --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def mix_with_replay(new_examples: list[dict], replay_examples: list[dict], replay_fraction: float) -> list[dict]:
    """Blend new escalation-derived examples with a replay slice of the
    original instruction data so the model doesn't regress on tasks it
    already handled fine."""
    if not replay_examples or replay_fraction <= 0:
        return new_examples
    n_replay = int(len(new_examples) * replay_fraction / (1 - replay_fraction))
    n_replay = min(n_replay, len(replay_examples))
    mixed = new_examples + random.sample(replay_examples, n_replay)
    random.shuffle(mixed)
    return mixed


def run_sft(cfg: dict, dataset_path: str, replay_path: str | None) -> str:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTConfig, SFTTrainer

    t = cfg["training"]
    examples = load_jsonl(dataset_path)
    replay = load_jsonl(replay_path) if replay_path and Path(replay_path).exists() else []
    examples = mix_with_replay(examples, replay, t["replay_fraction"])

    tokenizer = AutoTokenizer.from_pretrained(t["base_checkpoint"])

    def to_text(ex: dict) -> dict:
        return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}

    dataset = Dataset.from_list([to_text(e) for e in examples])

    model = AutoModelForCausalLM.from_pretrained(t["base_checkpoint"], device_map="auto")

    peft_config = None
    if t["method"] == "lora":
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )

    output_dir = str(Path(t["output_dir"]) / "sft-latest")
    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_steps=10,
        save_strategy="epoch",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--replay-dataset", default=None)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = run_sft(cfg, args.dataset, args.replay_dataset)
    print(f"saved checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
