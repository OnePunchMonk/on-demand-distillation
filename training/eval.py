"""Regression eval: a new checkpoint only replaces the serving model if it
doesn't regress on the held-out slices from every prior cycle, not just the
newest one.

Held-out files are expected at data/holdouts/<cycle_id>.jsonl, one per past
cycle, each a list of {"messages": [...]} SFT-format examples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def score_checkpoint(model_path: str, holdout_examples: list[dict]) -> float:
    """Exact-match rate of greedy generation against the recorded target.
    A stand-in scorer: swap in a task-specific verifier/judge for real use."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    correct = 0
    for ex in holdout_examples:
        user_msg = ex["messages"][0]["content"]
        target = ex["messages"][1]["content"]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if generated.strip() == target.strip():
            correct += 1
    return correct / len(holdout_examples) if holdout_examples else 1.0


def evaluate_all_cycles(model_path: str, holdout_dir: str) -> dict[str, float]:
    scores = {}
    for path in sorted(Path(holdout_dir).glob("*.jsonl")):
        examples = load_jsonl(path)
        scores[path.stem] = score_checkpoint(model_path, examples)
    return scores


def gate_promotion(new_scores: dict[str, float], baseline_scores: dict[str, float], tolerance: float = 0.02) -> bool:
    """A checkpoint is promoted only if it doesn't regress (beyond `tolerance`)
    on any cycle's held-out slice compared to the currently-served model."""
    for cycle_id, baseline in baseline_scores.items():
        new_score = new_scores.get(cycle_id)
        if new_score is None:
            continue
        if new_score < baseline - tolerance:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--baseline-model-path", required=True)
    parser.add_argument("--holdout-dir", default="data/holdouts")
    args = parser.parse_args()

    new_scores = evaluate_all_cycles(args.model_path, args.holdout_dir)
    baseline_scores = evaluate_all_cycles(args.baseline_model_path, args.holdout_dir)

    promote = gate_promotion(new_scores, baseline_scores)
    print(json.dumps({"new_scores": new_scores, "baseline_scores": baseline_scores, "promote": promote}, indent=2))


if __name__ == "__main__":
    main()
