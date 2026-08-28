"""CLI entrypoint: `modal run harness/run.py --input "..."`

Calls the deployed student/teacher Modal apps through their web endpoints.
Wires together all three signal types from the design doc:
  - structural: harness/verifiers.py checks against the classified task_class
  - self-reported: student is asked to emit `[[confidence: 0.xx]]`
  - sampling disagreement: for task classes flagged high-stakes in config,
    sample the student twice and compare
"""
from __future__ import annotations

import argparse
import re

import modal
import yaml

from harness.gate import BudgetTracker, EscalationGate, StudentAttempt
from harness.harness import DistillationHarness
from harness.logger import EscalationLogger
from harness.task_classifier import classify
from harness.verifiers import run_structural_check

CONFIDENCE_RE = re.compile(r"\[\[confidence:\s*([0-9.]+)\s*\]\]\s*$")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_confidence(raw: str) -> tuple[str, float | None]:
    m = CONFIDENCE_RE.search(raw)
    confidence = float(m.group(1)) if m else None
    output = CONFIDENCE_RE.sub("", raw).strip()
    return output, confidence


def outputs_disagree(a: str, b: str, similarity_floor: float = 0.6) -> bool:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio() < similarity_floor


def build_harness(cfg: dict) -> DistillationHarness:
    student_app = modal.Function.lookup(cfg["student"]["endpoint_app"], "generate")
    teacher_app = modal.Function.lookup(cfg["teacher"]["endpoint_app"], "generate")
    sampling_cfg = cfg["escalation"]["sampling_disagreement"]

    def student_fn(task_input: str) -> StudentAttempt:
        task_class = classify(task_input)
        raw = student_app.remote(task_input, temperature=cfg["student"]["temperature"])
        output, confidence = parse_confidence(raw)

        structural_ok = run_structural_check(task_class, output)

        samples_agree = None
        if sampling_cfg["enabled"]:
            raw2 = student_app.remote(task_input, temperature=sampling_cfg["temperature"])
            output2, _ = parse_confidence(raw2)
            samples_agree = not outputs_disagree(output, output2)

        return StudentAttempt(
            output=output,
            task_class=task_class,
            structural_ok=structural_ok,
            self_reported_confidence=confidence,
            samples_agree=samples_agree,
        )

    def teacher_fn(task_input: str) -> str:
        return teacher_app.remote(task_input, temperature=cfg["teacher"]["temperature"])

    def verifier_fn(task_input: str, teacher_output: str) -> str:
        task_class = classify(task_input)
        ok = run_structural_check(task_class, teacher_output)
        if ok is None:
            return "unchecked"
        return "pass" if ok else "fail"

    budget = BudgetTracker(
        session_budget=cfg["escalation"]["session_budget"],
        task_class_budget=cfg["escalation"]["task_class_budget"],
        default_task_class_budget=cfg["escalation"]["task_class_budget"].get("default", 200),
    )
    gate = EscalationGate(
        self_reported_confidence_threshold=cfg["escalation"]["self_reported_confidence_threshold"],
        budget=budget,
    )
    logger = EscalationLogger(cfg["logging"]["log_path"])
    return DistillationHarness(student_fn, teacher_fn, gate, logger, verifier_fn=verifier_fn)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    harness = build_harness(cfg)
    print(harness.handle(args.input))


if __name__ == "__main__":
    main()
