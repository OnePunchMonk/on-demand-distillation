"""CLI entrypoint: `modal run harness/run.py --input "..."`

Calls the deployed student/teacher Modal apps through their web endpoints.
Student is expected to emit a trailing `[[confidence: 0.xx]]` tag which is
stripped and parsed as the self-reported signal (see gate.py).
"""
from __future__ import annotations

import argparse
import re
import yaml

import modal

from harness.gate import BudgetTracker, EscalationGate, StudentAttempt
from harness.harness import DistillationHarness
from harness.logger import EscalationLogger

CONFIDENCE_RE = re.compile(r"\[\[confidence:\s*([0-9.]+)\s*\]\]\s*$")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_student_output(raw: str, task_class: str) -> StudentAttempt:
    m = CONFIDENCE_RE.search(raw)
    confidence = float(m.group(1)) if m else None
    output = CONFIDENCE_RE.sub("", raw).strip()
    return StudentAttempt(output=output, task_class=task_class, self_reported_confidence=confidence)


def build_harness(cfg: dict) -> DistillationHarness:
    student_app = modal.Function.lookup(cfg["student"]["endpoint_app"], "generate")
    teacher_app = modal.Function.lookup(cfg["teacher"]["endpoint_app"], "generate")

    def student_fn(task_input: str) -> StudentAttempt:
        raw = student_app.remote(task_input, temperature=cfg["student"]["temperature"])
        return parse_student_output(raw, task_class="default")

    def teacher_fn(task_input: str) -> str:
        return teacher_app.remote(task_input, temperature=cfg["teacher"]["temperature"])

    budget = BudgetTracker(
        session_budget=cfg["escalation"]["session_budget"],
        task_class_budget=cfg["escalation"]["task_class_budget"],
    )
    gate = EscalationGate(
        self_reported_confidence_threshold=cfg["escalation"]["self_reported_confidence_threshold"],
        budget=budget,
    )
    logger = EscalationLogger(cfg["logging"]["log_path"])
    return DistillationHarness(student_fn, teacher_fn, gate, logger)


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
