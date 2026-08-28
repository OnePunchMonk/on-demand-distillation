"""Assigns a task_class to an input so the gate's per-class budgets and the
dataset builder's per-class rebalancing have something to key on.

Keyword/pattern based on purpose: the design doc explicitly wants task
classes to be a human-owned taxonomy ("who defines task classes" is one of
the open questions), not a learned model. This is the seam where that human
input plugs in — extend TASK_CLASS_PATTERNS, don't replace this with a
classifier model.
"""
from __future__ import annotations

import re

TASK_CLASS_PATTERNS: dict[str, list[str]] = {
    "code": [r"\bcode\b", r"\bfunction\b", r"\bbug\b", r"\bcompile\b", r"\bstack trace\b", r"```"],
    "math": [r"\bsolve\b", r"\bequation\b", r"\bcalculate\b", r"\bproof\b", r"\bderivative\b"],
    "extraction": [r"\bextract\b", r"\bparse\b", r"\bschema\b", r"\bjson\b"],
}
DEFAULT_TASK_CLASS = "default"


def classify(task_input: str) -> str:
    lowered = task_input.lower()
    for task_class, patterns in TASK_CLASS_PATTERNS.items():
        if any(re.search(p, lowered) for p in patterns):
            return task_class
    return DEFAULT_TASK_CLASS
