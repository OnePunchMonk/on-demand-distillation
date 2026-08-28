"""Escalation gate: decides whether a student attempt should go to the teacher.

Three signal types from the design doc, in order of how much they lean on
the student's own self-assessment:

- structural:  a checkable output failed (code didn't run, schema invalid,
               tool call errored) -> escalate unconditionally.
- self_report: student emitted a confidence score / "not sure" flag as part
               of its output -> escalate if below threshold.
- disagreement: two samples at moderate temperature disagree -> escalate.

The gate itself is a plain threshold table, not a learned policy, by design:
the escalation log is what tells you later whether thresholds are off.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SignalType(str, Enum):
    STRUCTURAL_FAILURE = "structural_failure"
    SELF_REPORTED = "self_reported"
    SAMPLING_DISAGREEMENT = "sampling_disagreement"
    NONE = "none"


@dataclass
class StudentAttempt:
    output: str
    task_class: str = "default"
    structural_ok: bool | None = None       # None = not a structurally-checkable task
    self_reported_confidence: float | None = None
    samples_agree: bool | None = None


@dataclass
class EscalationDecision:
    escalate: bool
    signal: SignalType
    reason: str


@dataclass
class BudgetTracker:
    session_budget: int
    task_class_budget: dict[str, int] = field(default_factory=dict)
    default_task_class_budget: int = 200
    _session_used: int = 0
    _per_class_used: dict[str, int] = field(default_factory=dict)

    def remaining(self, task_class: str) -> int:
        class_cap = self.task_class_budget.get(task_class, self.default_task_class_budget)
        used = self._per_class_used.get(task_class, 0)
        return min(self.session_budget - self._session_used, class_cap - used)

    def consume(self, task_class: str) -> None:
        self._session_used += 1
        self._per_class_used[task_class] = self._per_class_used.get(task_class, 0) + 1


class EscalationGate:
    def __init__(
        self,
        self_reported_confidence_threshold: float = 0.6,
        budget: BudgetTracker | None = None,
    ) -> None:
        self.self_reported_confidence_threshold = self_reported_confidence_threshold
        self.budget = budget

    def decide(self, attempt: StudentAttempt) -> EscalationDecision:
        if self.budget is not None and self.budget.remaining(attempt.task_class) <= 0:
            return EscalationDecision(False, SignalType.NONE, "budget exhausted, forced no-escalate")

        if attempt.structural_ok is False:
            return EscalationDecision(True, SignalType.STRUCTURAL_FAILURE, "structural check failed")

        if attempt.samples_agree is False:
            return EscalationDecision(True, SignalType.SAMPLING_DISAGREEMENT, "samples disagreed")

        if (
            attempt.self_reported_confidence is not None
            and attempt.self_reported_confidence < self.self_reported_confidence_threshold
        ):
            return EscalationDecision(
                True,
                SignalType.SELF_REPORTED,
                f"self-reported confidence {attempt.self_reported_confidence:.2f} "
                f"< threshold {self.self_reported_confidence_threshold:.2f}",
            )

        return EscalationDecision(False, SignalType.NONE, "no escalation signal")
