"""Runtime shell around the student model.

Wires together: student call -> signal extraction -> escalation gate ->
optional teacher call -> logging -> response.

`student_fn` / `teacher_fn` are injected so this module has no hard
dependency on how models are actually served (Modal, local vLLM, HF
pipeline, ...). See modal_app/ for the Modal-backed implementations.
"""
from __future__ import annotations

from typing import Callable, Optional

from .gate import BudgetTracker, EscalationGate, StudentAttempt
from .logger import EscalationLogger, EscalationRecord

StudentFn = Callable[[str], StudentAttempt]
TeacherFn = Callable[[str], str]
VerifierFn = Optional[Callable[[str, str], str]]


class DistillationHarness:
    def __init__(
        self,
        student_fn: StudentFn,
        teacher_fn: TeacherFn,
        gate: EscalationGate,
        logger: EscalationLogger,
        verifier_fn: VerifierFn = None,
    ) -> None:
        self.student_fn = student_fn
        self.teacher_fn = teacher_fn
        self.gate = gate
        self.logger = logger
        self.verifier_fn = verifier_fn

    def handle(self, task_input: str) -> str:
        attempt = self.student_fn(task_input)
        decision = self.gate.decide(attempt)

        if not decision.escalate:
            return attempt.output

        if isinstance(self.gate.budget, BudgetTracker):
            self.gate.budget.consume(attempt.task_class)

        teacher_output = self.teacher_fn(task_input)
        verdict = self.verifier_fn(task_input, teacher_output) if self.verifier_fn else None

        self.logger.write(
            EscalationRecord(
                input=task_input,
                student_attempt=attempt.output,
                teacher_output=teacher_output,
                task_class=attempt.task_class,
                signal=decision.signal.value,
                reason=decision.reason,
                verifier_verdict=verdict,
            )
        )
        return teacher_output
