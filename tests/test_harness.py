import json
from pathlib import Path

from harness.gate import EscalationGate, StudentAttempt
from harness.harness import DistillationHarness
from harness.logger import EscalationLogger


def test_confident_student_skips_teacher(tmp_path: Path):
    def student_fn(task_input: str) -> StudentAttempt:
        return StudentAttempt(output="confident answer", self_reported_confidence=0.9)

    def teacher_fn(task_input: str) -> str:
        raise AssertionError("teacher should not be called")

    harness = DistillationHarness(
        student_fn, teacher_fn, EscalationGate(), EscalationLogger(tmp_path / "log.jsonl")
    )
    assert harness.handle("do the task") == "confident answer"


def test_escalation_logs_record(tmp_path: Path):
    def student_fn(task_input: str) -> StudentAttempt:
        return StudentAttempt(output="unsure answer", task_class="coding", structural_ok=False)

    def teacher_fn(task_input: str) -> str:
        return "teacher's fixed answer"

    log_path = tmp_path / "log.jsonl"
    harness = DistillationHarness(student_fn, teacher_fn, EscalationGate(), EscalationLogger(log_path))

    result = harness.handle("fix this code")
    assert result == "teacher's fixed answer"

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["student_attempt"] == "unsure answer"
    assert records[0]["teacher_output"] == "teacher's fixed answer"
    assert records[0]["task_class"] == "coding"
    assert records[0]["signal"] == "structural_failure"
