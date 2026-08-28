"""Append-only JSONL escalation log — the dataset for the next post-training cycle."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EscalationRecord:
    input: str
    student_attempt: str
    teacher_output: str
    task_class: str
    signal: str
    reason: str
    verifier_verdict: str | None = None
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.time()


class EscalationLogger:
    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: EscalationRecord) -> None:
        with self.log_path.open("a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def read_all(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        with self.log_path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
