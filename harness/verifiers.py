"""Structural verifiers: checkable outputs that let the gate escalate on a
hard failure with no model self-assessment needed.

Each verifier takes the student's raw output and returns True (passed),
False (failed), or None (not applicable / couldn't check) — None is treated
by the gate as "no structural signal", falling through to self-reported
confidence or sampling disagreement instead.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def extract_code_block(text: str) -> str | None:
    m = CODE_BLOCK_RE.search(text)
    return m.group(1) if m else None


def verify_python_runs(text: str, timeout_s: float = 5.0) -> bool | None:
    """Runs an extracted python code block in a subprocess. True if it exits
    zero, False if it raises/exits nonzero, None if there's no code block to
    check (not a coding task, or the model didn't answer in code)."""
    code = extract_code_block(text)
    if code is None:
        return None

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name

    try:
        result = subprocess.run(
            [sys.executable, path], capture_output=True, timeout=timeout_s, text=True
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(path).unlink(missing_ok=True)


def verify_json_schema(text: str, schema: dict | None = None) -> bool | None:
    """Validates the (first) JSON object found in text against a JSON schema.
    Returns None if no JSON object is present, or no schema was supplied for
    this task class."""
    if schema is None:
        return None

    try:
        import jsonschema
    except ImportError:
        return None

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return False

    try:
        jsonschema.validate(payload, schema)
        return True
    except jsonschema.ValidationError:
        return False


def verify_tool_call(tool_call_result: dict | None) -> bool | None:
    """For harness integrations that route through actual tool calls: pass
    the tool call's result envelope (e.g. {"ok": True, ...} / {"ok": False,
    "error": ...}) through here rather than inferring pass/fail from text."""
    if tool_call_result is None:
        return None
    return bool(tool_call_result.get("ok"))


TASK_CLASS_VERIFIERS = {
    "code": verify_python_runs,
    "extraction": verify_json_schema,
}


def run_structural_check(task_class: str, output: str, **kwargs) -> bool | None:
    """Dispatches to the verifier registered for this task_class, if any."""
    verifier = TASK_CLASS_VERIFIERS.get(task_class)
    if verifier is None:
        return None
    return verifier(output, **kwargs) if kwargs else verifier(output)
