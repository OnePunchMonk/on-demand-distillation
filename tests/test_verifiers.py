from harness.verifiers import (
    extract_code_block,
    run_structural_check,
    verify_json_schema,
    verify_python_runs,
    verify_tool_call,
)


def test_extract_code_block():
    text = "here you go:\n```python\nprint('hi')\n```"
    assert extract_code_block(text).strip() == "print('hi')"


def test_extract_code_block_missing():
    assert extract_code_block("no code here") is None


def test_verify_python_runs_success():
    text = "```python\nx = 1 + 1\n```"
    assert verify_python_runs(text) is True


def test_verify_python_runs_failure():
    text = "```python\nraise ValueError('boom')\n```"
    assert verify_python_runs(text) is False


def test_verify_python_runs_none_when_no_code():
    assert verify_python_runs("just text, no code block") is None


def test_verify_tool_call_ok():
    assert verify_tool_call({"ok": True}) is True
    assert verify_tool_call({"ok": False, "error": "x"}) is False
    assert verify_tool_call(None) is None


def test_verify_json_schema_none_without_schema():
    assert verify_json_schema('{"a": 1}') is None


def test_run_structural_check_dispatches_by_task_class():
    assert run_structural_check("code", "```python\n1+1\n```") is True
    assert run_structural_check("default", "no verifier for this class") is None
