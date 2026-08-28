from harness.task_classifier import classify


def test_classifies_code():
    assert classify("There's a bug in this function, can you fix the code?") == "code"


def test_classifies_math():
    assert classify("Please solve this equation for x") == "math"


def test_classifies_extraction():
    assert classify("Extract the fields into a json schema") == "extraction"


def test_falls_back_to_default():
    assert classify("What's the capital of France?") == "default"
