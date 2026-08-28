from distillation.build_dataset import (
    filter_records,
    format_dpo_example,
    format_sft_example,
    rebalance_records,
    split_holdout,
)


def make_record(input_, student, teacher, task_class="default", verdict=None):
    return {
        "input": input_,
        "student_attempt": student,
        "teacher_output": teacher,
        "task_class": task_class,
        "verifier_verdict": verdict,
    }


def test_filter_drops_failed_verification():
    records = [make_record("q", "a", "b", verdict="fail"), make_record("q2", "a", "b", verdict="pass")]
    filtered = filter_records(records, near_identical_threshold=0.97)
    assert len(filtered) == 1
    assert filtered[0]["input"] == "q2"


def test_filter_drops_near_identical_outputs():
    records = [make_record("q", "same answer text", "same answer text")]
    filtered = filter_records(records, near_identical_threshold=0.97)
    assert filtered == []


def test_rebalance_caps_per_class():
    records = [make_record(f"q{i}", "a", "b", task_class="c") for i in range(10)]
    rebalanced = rebalance_records(records, max_per_class=3)
    assert len(rebalanced) == 3


def test_format_sft_example_targets_teacher_output():
    record = make_record("question", "wrong", "right")
    ex = format_sft_example(record)
    assert ex["messages"][0]["content"] == "question"
    assert ex["messages"][1]["content"] == "right"


def test_format_dpo_example_teacher_is_chosen():
    record = make_record("question", "wrong", "right")
    ex = format_dpo_example(record)
    assert ex["chosen"] == "right"
    assert ex["rejected"] == "wrong"


def test_split_holdout_reserves_per_class():
    examples = [{"task_class": "a"} for _ in range(10)] + [{"task_class": "b"} for _ in range(10)]
    train, holdout = split_holdout(examples, holdout_fraction=0.2)
    assert len(train) + len(holdout) == 20
    assert len(holdout) >= 2
