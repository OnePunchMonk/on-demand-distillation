from harness.gate import BudgetTracker, EscalationGate, SignalType, StudentAttempt


def test_no_escalation_by_default():
    gate = EscalationGate()
    attempt = StudentAttempt(output="answer")
    decision = gate.decide(attempt)
    assert decision.escalate is False
    assert decision.signal == SignalType.NONE


def test_structural_failure_escalates():
    gate = EscalationGate()
    attempt = StudentAttempt(output="broken code", structural_ok=False)
    decision = gate.decide(attempt)
    assert decision.escalate is True
    assert decision.signal == SignalType.STRUCTURAL_FAILURE


def test_low_self_reported_confidence_escalates():
    gate = EscalationGate(self_reported_confidence_threshold=0.6)
    attempt = StudentAttempt(output="answer", self_reported_confidence=0.3)
    decision = gate.decide(attempt)
    assert decision.escalate is True
    assert decision.signal == SignalType.SELF_REPORTED


def test_high_self_reported_confidence_does_not_escalate():
    gate = EscalationGate(self_reported_confidence_threshold=0.6)
    attempt = StudentAttempt(output="answer", self_reported_confidence=0.9)
    decision = gate.decide(attempt)
    assert decision.escalate is False


def test_sampling_disagreement_escalates():
    gate = EscalationGate()
    attempt = StudentAttempt(output="answer", samples_agree=False)
    decision = gate.decide(attempt)
    assert decision.escalate is True
    assert decision.signal == SignalType.SAMPLING_DISAGREEMENT


def test_exhausted_budget_blocks_escalation():
    budget = BudgetTracker(session_budget=1, task_class_budget={"default": 10})
    budget.consume("default")
    gate = EscalationGate(budget=budget)
    attempt = StudentAttempt(output="broken", structural_ok=False)
    decision = gate.decide(attempt)
    assert decision.escalate is False


def test_task_class_budget_isolated_per_class():
    budget = BudgetTracker(session_budget=10, task_class_budget={"a": 1, "b": 5})
    budget.consume("a")
    assert budget.remaining("a") == 0
    assert budget.remaining("b") == 5
