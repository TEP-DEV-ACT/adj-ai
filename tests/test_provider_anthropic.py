import pytest
from adj_ai import Orchestrator, OrchestratorError
from .conftest import requires_anthropic, TASK_PROMPT, JUDGE_PROMPT


@requires_anthropic
def test_anthropic_single_candidate(anthropic_client):
    orch = Orchestrator(
        task_prompt=TASK_PROMPT,
        judge_prompt=JUDGE_PROMPT,
        anthropic_client=anthropic_client,
        candidate_number=1,
        max_task_tokens=100,
    )
    result = orch.run_task()
    assert isinstance(result, str)
    assert "paris" in result.lower()


@requires_anthropic
def test_anthropic_multi_candidate(anthropic_client):
    orch = Orchestrator(
        task_prompt=TASK_PROMPT,
        judge_prompt=JUDGE_PROMPT,
        anthropic_client=anthropic_client,
        candidate_number=2,
        max_task_tokens=100,
    )
    result = orch.run_task()
    assert isinstance(result, str)
    assert "paris" in result.lower()


def test_no_client_raises_error():
    with pytest.raises(OrchestratorError):
        Orchestrator(
            task_prompt=TASK_PROMPT,
            judge_prompt=JUDGE_PROMPT,
        )
