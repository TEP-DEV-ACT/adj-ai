from adj_ai import Orchestrator
from .conftest import requires_openai, TASK_PROMPT, JUDGE_PROMPT


@requires_openai
def test_openai_single_candidate(openai_client):
    orch = Orchestrator(
        task_prompt=TASK_PROMPT,
        judge_prompt=JUDGE_PROMPT,
        openai_client=openai_client,
        candidate_number=1,
        max_task_tokens=100,
    )
    result = orch.run_task()
    assert isinstance(result, str)
    assert "paris" in result.lower()


@requires_openai
def test_openai_multi_candidate(openai_client):
    orch = Orchestrator(
        task_prompt=TASK_PROMPT,
        judge_prompt=JUDGE_PROMPT,
        openai_client=openai_client,
        candidate_number=2,
        max_task_tokens=100,
    )
    result = orch.run_task()
    assert isinstance(result, str)
    assert "paris" in result.lower()
