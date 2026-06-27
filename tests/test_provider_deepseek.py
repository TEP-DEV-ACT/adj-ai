from adj_ai import Orchestrator
from .conftest import requires_deepseek, TASK_PROMPT, JUDGE_PROMPT


@requires_deepseek
def test_deepseek_single_candidate(deepseek_client):
    orch = Orchestrator(
        task_prompt=TASK_PROMPT,
        judge_prompt=JUDGE_PROMPT,
        deepseek_client=deepseek_client,
        candidate_number=1,
        max_task_tokens=100,
    )
    result = orch.run_task()
    assert isinstance(result, str)
    assert "paris" in result.lower()


@requires_deepseek
def test_deepseek_multi_candidate(deepseek_client):
    orch = Orchestrator(
        task_prompt=TASK_PROMPT,
        judge_prompt=JUDGE_PROMPT,
        deepseek_client=deepseek_client,
        candidate_number=2,
        max_task_tokens=100,
    )
    result = orch.run_task()
    assert isinstance(result, str)
    assert "paris" in result.lower()
