from adj_ai import Orchestrator
from .conftest import requires_all_providers, TASK_PROMPT, JUDGE_PROMPT


@requires_all_providers
def test_all_providers_single_candidate(anthropic_client, openai_client, deepseek_client):
    orch = Orchestrator(
        task_prompt=TASK_PROMPT,
        judge_prompt=JUDGE_PROMPT,
        anthropic_client=anthropic_client,
        openai_client=openai_client,
        deepseek_client=deepseek_client,
        candidate_number=1,
        max_task_tokens=100,
    )
    result = orch.run_task()
    assert isinstance(result, str)
    assert "paris" in result.lower()


@requires_all_providers
def test_all_providers_multi_candidate(anthropic_client, openai_client, deepseek_client):
    orch = Orchestrator(
        task_prompt=TASK_PROMPT,
        judge_prompt=JUDGE_PROMPT,
        anthropic_client=anthropic_client,
        openai_client=openai_client,
        deepseek_client=deepseek_client,
        candidate_number=2,
        max_task_tokens=100,
    )
    result = orch.run_task()
    assert isinstance(result, str)
    assert "paris" in result.lower()
