import os
from typing import Optional

import pytest

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from anthropic import Anthropic
from openai import OpenAI


def _key(env_var: str) -> Optional[str]:
    return os.environ.get(env_var) or None


requires_anthropic = pytest.mark.skipif(
    _key("ANTHROPIC_API_KEY") is None,
    reason="ANTHROPIC_API_KEY env var not set",
)

requires_openai = pytest.mark.skipif(
    _key("OPENAI_API_KEY") is None,
    reason="OPENAI_API_KEY env var not set",
)

requires_deepseek = pytest.mark.skipif(
    _key("DEEPSEEK_API_KEY") is None,
    reason="DEEPSEEK_API_KEY env var not set",
)

requires_all_providers = pytest.mark.skipif(
    any(_key(k) is None for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY")),
    reason="All three API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY) must be set",
)

TASK_PROMPT = "In one sentence, what is the capital of France?"
JUDGE_PROMPT = "Pick the response that most concisely and accurately names the capital city."


@pytest.fixture(scope="session")
def anthropic_client():
    key = _key("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return Anthropic(api_key=key)


@pytest.fixture(scope="session")
def openai_client():
    key = _key("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)


@pytest.fixture(scope="session")
def deepseek_client():
    key = _key("DEEPSEEK_API_KEY")
    if not key:
        pytest.skip("DEEPSEEK_API_KEY not set")
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")
