# adj-ai

`adj-ai` is a multi-provider LLM orchestrator. It generates several candidate
responses to a task from one or more model providers (Anthropic, OpenAI,
DeepSeek), then uses a "judge" model to adjudicate the candidates and return the
best one.

The core idea is **analyse-delegate-judge**: by producing multiple candidates and
scoring them against a judging rubric, you get higher-quality outputs than a
single one-shot completion.

> **Status:** Early development. The Anthropic, OpenAI, and DeepSeek
> generation/judging paths are all implemented.

## How it works

1. **Model selection** – A lightweight model (e.g. Anthropic Haiku) inspects the
   task prompt and picks the most appropriate generation model, balancing cost
   and performance.
2. **Candidate generation** – The chosen model produces `candidates` responses to
   the `task_prompt`.
3. **Judging** – A stronger model (e.g. Anthropic Sonnet) ranks the candidates
   using the `judge_prompt` and returns them in order of preference.
4. **Result** – The top-ranked candidate is returned.

When multiple clients are configured, judging falls back in priority order:
**Anthropic → OpenAI → DeepSeek**.

## Installation

```bash
pip install adj-ai
```

This pulls in the official provider SDKs (`anthropic`, `openai`) automatically.

> DeepSeek and OpenAI share the `openai` SDK (DeepSeek exposes an
> OpenAI-compatible API).

## Usage

```python
from anthropic import Anthropic
from openai import OpenAI

from adj_ai import Orchestrator

anthropic_client = Anthropic(api_key="...")
openai_client = OpenAI(api_key="...")
deepseek_client = OpenAI(api_key="...", base_url="https://api.deepseek.com")

orch = Orchestrator(
    task_prompt="Write a concise product description for a smart water bottle.",
    judge_prompt="Pick the description that is clearest and most persuasive.",
    anthropic_client=anthropic_client,
    openai_client=openai_client,
    deepseek_client=deepseek_client,
    candidate_number=5,
    max_task_tokens=1000,
)

best = orch.run_task()
print(best)
```

At least one of `anthropic_client`, `openai_client`, or `deepseek_client` must be
provided, otherwise an `OrchestratorError` is raised.

## Configuration

| Parameter          | Type        | Default | Description                                              |
| ------------------ | ----------- | ------- | -------------------------------------------------------- |
| `task_prompt`      | `str`       | —       | The task to generate candidate responses for.            |
| `judge_prompt`     | `str`       | —       | The rubric used to rank candidates.                      |
| `anthropic_client` | `Anthropic` | —       | Optional Anthropic client.                               |
| `openai_client`    | `OpenAI`    | —       | Optional OpenAI client.                                  |
| `deepseek_client`  | `OpenAI`    | —       | Optional DeepSeek (OpenAI-compatible) client.            |
| `candidate_number` | `int`       | `1`     | Number of candidate responses to generate.               |
| `max_task_tokens`  | `int`       | `1000`  | Maximum tokens per generated candidate.                  |

## TODO

- [x] Package the project for distribution

## License

Released under the [MIT License](LICENSE).
