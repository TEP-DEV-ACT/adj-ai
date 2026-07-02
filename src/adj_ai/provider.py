import re
from dataclasses import dataclass, field
from typing import Callable, Optional
from .exceptions import OrchestratorError
from enum import Enum

SELECT_SYSTEM_TEMPLATE = (
    "Pick the most appropriate model for the task, balancing performance and cost. "
    "Reply with ONLY the exact model id, nothing else.\n"
    "Available models:\n{models}"
)

JUDGE_SYSTEM_TEMPLATE = (
    "You are judging candidate responses against the judge prompt. "
    "Each candidate is labelled with an index like [0], [1], etc. "
    "Reply with ONLY the candidate indices, best first, as a comma-separated "
    "list (e.g. '2,0,1'). Do not include any other text.\n\n"
    "Candidates:\n{numbered}"
)

class ProviderType(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI_STYLE = "openai_style"  # covers both OpenAI and Deepseek clients

@dataclass
class Provider:
    """A single LLM provider: its client, API shape, and selection/judging rules."""
 
    provider_type: ProviderType
    label: str
    client: object  # Anthropic | OpenAI
    analyser_filter: Callable[[list[str]], list[str]]
    judge_filter: Callable[[list[str]], list[str]]
    models_filter: Optional[Callable[[list[str]], list[str]]] = None
    judge_max_tokens: int = 100
    analyser_strict: bool = True   # if no analyser match, raise instead of falling back
    judge_strict: bool = True      # if no judge match, raise instead of falling back
 
    chosen_model: Optional[str] = field(default=None, init=False)
 
    def list_models(self) -> list[str]:
        return [m.id for m in self.client.models.list().data]
 
    def complete(self, model: str, user: str, *, system: Optional[str] = None, max_tokens: int = 1000) -> str:
        """Dispatch on `provider_type` to speak whichever API shape this provider uses."""
        match self.provider_type:
            case ProviderType.ANTHROPIC:
                kwargs = {"model": model, "messages": [{"role": "user", "content": user}], "max_tokens": max_tokens}
                if system:
                    kwargs["system"] = system
                response = self.client.messages.create(**kwargs)
                return response.content[0].text

            case ProviderType.OPENAI_STYLE:
                # OpenAI and Deepseek both speak chat.completions
                messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
                response = self.client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
                return response.choices[0].message.content

            case _:
                raise OrchestratorError(f"Unsupported provider type: {self.provider_type}")
 
    def select_model(self, task_prompt: str) -> str:
        all_models = self.list_models()
        if not all_models:
            raise OrchestratorError(f"No available models found for {self.label}.")
        available_models = self.models_filter(all_models) if self.models_filter else all_models
 
        matches = self.analyser_filter(available_models)
        if matches:
            task_analyser_model = matches[0]
        elif not self.analyser_strict:
            task_analyser_model = available_models[0]
        else:
            raise OrchestratorError(
                f"No analyser models found available for {self.label}, unable to analyse the task."
            )
 
        system = SELECT_SYSTEM_TEMPLATE.format(models="\n".join(available_models))
        chosen_model = self.complete(task_analyser_model, task_prompt, system=system, max_tokens=100).strip()
        if chosen_model not in available_models:
            chosen_model = task_analyser_model
        if not chosen_model:
            raise OrchestratorError(f"No suitable model was selected for {self.label}.")
 
        self.chosen_model = chosen_model
        return chosen_model
 
    def generate_candidates(self, task_prompt: str, count: int, max_tokens: int) -> list[str]:
        return [self.complete(self.chosen_model, task_prompt, max_tokens=max_tokens) for _ in range(count)]
 
    def judge(self, judge_prompt: str, candidates: list[str]) -> list[str]:
        available_models = self.list_models()
        matches = self.judge_filter(available_models)
        if matches:
            judge_model = matches[0]
        elif not self.judge_strict and available_models:
            judge_model = available_models[-1]
        else:
            raise OrchestratorError(f"No judge models found available for {self.label}, unable to judge candidates.")
 
        numbered = "\n\n".join(f"[{i}]\n{c}" for i, c in enumerate(candidates))
        system = JUDGE_SYSTEM_TEMPLATE.format(numbered=numbered)
        text = self.complete(judge_model, judge_prompt, system=system, max_tokens=self.judge_max_tokens)
 
        ranking = re.findall(r"\d+", text)
        judged_candidates: list[str] = []
        seen: set[int] = set()
        for token in ranking:
            idx = int(token)
            if idx not in seen and 0 <= idx < len(candidates):
                seen.add(idx)
                judged_candidates.append(candidates[idx])
        return judged_candidates