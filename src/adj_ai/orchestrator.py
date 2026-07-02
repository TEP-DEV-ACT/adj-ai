from dataclasses import dataclass, field
from typing import Optional

from anthropic import Anthropic
from openai import OpenAI

from .exceptions import OrchestratorError
from .provider import Provider, ProviderType


@dataclass
class Orchestrator:
    """Generate candidate responses from one or more LLM providers and pick a winner.

    At least one provider client (Anthropic, OpenAI, or Deepseek) must be supplied.
    Each supplied client is wrapped in a `Provider`, which encapsulates that
    provider's API shape and its model-selection/judging rules.
    """

    task_prompt: str
    judge_prompt: str
    anthropic_client: Optional[Anthropic] = None
    openai_client: Optional[OpenAI] = None
    deepseek_client: Optional[OpenAI] = None
    candidate_number: int = 1
    max_task_tokens: int = 1000

    # Built during __post_init__ from the supplied clients; not a constructor argument.
    # Ordered by judging preference: Anthropic > OpenAI > Deepseek.
    providers: list[Provider] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.anthropic_client is not None:
            self.providers.append(self._build_anthropic(self.anthropic_client))
        if self.openai_client is not None:
            self.providers.append(self._build_openai(self.openai_client))
        if self.deepseek_client is not None:
            self.providers.append(self._build_deepseek(self.deepseek_client))

        if not self.providers:
            raise OrchestratorError(
                "At least one client of Anthropic, OpenAI, or Deepseek must be provided."
            )

    @staticmethod
    def _build_anthropic(client: Anthropic) -> Provider:
        return Provider(
            provider_type=ProviderType.ANTHROPIC,
            label="Anthropic",
            client=client,
            analyser_filter=lambda models: [m for m in models if "haiku" in m.lower()],
            judge_filter=lambda models: [m for m in models if "sonnet" in m.lower()],
            judge_max_tokens=100,
            analyser_strict=True,
            judge_strict=True,
        )

    @staticmethod
    def _build_openai(client: OpenAI) -> Provider:
        return Provider(
            provider_type=ProviderType.OPENAI_STYLE,
            label="OpenAI",
            client=client,
            # Restrict to chat-compatible models (gpt-* and o-series); fall back to all.
            models_filter=lambda models: [m for m in models if m.startswith("gpt-") or m.startswith("o")] or models,
            analyser_filter=lambda models: [m for m in models if "mini" in m.lower()],
            judge_filter=lambda models: [m for m in models if "gpt-4" in m.lower() and "mini" not in m.lower()],
            judge_max_tokens=100,
            analyser_strict=True,
            judge_strict=True,
        )

    @staticmethod
    def _build_deepseek(client: OpenAI) -> Provider:
        return Provider(
            provider_type=ProviderType.OPENAI_STYLE,
            label="Deepseek",
            client=client,
            analyser_filter=lambda models: [m for m in models if "chat" in m.lower() or "flash" in m.lower()],
            judge_filter=lambda models: [m for m in models if "reasoner" in m.lower() or "pro" in m.lower()],
            judge_max_tokens=1024,
            analyser_strict=False,
            judge_strict=False,
        )

    def run_task(self) -> Optional[str]:
        """Select models, generate candidates, and return the winning response."""
        # Select the model each provider will use before generating candidates.
        for provider in self.providers:
            provider.select_model(self.task_prompt)

        # Generate candidates from every provider.
        candidates: list[str] = []
        for provider in self.providers:
            candidates += provider.generate_candidates(
                self.task_prompt, self.candidate_number, self.max_task_tokens
            )

        # Judge the candidates only if more than one was requested per provider.
        if self.candidate_number > 1:
            return self._judge_candidates(candidates)
        return candidates[0] if candidates else None

    def _judge_candidates(self, candidates: list[str]) -> str:
        """Rank candidates using the highest-preference available provider."""
        if not candidates:
            raise OrchestratorError("No candidates were provided to judge.")

        # Providers are ordered by preference (Anthropic > OpenAI > Deepseek),
        # so the first one judges.
        judged_candidates = self.providers[0].judge(self.judge_prompt, candidates)
        if not judged_candidates:
            raise OrchestratorError("No candidates were judged successfully.")

        return judged_candidates[0]
