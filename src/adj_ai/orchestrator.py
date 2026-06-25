import re
from dataclasses import dataclass, field
from typing import Optional

from anthropic import Anthropic
from openai import OpenAI

from .exceptions import OrchestratorError


@dataclass
class Orchestrator:
    """Generate candidate responses from one or more LLM providers and pick a winner.

    At least one provider client (Anthropic, OpenAI, or Deepseek) must be supplied.
    """

    task_prompt: str
    judge_prompt: str
    anthropic_client: Optional[Anthropic] = None
    openai_client: Optional[OpenAI] = None
    deepseek_client: Optional[OpenAI] = None
    candidate_number: int = 1
    max_task_tokens: int = 1000

    # Internal state populated during model selection; not constructor arguments.
    chosen_anthropic_model: Optional[str] = field(default=None, init=False)
    chosen_openai_model: Optional[str] = field(default=None, init=False)
    chosen_deepseek_model: Optional[str] = field(default=None, init=False)

    def __post_init__(self) -> None:
        has_atleast_one_client = (
            self.anthropic_client is not None
            or self.openai_client is not None
            or self.deepseek_client is not None
        )
        if not has_atleast_one_client:
            raise OrchestratorError(
                "At least one client of Anthropic, OpenAI, or Deepseek must be provided."
            )

    def run_task(self):
        """Select models, generate candidates, and return the winning response."""
        # Select the models to use before generating candidates
        self._model_selection()

        # Generate candidates using the provided clients
        candidates = []
        if self.anthropic_client:
            candidates += self._generate_candidates_anthropic()
        if self.openai_client:
            candidates += self._generate_candidates_openai()
        if self.deepseek_client:
            candidates += self._generate_candidates_deepseek()

        # Judge the candidates if there are multiple candidates
        if self.candidate_number > 1:
            winner_candidate = self._judge_candidates(candidates)
        else:
            winner_candidate = candidates[0] if candidates else None

        return winner_candidate
    
    def _generate_candidates_anthropic(self):
        # Use the Anthropic client to generate candidates
        candidates = [self._get_anthropic_response(i) for i in range(self.candidate_number)]
        return candidates
    
    def _get_anthropic_response(self,i):
        response = self.anthropic_client.messages.create(
            model=self.chosen_anthropic_model,
            messages=[{"role": "user", "content": f"{self.task_prompt}"}],
            max_tokens=self.max_task_tokens
        )
        return response.content[0].text
    
    def _generate_candidates_openai(self):
        # Use the OpenAI client to generate candidates
        candidates = [self._get_openai_response(i) for i in range(self.candidate_number)]
        return candidates

    def _get_openai_response(self, i):
        response = self.openai_client.chat.completions.create(
            model=self.chosen_openai_model,
            messages=[{"role": "user", "content": f"{self.task_prompt}"}],
            max_tokens=self.max_task_tokens
        )
        return response.choices[0].message.content

    def _generate_candidates_deepseek(self):
        # Use the Deepseek client to generate candidates
        candidates = [self._get_deepseek_response(i) for i in range(self.candidate_number)]
        return candidates

    def _get_deepseek_response(self, i):
        response = self.deepseek_client.chat.completions.create(
            model=self.chosen_deepseek_model,
            messages=[{"role": "user", "content": f"{self.task_prompt}"}],
            max_tokens=self.max_task_tokens
        )
        return response.choices[0].message.content
    
    def _judge_candidates(self, candidates: list):
        # Use the judge prompt to evaluate candidates
        winner_candidate = None
        judged_candidates = [] #ordered list of candidates with their scores
        # For each candidate, we can use the judge prompt to evaluate its quality
        # Only judge the candidates once with the preference for judgement as anthropic client > openai client > deepseek client
        if self.anthropic_client:
            judged_candidates = self._judge_candidates_anthropic(candidates)
        elif self.openai_client:
            judged_candidates = self._judge_candidates_openai(candidates)
        elif self.deepseek_client:
            judged_candidates = self._judge_candidates_deepseek(candidates)

        if judged_candidates:
            winner_candidate = judged_candidates[0]  # Assuming the first candidate is the best after judging
        else:
            raise OrchestratorError("No candidates were judged successfully.")

        return winner_candidate
    
    def _model_selection(self):
        # Logic to select the best model based on some criteria
        if self.anthropic_client:
            self._model_selection_anthropic()
        if self.openai_client:
            self._model_selection_openai()
        if self.deepseek_client:
            self._model_selection_deepseek()
        elif not (self.chosen_anthropic_model or self.chosen_openai_model or self.chosen_deepseek_model):
            raise NotImplementedError("Model selection logic is not implemented yet.")  

    def _model_selection_anthropic(self):
        # Logic to select the best model for Anthropic
        available_models = [m.id for m in self.anthropic_client.models.list().data]
        if not available_models:
            raise OrchestratorError("No available models found for Anthropic.")

        haiku_models = [m for m in available_models if 'haiku' in m.lower()]
        if not haiku_models:
            raise OrchestratorError("No haiku models found available for Anthropic, unable to analyse the task.")
        task_analyser_model = haiku_models[0]
        
        analysis = self.anthropic_client.messages.create(
            model = task_analyser_model,
            max_tokens = 100,
            system=(
            "Pick the most appropriate model for the task, balancing performance and cost. "
            "Reply with ONLY the exact model id, nothing else.\n"
            "Available models:\n" + "\n".join(available_models)
            ),
            messages=[{"role": "user", "content": self.task_prompt}],
        )
        chosen_model = analysis.content[0].text.strip()
        if chosen_model not in available_models:
            chosen_model = task_analyser_model

        if not chosen_model:
            raise OrchestratorError("No suitable model was selected for Anthropic.")

        self.chosen_anthropic_model = chosen_model

    def _model_selection_openai(self):
        # Logic to select the best model for OpenAI
        available_models = [m.id for m in self.openai_client.models.list().data]
        if not available_models:
            raise OrchestratorError("No available models found for OpenAI.")

        mini_models = [m for m in available_models if 'mini' in m.lower()]
        if not mini_models:
            raise OrchestratorError("No mini models found available for OpenAI, unable to analyse the task.")
        task_analyser_model = mini_models[0]

        analysis = self.openai_client.chat.completions.create(
            model=task_analyser_model,
            max_tokens=100,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Pick the most appropriate model for the task, balancing performance and cost. "
                        "Reply with ONLY the exact model id, nothing else.\n"
                        "Available models:\n" + "\n".join(available_models)
                    ),
                },
                {"role": "user", "content": self.task_prompt},
            ],
        )
        chosen_model = analysis.choices[0].message.content.strip()
        if chosen_model not in available_models:
            chosen_model = task_analyser_model

        if not chosen_model:
            raise OrchestratorError("No suitable model was selected for OpenAI.")

        self.chosen_openai_model = chosen_model

    def _model_selection_deepseek(self):
        # Logic to select the best model for Deepseek
        available_models = [m.id for m in self.deepseek_client.models.list().data]
        if not available_models:
            raise OrchestratorError("No available models found for Deepseek.")

        chat_models = [m for m in available_models if 'chat' in m.lower()]
        if not chat_models:
            raise OrchestratorError("No chat models found available for Deepseek, unable to analyse the task.")
        task_analyser_model = chat_models[0]

        analysis = self.deepseek_client.chat.completions.create(
            model=task_analyser_model,
            max_tokens=100,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Pick the most appropriate model for the task, balancing performance and cost. "
                        "Reply with ONLY the exact model id, nothing else.\n"
                        "Available models:\n" + "\n".join(available_models)
                    ),
                },
                {"role": "user", "content": self.task_prompt},
            ],
        )
        chosen_model = analysis.choices[0].message.content.strip()
        if chosen_model not in available_models:
            chosen_model = task_analyser_model

        if not chosen_model:
            raise OrchestratorError("No suitable model was selected for Deepseek.")

        self.chosen_deepseek_model = chosen_model

    def _judge_candidates_anthropic(self, candidates: list):
        if not candidates:
            raise OrchestratorError("No candidates were provided to judge.")

        available_models = [m.id for m in self.anthropic_client.models.list().data]
        sonnet_models = [m for m in available_models if 'sonnet' in m.lower()]
        if not sonnet_models:
            raise OrchestratorError("No sonnet models found available for Anthropic, unable to judge candidates.")
        judge_model = sonnet_models[0]

        # Number the candidates so the judge can refer to them unambiguously.
        numbered = "\n\n".join(f"[{i}]\n{c}" for i, c in enumerate(candidates))

        analysis = self.anthropic_client.messages.create(
            model=judge_model,
            max_tokens=100,
            system=(
                "You are judging candidate responses against the judge prompt. "
                "Each candidate is labelled with an index like [0], [1], etc. "
                "Reply with ONLY the candidate indices, best first, as a comma-separated "
                "list (e.g. '2,0,1'). Do not include any other text.\n\n"
                "Candidates:\n" + numbered
            ),
            messages=[{"role": "user", "content": self.judge_prompt}],
        )

        # Parse the indices from the response and map them back to candidates.
        ranking = re.findall(r"\d+", analysis.content[0].text)
        judged_candidates = []
        seen = set()
        for token in ranking:
            idx = int(token)
            if idx not in seen and 0 <= idx < len(candidates):
                seen.add(idx)
                judged_candidates.append(candidates[idx])

        if not judged_candidates:
            raise OrchestratorError("No candidates were judged successfully by Anthropic.")
        return judged_candidates

    def _judge_candidates_openai(self, candidates: list):
        # Logic to judge candidates using OpenAI
        if not candidates:
            raise OrchestratorError("No candidates were provided to judge.")

        available_models = [m.id for m in self.openai_client.models.list().data]
        judge_models = [m for m in available_models if 'gpt-4' in m.lower() and 'mini' not in m.lower()]
        if not judge_models:
            raise OrchestratorError("No gpt-4 models found available for OpenAI, unable to judge candidates.")
        judge_model = judge_models[0]

        # Number the candidates so the judge can refer to them unambiguously.
        numbered = "\n\n".join(f"[{i}]\n{c}" for i, c in enumerate(candidates))

        analysis = self.openai_client.chat.completions.create(
            model=judge_model,
            max_tokens=100,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are judging candidate responses against the judge prompt. "
                        "Each candidate is labelled with an index like [0], [1], etc. "
                        "Reply with ONLY the candidate indices, best first, as a comma-separated "
                        "list (e.g. '2,0,1'). Do not include any other text.\n\n"
                        "Candidates:\n" + numbered
                    ),
                },
                {"role": "user", "content": self.judge_prompt},
            ],
        )

        # Parse the indices from the response and map them back to candidates.
        ranking = re.findall(r"\d+", analysis.choices[0].message.content)
        judged_candidates = []
        seen = set()
        for token in ranking:
            idx = int(token)
            if idx not in seen and 0 <= idx < len(candidates):
                seen.add(idx)
                judged_candidates.append(candidates[idx])

        if not judged_candidates:
            raise OrchestratorError("No candidates were judged successfully by OpenAI.")
        return judged_candidates

    def _judge_candidates_deepseek(self, candidates: list):
        # Logic to judge candidates using Deepseek
        if not candidates:
            raise OrchestratorError("No candidates were provided to judge.")

        available_models = [m.id for m in self.deepseek_client.models.list().data]
        reasoner_models = [m for m in available_models if 'reasoner' in m.lower()]
        if not reasoner_models:
            raise OrchestratorError("No reasoner models found available for Deepseek, unable to judge candidates.")
        judge_model = reasoner_models[0]

        # Number the candidates so the judge can refer to them unambiguously.
        numbered = "\n\n".join(f"[{i}]\n{c}" for i, c in enumerate(candidates))

        analysis = self.deepseek_client.chat.completions.create(
            model=judge_model,
            max_tokens=100,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are judging candidate responses against the judge prompt. "
                        "Each candidate is labelled with an index like [0], [1], etc. "
                        "Reply with ONLY the candidate indices, best first, as a comma-separated "
                        "list (e.g. '2,0,1'). Do not include any other text.\n\n"
                        "Candidates:\n" + numbered
                    ),
                },
                {"role": "user", "content": self.judge_prompt},
            ],
        )

        # Parse the indices from the response and map them back to candidates.
        ranking = re.findall(r"\d+", analysis.choices[0].message.content)
        judged_candidates = []
        seen = set()
        for token in ranking:
            idx = int(token)
            if idx not in seen and 0 <= idx < len(candidates):
                seen.add(idx)
                judged_candidates.append(candidates[idx])

        if not judged_candidates:
            raise OrchestratorError("No candidates were judged successfully by Deepseek.")
        return judged_candidates