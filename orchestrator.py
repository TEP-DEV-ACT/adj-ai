from . import _constants, _exceptions
from anthropic import Anthropic
from openai import OpenAI

class orchestrator:
    # client options
    anthropic_client: Anthropic
    openai_client: OpenAI
    deepseek_client: OpenAI
    candidates: int = 5
    max_task_tokens: int = 1000

    # constants
    task_prompt: str
    judge_prompt: str 

    def __init__(
    self,
    task_prompt: str,
    judge_prompt: str,
    anthropic_client: Anthropic,
    openai_client: OpenAI,
    deepseek_client: OpenAI,
    candidates: int = 5,
    max_task_tokens: int = 1000,
    ):
        has_atleast_one_client = (
            anthropic_client is not None or 
            openai_client is not None or 
            deepseek_client is not None
            )
        if not has_atleast_one_client:
            raise _exceptions.OrchestratorError(
                "At least one client of Anthropic, OpenAI, or Deepseek must be provided."
            )
        
        self.task_prompt = task_prompt
        self.judge_prompt = judge_prompt
        self.anthropic_client = anthropic_client
        self.openai_client = openai_client
        self.deepseek_client = deepseek_client
        self.candidates = candidates
        self.max_task_tokens = max_task_tokens
    def run_task(self):
        # Generate candidates using the provided clients
        candidates = []
        if self.anthropic_client:
            candidates += self._generate_candidates_anthropic(self)
        if self.openai_client:
            candidates += self._generate_candidates_openai(self)
        if self.deepseek_client:
            candidates += self._generate_candidates_deepseek(self)

        # Judge the candidates
        winner_candidate = self._judge_candidates(candidates)

        return winner_candidate
    
    def _generate_candidates_anthropic(self):
        # Use the Anthropic client to generate candidates
        response = self.anthropic_client.completions.create(
            model=self.chosen_anthropic_model,
            prompt=f"{self.task_prompt}",
            max_tokens=self.max_task_tokens,
            n=self.candidates
        )
        return [choice.text for choice in response.choices]
    
    def _generate_candidates_openai(self):
        raise NotImplementedError("OpenAI candidate generation is not implemented yet.")
    
    def _generate_candidates_deepseek(self):
        raise NotImplementedError("Deepseek candidate generation is not implemented yet.")
    
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
            raise _exceptions.OrchestratorError("No candidates were judged successfully.")
    
    def _model_selection(self):
        # Logic to select the best model based on some criteria
        if self.anthropic_client:
            self._model_selection_anthropic()
        raise NotImplementedError("Model selection logic is not implemented yet.")  

    def _model_selection_anthropic(self):
        # Logic to select the best model for Anthropic
        available_models = [m for m in self.anthropic_client.models.list().data]
        if not available_models:
            raise _exceptions.OrchestratorError("No available models found for Anthropic.")

        task_analyser_model = [m for m in available_models if 'haiku' in m.lower()][0]
        if not task_analyser_model:
            raise _exceptions.OrchestratorError("No haiku models found available for Anthropic, unable to analyse the task.")
        
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
        
        self.chosen_anthropic_model = chosen_model

    def _model_selection_openai(self):
        # Logic to select the best model for OpenAI
        available_models = [m for m in self.openai_client.models.list().data]
        raise NotImplementedError("Model selection logic is not implemented yet.")  

    def _model_selection_deepseek(self):
        available_models = [m for m in self.deepseek_client.models.list().data]
        raise NotImplementedError("Model selection logic is not implemented yet.")

    def _judge_candidates_anthropic(self, candidates: list):
        available_models = [m for m in self.anthropic_client.models.list().data]
        if not available_models:
            raise _exceptions.OrchestratorError("No available models found for Anthropic.")
        
        judge_model = [m for m in available_models if 'sonnet' in m.lower()][0]
        if not judge_model:
            raise _exceptions.OrchestratorError("No sonnet models found available for Anthropic, unable to judge candidates.")
        
        analysis = self.anthropic_client.messages.create(
            model = judge_model,
            max_tokens = 100,
            system=(
            "Judge the quality of the candidates based on the task prompt. "
            "Reply with ONLY the candidates in order of preference, nothing else.\n"
            "Candidates:\n" + "\n".join(candidates)
            ),
            messages=[{"role": "user", "content": self.judge_prompt}],
        )

        judged_candidates = [line.strip() for line in analysis.content[0].text.splitlines() if line.strip() in candidates]
        if not judged_candidates:
            raise _exceptions.OrchestratorError("No candidates were judged successfully by Anthropic.")
        return judged_candidates

    def _judge_candidates_openai(self, candidates: list):
        # Logic to judge candidates using OpenAI
        raise NotImplementedError("Judging logic for OpenAI is not implemented yet.")

    def _judge_candidates_deepseek(self, candidates: list):
        # Logic to judge candidates using Deepseek
        raise NotImplementedError("Judging logic for Deepseek is not implemented yet.") 