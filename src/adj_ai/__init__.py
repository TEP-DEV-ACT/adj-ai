"""adj-ai: a multi-provider LLM orchestrator.

Generate-then-judge: produce several candidate responses to a task from one or
more model providers (Anthropic, OpenAI, DeepSeek), then use a "judge" model to
rank them and return the best one.
"""

from .exceptions import OrchestratorError
from .orchestrator import Orchestrator

__version__ = "1.0.1"

__all__ = ["Orchestrator", "OrchestratorError", "__version__"]
