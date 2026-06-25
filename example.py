from anthropic import Anthropic
from openai import OpenAI
import os
from dotenv import load_dotenv
from orchestrator import Orchestrator

load_dotenv()  # Load environment variables from .env file
anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
#openai_client = OpenAI(api_key="...")
#deepseek_client = OpenAI(api_key="...", base_url="https://api.deepseek.com")

orch = Orchestrator(
    task_prompt="Write a concise product description for a smart water bottle.",
    judge_prompt="Pick the description that is clearest and most persuasive.",
    anthropic_client=anthropic_client,
    #openai_client=openai_client,
    #deepseek_client=deepseek_client,
    candidate_number=2,
    max_task_tokens=1000,
)

best = orch.run_task()
print(best)