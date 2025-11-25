import os
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI
from utils import get_api_key

# Configuration
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = get_api_key()
MODEL_NAME = "qwen/qwen3-vl-235b-a22b-instruct"


class OpenRouterLLM(DeepEvalBaseLLM):
    def __init__(self, model_name, api_key, base_url):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return chat_completion.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        # DeepEval relies heavily on async execution.
        # Ideally, use AsyncOpenAI here, but wrapping the sync call works for simple cases.
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


# Initialize the custom LLM wrapper
openrouter_llm = OpenRouterLLM(
    model_name=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL
)

# Test Data
actual_output = "We offer a 30-day full refund at no extra cost."
expected_output = "You are eligible for a 30 day full refund at no extra cost."
retrieval_context = ["All customers are eligible for a 30 day full refund at no extra cost."]

# Initialize Metric with the custom model
metric = ContextualPrecisionMetric(
    threshold=0.7,
    model=openrouter_llm,  # Pass the custom wrapper instance here
    include_reason=True
)

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=actual_output,
    expected_output=expected_output,
    retrieval_context=retrieval_context
)

# Execution
# Note: evaluate() handles the loop and printing internally
evaluate(test_cases=[test_case], metrics=[metric])
