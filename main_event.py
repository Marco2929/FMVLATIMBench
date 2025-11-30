from pathlib import Path
import argparse
from typing import List

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.models.base_model import DeepEvalBaseLLM

from deepeval_openrouter_utils import OpenRouterLLM
from benchmark3_event.system_prompts.qwen3vl_outcome_text import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_OUTCOME_TEXT
from benchmark3_event.system_prompts.qwen3vl_outcome_visual import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_OUTCOME_VISUAL
from benchmark3_event.system_prompts.qwen3vl_cause_visual import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_CAUSE_VISUAL
from benchmark3_event.system_prompts.qwen3vl_cause_text import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_CAUSE_TEXT
from benchmark3_event.system_prompts.qwen3vl_effect_text import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_EFFECT_TEXT
from benchmark3_event.system_prompts.qwen3vl_effect_visual import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_EFFECT_VISUAL

from utils import get_api_key, generate_model_response, parse_ground_truth

allowed_categories = ["outcome_text", "outcome_visual", "effect_text", "effect_visual", "cause_text",
                      "cause_visual"]

PROMPT_MAPPING = {
    "outcome_text": SYSTEM_PROMPT_OUTCOME_TEXT,
    "outcome_visual": SYSTEM_PROMPT_OUTCOME_VISUAL,
    "effect_text": SYSTEM_PROMPT_EFFECT_TEXT,
    "effect_visual": SYSTEM_PROMPT_EFFECT_VISUAL,
    "cause_text": SYSTEM_PROMPT_CAUSE_TEXT,
    "cause_visual": SYSTEM_PROMPT_CAUSE_VISUAL,
}


def get_system_prompt(input_category: str):
    if input_category in PROMPT_MAPPING:
        return PROMPT_MAPPING[input_category]
    else:
        raise ValueError(f"Invalid category: {input_category}")


def evaluate_response_deep_eval(openrouter_llm: DeepEvalBaseLLM, input: str, expected_output: str, actual_output: str,
                                retrieval_context: List[str]):
    # Initialize Metric with the custom model
    metric = ContextualPrecisionMetric(
        threshold=0.7,
        model=openrouter_llm,  # Pass the custom wrapper instance here
        include_reason=True
    )

    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context
    )

    results = evaluate(test_cases=[test_case], metrics=[metric])

    return results.test_results[0].success


def calculate_benchmark_results():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Grounding Model Evaluation")
    parser.add_argument("--input", required=True, type=str, metavar="FILE",
                        help="Path to the input test that expects .PNG, .py and .json files.",
                        )
    parser.add_argument(
        "--category",
        type=str,
        choices=allowed_categories,
        help=f"Possible categories are: {allowed_categories}",
        default=allowed_categories[0]
    )

    args = parser.parse_args()

    input_category = args.category.lower()
    if input_category not in allowed_categories:
        raise ValueError(f"Category {input_category} is not supported.")

    input_png = Path(args.input).with_suffix(".png")
    if not input_png.exists():
        raise FileNotFoundError(f"Input image file not found: {input_png}")

    input_py = Path(args.input).with_suffix(".py")
    if not input_py.exists():
        raise FileNotFoundError(f"Input Python file not found: {input_py}")
    else:
        with open(input_py, 'r') as f:
            raw_content = f.read()
            parts =raw_content.split('"')
            instruct_prompt = parts[1]

    input_json = Path(args.input).with_suffix(".json")
    if not input_json.exists():
        raise FileNotFoundError(f"Input Json file not found: {input_json}")

    SYSTEM_PROMPT = get_system_prompt(input_category=input_category)

    API_KEY = get_api_key()
    model_name = "qwen/qwen3-vl-235b-a22b-instruct"

    openrouter_llm = OpenRouterLLM(
        model_name=model_name,
        api_key=API_KEY,
    )

    ground_truth = parse_ground_truth(input_json)

    # response = generate_model_response(input_png, model_name="qwen/qwen3-vl-30b-a3b-instruct") or ""
    response = generate_model_response(input_png, api_key=API_KEY, SYSTEM_PROMPT=SYSTEM_PROMPT,
                                       instruct_prompt=instruct_prompt,
                                       model_name=model_name)
    score = evaluate_response_deep_eval(openrouter_llm=openrouter_llm, input=SYSTEM_PROMPT, expected_output=ground_truth,
                                        actual_output=response, retrieval_context=[instruct_prompt])
    print(f"Task: {instruct_prompt}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Parsed Response: {response}")
    print(f"Evaluation Score: {score}")
