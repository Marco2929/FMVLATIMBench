import base64
import json
from pathlib import Path
from openai import OpenAI
import argparse
from typing import List
import os

from benchmark2_understanding.system_prompts.qwen3vl_object_property_ident_with_instruct import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_WITH
from benchmark2_understanding.system_prompts.qwen3vl_object_property_ident_without_instruct import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_WITHOUT
from benchmark2_understanding.system_prompts.qwen3vl_object_state_ident import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_STATE_IDENT
from utils import get_api_key, parse_ground_truth, generate_model_response, parse_response, evaluate_response

allowed_categories = ["with_instruct", "without_instruct", "state_ident"]


def get_system_prompt(input_category: List[str]):
    if input_category == allowed_categories[0]:
        return SYSTEM_PROMPT_WITH
    elif input_category == allowed_categories[1]:
        return SYSTEM_PROMPT_WITHOUT
    else:
        return SYSTEM_PROMPT_STATE_IDENT


def calculate_benchmark_results():
    pass


if __name__ == "__main__":
    allowed_categories = ["with_instruct", "without_instruct", "state_ident"]

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

    ground_truth = parse_response(parse_ground_truth(input_json))
    # response = generate_model_response(input_png, model_name="qwen/qwen3-vl-30b-a3b-instruct") or ""
    response = generate_model_response(input_png, api_key=API_KEY, SYSTEM_PROMPT=SYSTEM_PROMPT,
                                       instruct_prompt=instruct_prompt,
                                       model_name="qwen/qwen3-vl-235b-a22b-instruct") or ""
    response = parse_response(response)
    score = evaluate_response(ground_truth, response)
    print(f"Ground Truth: {ground_truth}")
    print(f"Parsed Response: {response}")
    print(f"Evaluation Score: {score}")
