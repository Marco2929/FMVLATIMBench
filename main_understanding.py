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
from utils import get_api_key, encode_image

allowed_categories = ["with_instruct", "without_instruct", "state_ident"]


def get_system_prompt(input_category: List[str]):
    if input_category == allowed_categories[0]:
        return SYSTEM_PROMPT_WITH
    elif input_category == allowed_categories[1]:
        return SYSTEM_PROMPT_WITHOUT
    else:
        return SYSTEM_PROMPT_STATE_IDENT


def parse_ground_truth(json_path: Path) -> str:
    '''Example:

  "version": "TIM2",
  "title": "OBJ_REC1",
  "description": "Identify the object.",
  "background": {
    "color": 3
  },
  "global_settings": {
    "pressure": 67,
    "gravity": 272,
    "music": 1000,
    "num_moving": 1
  },
  "parts": [
    {
      "part_type": "BASKETBALL",
      "position": {
        "x": 186,
        "y": 108
      },
      "size": {
        "width_1": 32,
        "height_1": 32,
        "width_2": 32,
        "height_2": 32
      },
      "flags_3": [
        "UNKNOWN_0x8",
        "LOCKED",
        "SHOW_SOLUTION_ICON"
      ]
    }
  ]
}
which converts to: BASKETBALL
    '''
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["solution"]


def generate_model_response(image_path: Path, api_key: str, SYSTEM_PROMPT: str, instruct_prompt: str,
                            model_name="qwen/qwen3-vl-8b-instruct",
                            base_url="https://openrouter.ai/api/v1"):
    client = OpenAI(api_key=api_key, base_url=base_url)
    base64_image = encode_image(image_path)
    data_url = f"data:image/jpeg;base64,{base64_image}"
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + instruct_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }
            ]
        }
    ]
    response = client.chat.completions.create(model=model_name, messages=messages)
    part_name = response.choices[0].message.content
    print(f"Model Response: {part_name}")
    return part_name


def parse_model_response(response: str):
    normalized_response = response.upper().replace(" ", "_")

    return normalized_response.strip()


def evaluate_response(ground_truth: str, response: str):
    return ground_truth == response


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

    input_png = Path(args.input).with_suffix(".png")
    if not input_png.exists():
        raise FileNotFoundError(f"Input image file not found: {input_png}")
    input_json = Path(args.input).with_suffix(".json")
    if not input_json.exists():
        raise FileNotFoundError(f"Input Json file not found: {input_json}")
    input_py = Path(args.input).with_suffix(".py")
    if not input_py.exists():
        raise FileNotFoundError(f"Input Python file not found: {input_py}")
    else:
        with open(input_py, 'r') as f:
            instruct_prompt = f.read()
    input_category = args.category.lower()
    if input_category not in allowed_categories:
        raise ValueError(f"Category {input_category} is not supported.")

    SYSTEM_PROMPT = get_system_prompt(input_category=input_category)

    API_KEY = get_api_key()

    ground_truth = parse_ground_truth(input_json)
    # response = generate_model_response(input_png, model_name="qwen/qwen3-vl-30b-a3b-instruct") or ""
    response = generate_model_response(input_png, api_key=API_KEY, SYSTEM_PROMPT=SYSTEM_PROMPT,
                                       instruct_prompt=instruct_prompt,
                                       model_name="qwen/qwen3-vl-235b-a22b-instruct") or ""
    response = parse_model_response(response)
    score = evaluate_response(ground_truth, response)
    print(f"Ground Truth: {ground_truth}")
    print(f"Parsed Response: {response}")
    print(f"Evaluation Score: {score}")
