import base64
import json
from pathlib import Path
from openai import OpenAI
import argparse
import os

from benchmark_grounding.system_prompts.qwen3vl_object_recognition import SYSTEM_PROMPT

def encode_image(image_path):
    """Encode the image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_api_key() -> str:
    with open('../.env', 'r') as f:
        for line in f:
            key, value = line.strip().split('=', 1)
            os.environ[key] = value
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not API_KEY:
        raise ValueError("Please set the OPENROUTER_API_KEY environment variable (e.g. in .env)")
    return API_KEY

def parse_ground_truth(json_path:Path) -> str:
    '''Example:
{
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
    part_names = []
    for part in data.get("parts", []):
        part_type = part.get("part_type")
        if part_type:
            part_names.append(part_type)
    data = ",".join(part_names) if part_names else "NONE"
    return data

def generate_model_response(image_path:Path, api_key:str, model_name="qwen/qwen3-vl-8b-instruct", base_url="https://openrouter.ai/api/v1"):
    client = OpenAI(api_key=api_key, base_url=base_url)
    base64_image = encode_image(image_path)
    data_url = f"data:image/jpeg;base64,{base64_image}"
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
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
    return response.strip()

def evaluate_response(ground_truth: str, response: str):
    return ground_truth == response

def calculate_benchmark_results():
    pass

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Grounding Model Evaluation")
    parser.add_argument("--input", required=True, type=str, metavar="FILE", help="Path to the input test that expects .PNG and .json files.")

    args = parser.parse_args()

    input_png = Path(args.input).with_suffix(".png")
    if not input_png.exists():
        raise FileNotFoundError(f"Input image file not found: {input_png}")
    input_json = Path(args.input).with_suffix(".json")
    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON file not found: {input_json}")

    API_KEY = get_api_key()

    ground_truth = parse_ground_truth(input_json)
    # response = generate_model_response(input_png, model_name="qwen/qwen3-vl-30b-a3b-instruct") or ""
    response = generate_model_response(input_png, api_key=API_KEY, model_name="qwen/qwen3-vl-235b-a22b-instruct") or ""
    response = parse_model_response(response)
    score = evaluate_response(ground_truth, response)
    print(f"Ground Truth: {ground_truth}")
    print(f"Response: {response}")
    print(f"Evaluation Score: {score}")
