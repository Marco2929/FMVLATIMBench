import base64
import json
import os
from pathlib import Path
from PIL import Image
from openai import OpenAI


def encode_image(image_path):
    """Encode the image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def pad_image(image_path: Path, grid_size) -> Path:
    """Pad the image to make its dimensions multiples of grid_size.
        Args:
            image_path (str): Path to the input image.
            grid_size (int): The grid size to pad to.
        Returns:
            str: Path to the padded image (image_path with '_padded' suffix).
    """

    image = Image.open(image_path)
    width, height = image.size
    new_width = ((width + grid_size - 1) // grid_size) * grid_size
    new_height = ((height + grid_size - 1) // grid_size) * grid_size

    print(f"Padding image from ({width}, {height}) to ({new_width}, {new_height})")

    padded_image = Image.new("RGB", (new_width, new_height))
    padded_image.paste(image, (0, 0))
    padded_image_path = image_path.with_name(image_path.stem + ".g.png")
    padded_image.save(padded_image_path)
    return padded_image_path


def get_api_key() -> str:
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    if API_KEY:
        return API_KEY
    else:
        try:
            with open('.env', 'r') as f:
                for line in f:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
            API_KEY = os.getenv("OPENROUTER_API_KEY")
        except FileNotFoundError:
            raise ValueError("Please set the OPENROUTER_API_KEY environment variable (e.g. in .env)")
    return API_KEY


def parse_ground_truth(json_path: Path) -> str:
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

def parse_response(response: str):
    normalized_response = response.upper().replace(" ", "_")

    return normalized_response.strip()


def evaluate_response(ground_truth: str, response: str):
    return ground_truth == response
