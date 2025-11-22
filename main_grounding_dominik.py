import json
from pathlib import Path
from pprint import pprint
from openai import OpenAI
import argparse
from utils import get_api_key, encode_image, pad_image

from benchmark1_grounding.system_prompts.ui_tars_1_5_7B_single_bbox import SYSTEM_PROMPT

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

def parse_ground_truth_bbox(json_path:Path) -> tuple[str|None, list[int]]:
    # example converts to: {"bbox": [186, 108, 218, 140], "label": "BASKETBALL"}
    with open(json_path, "r") as f:
        data = json.load(f)
    parts = data.get("parts", [])
    if not parts:
        return (None, [])
    part = parts[0]
    part_type = part.get("part_type")
    position = part.get("position", {})
    size = part.get("size", {})
    x_min = position.get("x")
    y_min = position.get("y")
    width = size.get("width_1")
    height = size.get("height_1")
    if None in (x_min, y_min, width, height):
        return (None, [])
    x_max = x_min + width
    y_max = y_min + height
    return (part_type, [x_min, y_min, x_max, y_max])

def generate_model_response(image_path:Path, api_key:str, additional_user_prompt="", model_name="qwen/qwen3-vl-8b-instruct", base_url="https://openrouter.ai/api/v1"):
    client = OpenAI(api_key=api_key, base_url=base_url)
    base64_image = encode_image(pad_image(image_path, 28))
    data_url = f"data:image/jpeg;base64,{base64_image}"
    user_prompt = []
    if additional_user_prompt:
        user_prompt.append({"type": "text", "text": additional_user_prompt})
    user_prompt.append({
        "type": "image_url",
        "image_url": {
            "url": data_url
        }
    })
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
    response = client.chat.completions.create(model=model_name, messages=messages, temperature=0.1)
    part_name = response.choices[0].message.content
    pprint(response.model_dump())
    print(f"Model Response: {part_name}")
    return part_name

def parse_model_response(response: str):
    return response.strip()

def parse_model_response_bbox(response: str) -> tuple[str|None, list[int]]:
    PNG_WIDTH = 640
    PNG_HEIGHT = 441
    response_text = response.strip()
    try:
        bbox_data = json.loads(response_text)
        bbox = bbox_data.get("bbox")
        if bbox is None:
            print("No bowlingball detected.")
            return (None, [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("Invalid bounding box format.")
        
        label = bbox_data.get("label")
        if not label:
            print("No label given.")
            return (None, [])
        
        # Convert normalized coordinates (0-1000) to absolute pixels
        x_min, y_min, x_max, y_max = bbox
        x_min_px = int((x_min / 1000.0) * PNG_WIDTH)
        y_min_px = int((y_min / 1000.0) * PNG_HEIGHT)
        x_max_px = int((x_max / 1000.0) * PNG_WIDTH)
        y_max_px = int((y_max / 1000.0) * PNG_HEIGHT)
        
        return (label, [x_min_px, y_min_px, x_max_px, y_max_px])
    except json.JSONDecodeError:
        print("Failed to parse JSON from model response.")
        print("Raw response:", response_text)
        return (None, [])

def parse_model_response_uitars(response: str) -> tuple[int, int]:
    response_text = response.strip()
    for line in response_text.splitlines():
        if line.startswith("Action:"):
            # regex Action: click(start_box='(230,131)')
            import re
            match = re.search(r"click\(.*='\(\s*(\d+)\s*,\s*(\d+)\s*\)'", line)
            if match:
                x = int(match.group(1))
                y = int(match.group(2))
                return (x, y)
    return (-1, -1)

def evaluate_response(ground_truth: str, response: str):
    return ground_truth == response

def evaluate_response_bbox(ground_truth: tuple[str|None, list[int]], response: tuple[str|None, list[int]]):
    # calc IoU for bbox and compare labels
    gt_label, gt_bbox = ground_truth
    resp_label, resp_bbox = response
    if gt_label != resp_label:
        return 0.0
    if not gt_bbox or not resp_bbox:
        return 0.0
    xA = max(gt_bbox[0], resp_bbox[0])
    yA = max(gt_bbox[1], resp_bbox[1])
    xB = min(gt_bbox[2], resp_bbox[2])
    yB = min(gt_bbox[3], resp_bbox[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    boxBArea = (resp_bbox[2] - resp_bbox[0]) * (resp_bbox[3] - resp_bbox[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def evaluate_response_point(ground_truth: tuple[int, int], response: tuple[int, int]):
    gt_x, gt_y = ground_truth
    resp_x, resp_y = response
    distance = ((gt_x - resp_x) ** 2 + (gt_y - resp_y) ** 2) ** 0.5
    return distance

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

    # ground_truth = parse_ground_truth(input_json)
    ground_truth_bbox = parse_ground_truth_bbox(input_json)
    # response = generate_model_response(input_png, model_name="qwen/qwen3-vl-30b-a3b-instruct") or ""
    # response = generate_model_response(input_png, api_key=API_KEY, model_name="qwen/qwen3-vl-235b-a22b-instruct") or ""
    # response_bbox = generate_model_response(input_png, api_key=API_KEY, additional_user_prompt=ground_truth_bbox[0], model_name="qwen/qwen3-vl-235b-a22b-instruct") or ""
    additional_user_prompt = f"Click the {ground_truth_bbox[0].lower()}" if ground_truth_bbox[0] else "Click the object"
    response_bbox = generate_model_response(input_png, api_key=API_KEY, additional_user_prompt=additional_user_prompt, model_name="bytedance/ui-tars-1.5-7b") or ""
    # response_bbox = parse_model_response_bbox(response_bbox)
    response_bbox = parse_model_response_uitars(response_bbox)
    # score = evaluate_response_bbox(ground_truth_bbox, response_bbox)
    score = evaluate_response_point(((ground_truth_bbox[1][0] + ground_truth_bbox[1][2]) // 2, (ground_truth_bbox[1][1] + ground_truth_bbox[1][3]) // 2), response_bbox)
    print(f"Ground Truth: {ground_truth_bbox}")
    print(f"Response: {response_bbox}")
    print(f"Evaluation Score: {score}")
