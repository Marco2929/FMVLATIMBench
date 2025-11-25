import json
from pathlib import Path
from pprint import pprint
from openai import OpenAI
import argparse
from enum import Enum
from utils import get_api_key, encode_image, pad_image

from benchmark1_grounding.system_prompts.ui_tars_1_5_7B_single_bbox import SYSTEM_PROMPT as UITARS_LOCALIZE_SYSTEM_PROMPT
from benchmark1_grounding.system_prompts.qwen3vl_object_recognition import SYSTEM_PROMPT as QWEN3_CLASSIFY_SYSTEM_PROMPT
from benchmark1_grounding.system_prompts.qwen3vl_single_bbox import SYSTEM_PROMPT as QWEN3_LOCALIZE_SYSTEM_PROMPT
from benchmark1_grounding.system_prompts.qwen3vl_multi_bbox import SYSTEM_PROMPT as QWEN3_MULTILOCALIZE_SYSTEM_PROMPT


class BenchmarkType(Enum):
    QWEN3_CLASSIFY = 'qwen3_classify'
    QWEN3_LOCALIZE = 'qwen3_localize'
    QWEN3_LOCALIZE_MULTI = 'qwen3_localize_multi'
    UITARS_LOCALIZE = 'uitars_localize'

    def get_model_name(self) -> str:
        match self:
            case BenchmarkType.QWEN3_CLASSIFY | BenchmarkType.QWEN3_LOCALIZE | BenchmarkType.QWEN3_LOCALIZE_MULTI:
                # return "qwen/qwen3-vl-8b-instruct"
                return "qwen/qwen3-vl-235b-a22b-instruct"
            case BenchmarkType.UITARS_LOCALIZE:
                return "bytedance/ui-tars-1.5-7b"
            case _:
                raise ValueError(f"Benchmark type not implemented: {self}")

    def get_system_prompt(self) -> str:
        match self:
            case BenchmarkType.QWEN3_CLASSIFY:
                return QWEN3_CLASSIFY_SYSTEM_PROMPT
            case BenchmarkType.QWEN3_LOCALIZE:
                return QWEN3_LOCALIZE_SYSTEM_PROMPT
            case BenchmarkType.QWEN3_LOCALIZE_MULTI:
                return QWEN3_MULTILOCALIZE_SYSTEM_PROMPT
            case BenchmarkType.UITARS_LOCALIZE:
                return UITARS_LOCALIZE_SYSTEM_PROMPT
            case _:
                raise ValueError(f"Benchmark type not implemented: {self}")

def parse_ground_truth(json_path:Path, benchmark_type: BenchmarkType):
    '''Takes a benchmark type and a json file and converts it into all possible benchmark results that 
    Example:
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
which converts to: 
    BASKETBALL for classify, 
    ("BASKETBALL", [186, 108, 218, 140]) for localize,
    [("BASKETBALL", [186, 108, 218, 140])] for localize_multi
    '''
    with open(json_path, "r") as f:
        data = json.load(f)
    parts: list[tuple[str, list[int]]] = []
    for part in data.get("parts", []):
        match benchmark_type:
            case BenchmarkType.QWEN3_CLASSIFY:
                part_name = parse_classification(part)
                return part_name if part_name else "NONE"
            case BenchmarkType.QWEN3_LOCALIZE | BenchmarkType.UITARS_LOCALIZE | BenchmarkType.QWEN3_LOCALIZE_MULTI:
                label, bbox = parse_bbox(part)
                if label:
                    assert 'UNKNOWN' not in label, "Ground truth contains UNKNOWN label. Model won't be able to predict it."
                    parts.append((label, bbox))
            case _:
                raise ValueError(f"Benchmark type not implemented: {benchmark_type}")
    return parts

def parse_classification(part: dict) -> str|None:
    part_type = part.get("part_type")
    return part_type if part_type else None

def parse_bbox(part: dict) -> tuple[str|None, list[int]]:
    # example converts to: ("BASKETBALL", [186, 108, 218, 140])
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

def generate_model_response(image_path:Path, api_key:str, system_prompt:str, additional_user_prompt="", model_name="qwen/qwen3-vl-8b-instruct", base_url="https://openrouter.ai/api/v1"):
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
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
    print("Sending request to model...")
    response = client.chat.completions.create(model=model_name, messages=messages, temperature=0.1, timeout=30, max_tokens=10000)
    part_name = response.choices[0].message.content
    pprint(response.model_dump())
    print(f"Model Response: {part_name}")
    return part_name

def parse_model_response(response: str):
    return response.strip()

def parse_model_response_bbox(response: str) -> tuple[str|None, list[int]]:
    PNG_WIDTH = 640
    PNG_HEIGHT = 441
    response_text = response.strip().replace('```json', '').replace('```', '')
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
        
        return (label.upper(), [x_min_px, y_min_px, x_max_px, y_max_px])
    except json.JSONDecodeError:
        print("Failed to parse JSON from model response.")
        print("Raw response:", response_text)
        return (None, [])
    
def parse_model_response_bboxes(response: str) -> list[tuple[str|None, list[int]]]:
    PNG_WIDTH = 640
    PNG_HEIGHT = 441
    response_text = response.strip().replace('```json', '').replace('```', '')
    results = []
    try:
        bbox_list = json.loads(response_text)
        if not isinstance(bbox_list, list):
            raise ValueError("Invalid bounding boxes format.")
        
        for bbox_data in bbox_list:
            bbox = bbox_data.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                print("Invalid bounding box format, skipping.")
                continue
            
            label = bbox_data.get("label")
            if not label:
                print("No label given, skipping.")
                continue
            
            # Convert normalized coordinates (0-1000) to absolute pixels
            x_min, y_min, x_max, y_max = bbox
            x_min_px = int((x_min / 1000.0) * PNG_WIDTH)
            y_min_px = int((y_min / 1000.0) * PNG_HEIGHT)
            x_max_px = int((x_max / 1000.0) * PNG_WIDTH)
            y_max_px = int((y_max / 1000.0) * PNG_HEIGHT)
            
            results.append((label.upper(), [x_min_px, y_min_px, x_max_px, y_max_px]))
        return results
    except json.JSONDecodeError:
        print("Failed to parse JSON from model response.")
        print("Raw response:", response_text)
        return []

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

def evaluate_response_bboxes(ground_truth: list[tuple[str|None, list[int]]], response: list[tuple[str|None, list[int]]]):
    # calc average IoU for multiple bboxes and compare labels
    if not ground_truth or not response:
        return 0.0
    total_iou = 0.0
    matched = 0
    for gt in ground_truth:
        for resp in response:
            iou = evaluate_response_bbox(gt, resp)
            if iou > 0:
                total_iou += iou
                matched += 1
                break
    if matched == 0:
        return 0.0
    average_iou = total_iou / matched
    return average_iou

def evaluate_response_point(ground_truth: tuple[int, int], response: tuple[int, int]):
    gt_x, gt_y = ground_truth
    resp_x, resp_y = response
    distance = ((gt_x - resp_x) ** 2 + (gt_y - resp_y) ** 2) ** 0.5
    return distance

def get_relevant_files(benchmark_type: BenchmarkType) -> list[Path]:
    base_path = Path("benchmark1_grounding/examples")
    single_object = base_path / "single_object"
    multi_object = base_path / "multi_object"
    
    match benchmark_type:
        case BenchmarkType.QWEN3_CLASSIFY:
            folders = [single_object]
        case BenchmarkType.QWEN3_LOCALIZE:
            folders = [single_object, multi_object]
        case BenchmarkType.QWEN3_LOCALIZE_MULTI:
            folders = [single_object, multi_object]
        case BenchmarkType.UITARS_LOCALIZE:
            folders = [single_object, multi_object]
        case _:
            raise ValueError(f"Benchmark type not implemented: {benchmark_type}")
    
    file_paths = []
    for folder in folders:
        file_paths.extend([p for p in folder.glob("*.json")])
    return sorted(file_paths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Grounding Model Evaluation")
    parser.add_argument("--save", action="store_true", help="Flag to save the results in <input>.txt")
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=[bt.value for bt in BenchmarkType],
        default=BenchmarkType.QWEN3_CLASSIFY.value,
        help="Benchmark type with combination of model and task."
    )

    args = parser.parse_args()
    benchmark = BenchmarkType(args.benchmark)

    API_KEY = get_api_key()

    benchmark_files = get_relevant_files(benchmark)
    
    for file_path in benchmark_files:
        input_png = file_path.with_suffix(".png")
        input_json = file_path.with_suffix(".json")

        ground_truth = parse_ground_truth(input_json, benchmark)

        response = None
        score = None
        match benchmark:
            case BenchmarkType.QWEN3_CLASSIFY:
                assert isinstance(ground_truth, str)
                response = generate_model_response(input_png, api_key=API_KEY, system_prompt=benchmark.get_system_prompt(), model_name=benchmark.get_model_name()) or ""
                response = parse_model_response(response)
                score = evaluate_response(ground_truth, response)
            case BenchmarkType.QWEN3_LOCALIZE:
                assert isinstance(ground_truth, list)
                assert len(ground_truth) >= 1
                ground_truth_bbox = ground_truth[0]
                assert ground_truth_bbox[0] is not None
                additional_user_prompt = f"Locate the {ground_truth_bbox[0]}"
                response = generate_model_response(input_png, api_key=API_KEY, system_prompt=benchmark.get_system_prompt(), additional_user_prompt=additional_user_prompt, model_name=benchmark.get_model_name()) or ""
                response = parse_model_response_bbox(response)
                score = evaluate_response_bbox(ground_truth_bbox, response)
            case BenchmarkType.QWEN3_LOCALIZE_MULTI:
                assert isinstance(ground_truth, list)
                ground_truth_bbox = ground_truth
                assert len(ground_truth_bbox) >= 1
                object_list = ", ".join([gt[0] for gt in ground_truth_bbox if gt[0] is not None])
                additional_user_prompt = f"Locate the following objects: {object_list}"
                response = generate_model_response(input_png, api_key=API_KEY, system_prompt=benchmark.get_system_prompt(), additional_user_prompt=additional_user_prompt, model_name=benchmark.get_model_name()) or ""
                response = parse_model_response_bboxes(response)
                score = evaluate_response_bboxes(ground_truth_bbox, response)
            case BenchmarkType.UITARS_LOCALIZE:
                assert isinstance(ground_truth, list)
                assert len(ground_truth) >= 1
                ground_truth_bbox = ground_truth[0]
                assert ground_truth_bbox[0] is not None
                additional_user_prompt = f"Click the {ground_truth_bbox[0]}"
                response = generate_model_response(input_png, api_key=API_KEY, system_prompt=benchmark.get_system_prompt(), additional_user_prompt=additional_user_prompt, model_name=benchmark.get_model_name()) or ""
                response = parse_model_response_uitars(response)
                score = evaluate_response_point(((ground_truth_bbox[1][0] + ground_truth_bbox[1][2]) // 2, (ground_truth_bbox[1][1] + ground_truth_bbox[1][3]) // 2), response)
            case _:
                raise ValueError(f"Benchmark type not implemented: {benchmark}")

        if score is not None:
            if args.save:
                results_path = file_path.with_suffix(f".{benchmark.value}.txt")
                with open(results_path, "w") as f:
                    f.write(f"Ground Truth: {ground_truth}\n")
                    f.write(f"Response: {response}\n")
                    f.write(f"Evaluation Score: {score}\n")
                print(f"Results saved to {results_path}")

        print(f"Ground Truth: {ground_truth}")
        print(f"Response: {response}")
        print(f"Evaluation Score: {score}")
