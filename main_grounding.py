import json
from pathlib import Path
import argparse
from enum import Enum

from benchmark1_grounding.system_prompts.ui_tars_1_5_7B_single_bbox import SYSTEM_PROMPT as UITARS_LOCALIZE_SYSTEM_PROMPT
from benchmark1_grounding.system_prompts.qwen3vl_object_recognition import SYSTEM_PROMPT as QWEN3_CLASSIFY_SYSTEM_PROMPT
from benchmark1_grounding.system_prompts.qwen3vl_single_bbox import SYSTEM_PROMPT as QWEN3_LOCALIZE_SYSTEM_PROMPT
from benchmark1_grounding.system_prompts.qwen3vl_multi_bbox import SYSTEM_PROMPT as QWEN3_MULTILOCALIZE_SYSTEM_PROMPT
from src.basics import get_api_key, get_base_url
from src.image_processing import get_image_dimensions
from src.llm_wrapper import BoundingBox, Qwen3VLLLMWrapper, UiTarsLLMWrapper


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
    parts: list[BoundingBox] = []
    for part in data.get("parts", []):
        match benchmark_type:
            case BenchmarkType.QWEN3_CLASSIFY:
                part_name = parse_classification(part)
                return part_name if part_name else "NONE"
            case BenchmarkType.QWEN3_LOCALIZE | BenchmarkType.UITARS_LOCALIZE | BenchmarkType.QWEN3_LOCALIZE_MULTI:
                bbox = parse_bbox(part)
                if bbox:
                    assert 'UNKNOWN' not in bbox.label, "Ground truth contains UNKNOWN label. Model won't be able to predict it."
                    parts.append(bbox)
            case _:
                raise ValueError(f"Benchmark type not implemented: {benchmark_type}")
    return parts

def parse_classification(part: dict) -> str|None:
    part_type = part.get("part_type")
    return part_type if part_type else None

def parse_bbox(part: dict) -> BoundingBox|None:
    # example converts to: ("BASKETBALL", [186, 108, 218, 140])
    part_type = part.get("part_type") or ''
    position = part.get("position", {})
    size = part.get("size", {})
    x_min = position.get("x")
    y_min = position.get("y")
    width = size.get("width_1")
    height = size.get("height_1")
    if None in (x_min, y_min, width, height):
        return None
    x_max = x_min + width
    y_max = y_min + height
    return BoundingBox(
        label=part_type.upper(),
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max
    )


def evaluate_response(ground_truth: str, response: str):
    return ground_truth == response

def evaluate_response_bbox(ground_truth: BoundingBox, response: BoundingBox):
    # calc IoU for bbox and compare labels
    if ground_truth.label != response.label:
        return 0.0
    if not ground_truth or not response:
        return 0.0
    iou = ground_truth.intersection_over_union(response)
    return iou

def evaluate_response_bboxes(ground_truth: list[BoundingBox], response: list[BoundingBox]):
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
        required=True,
        help="Benchmark type with combination of model and task."
    )

    args = parser.parse_args()
    benchmark = BenchmarkType(args.benchmark)

    API_KEY = get_api_key()
    BASE_URL = get_base_url()

    benchmark_files = get_relevant_files(benchmark)
    
    for file_path in benchmark_files:
        input_png = file_path.with_suffix(".png")
        input_json = file_path.with_suffix(".json")

        image_width, image_height = get_image_dimensions(input_png)

        ground_truth = parse_ground_truth(input_json, benchmark)

        response = None
        score = None
        match benchmark:
            case BenchmarkType.QWEN3_CLASSIFY:
                assert isinstance(ground_truth, str)
                model = Qwen3VLLLMWrapper(api_key=API_KEY, base_url=BASE_URL, model_name=benchmark.get_model_name())
                response = model.generate_model_response(input_png, system_prompt=benchmark.get_system_prompt()) or ""
                response = model.parse_response_text(response)
                score = evaluate_response(ground_truth, response)
            case BenchmarkType.QWEN3_LOCALIZE:
                assert isinstance(ground_truth, list)
                assert len(ground_truth) >= 1
                ground_truth_bbox = ground_truth[0]
                additional_user_prompt = f"Locate the {ground_truth_bbox.label}"
                model = Qwen3VLLLMWrapper(api_key=API_KEY, base_url=BASE_URL, model_name=benchmark.get_model_name())
                response = model.generate_model_response(input_png, system_prompt=benchmark.get_system_prompt(), additional_user_prompt=additional_user_prompt) or ""
                response = model.parse_response_bbox(response, image_width=image_width, image_height=image_height)
                score = 0 if response is None else evaluate_response_bbox(ground_truth_bbox, response)
            case BenchmarkType.QWEN3_LOCALIZE_MULTI:
                assert isinstance(ground_truth, list)
                ground_truth_bbox = ground_truth
                assert len(ground_truth_bbox) >= 1
                object_list = ", ".join([gt.label for gt in ground_truth_bbox if gt.label is not None])
                additional_user_prompt = f"Locate the following objects: {object_list}"
                model = Qwen3VLLLMWrapper(api_key=API_KEY, base_url=BASE_URL, model_name=benchmark.get_model_name())
                response = model.generate_model_response(input_png, system_prompt=benchmark.get_system_prompt(), additional_user_prompt=additional_user_prompt) or ""
                response = model.parse_response_bboxes(response, image_height=image_height, image_width=image_width)
                score = evaluate_response_bboxes(ground_truth_bbox, response)
            case BenchmarkType.UITARS_LOCALIZE:
                assert isinstance(ground_truth, list)
                assert len(ground_truth) >= 1
                ground_truth_bbox = ground_truth[0]
                additional_user_prompt = f"Click the {ground_truth_bbox.label}"
                model = UiTarsLLMWrapper(api_key=API_KEY, base_url=BASE_URL, model_name=benchmark.get_model_name())
                response = model.generate_model_response(input_png, system_prompt=benchmark.get_system_prompt(), additional_user_prompt=additional_user_prompt) or ""
                response = model.parse_response_point(response)
                score = evaluate_response_point(ground_truth_bbox.center(), response)
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
