import json
from pathlib import Path
from typing import override
from itertools import combinations
from tqdm import tqdm

from benchmark1_grounding.system_prompts.ui_tars_1_5_7B_single_bbox import SYSTEM_PROMPT as UITARS_LOCALIZE_SYSTEM_PROMPT
from benchmark1_grounding.system_prompts.qwen3vl_object_recognition import SYSTEM_PROMPT as QWEN3_CLASSIFY_SYSTEM_PROMPT
from benchmark1_grounding.system_prompts.qwen3vl_single_bbox import SYSTEM_PROMPT as QWEN3_LOCALIZE_SYSTEM_PROMPT
from benchmark1_grounding.system_prompts.qwen3vl_multi_bbox import SYSTEM_PROMPT as QWEN3_MULTILOCALIZE_SYSTEM_PROMPT
from src.benchmark_base import BenchmarkBase, BenchmarkCli
from src.image_processing import get_image_dimensions
from src.llm_wrapper import BoundingBox, Point, Qwen3VLLLMWrapper, UiTarsLLMWrapper
from src.results_model import SingleTaskResult


class GroundingBenchmarkType(BenchmarkBase):
    QWEN3_CLASSIFY = 'qwen3_classify'
    QWEN3_LOCALIZE = 'qwen3_localize'
    QWEN3_LOCALIZE_MULTI = 'qwen3_localize_multi'
    UITARS_LOCALIZE = 'uitars_localize'

    @override
    def get_model_name(self) -> str:
        match self:
            case GroundingBenchmarkType.QWEN3_CLASSIFY | GroundingBenchmarkType.QWEN3_LOCALIZE | GroundingBenchmarkType.QWEN3_LOCALIZE_MULTI:
                # return "qwen/qwen3-vl-8b-instruct"
                return "qwen/qwen3-vl-235b-a22b-instruct"
            case GroundingBenchmarkType.UITARS_LOCALIZE:
                return "bytedance/ui-tars-1.5-7b"
            case _:
                raise ValueError(f"Benchmark type not implemented: {self}")

    @override
    def get_system_prompt(self) -> str:
        match self:
            case GroundingBenchmarkType.QWEN3_CLASSIFY:
                return QWEN3_CLASSIFY_SYSTEM_PROMPT
            case GroundingBenchmarkType.QWEN3_LOCALIZE:
                return QWEN3_LOCALIZE_SYSTEM_PROMPT
            case GroundingBenchmarkType.QWEN3_LOCALIZE_MULTI:
                return QWEN3_MULTILOCALIZE_SYSTEM_PROMPT
            case GroundingBenchmarkType.UITARS_LOCALIZE:
                return UITARS_LOCALIZE_SYSTEM_PROMPT
            case _:
                raise ValueError(f"Benchmark type not implemented: {self}")

    @override
    def get_relevant_files(self) -> list[Path]:
        base_path = Path("benchmark1_grounding/examples")
        single_object = base_path / "single_object"
        multi_object = base_path / "multi_object"
        
        match self:
            case GroundingBenchmarkType.QWEN3_CLASSIFY:
                folders = [single_object]
            case GroundingBenchmarkType.QWEN3_LOCALIZE:
                folders = [single_object, multi_object]
            case GroundingBenchmarkType.QWEN3_LOCALIZE_MULTI:
                folders = [multi_object]
            case GroundingBenchmarkType.UITARS_LOCALIZE:
                folders = [single_object, multi_object]
            case _:
                raise ValueError(f"Benchmark type not implemented: {self}")
        
        file_paths = []
        for folder in folders:
            file_paths.extend([p for p in folder.glob("*.json")])
        return sorted(file_paths)


def parse_ground_truth(json_path:Path, benchmark_type: GroundingBenchmarkType):
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
            case GroundingBenchmarkType.QWEN3_CLASSIFY:
                part_name = parse_classification(part)
                return part_name if part_name else "NONE"
            case GroundingBenchmarkType.QWEN3_LOCALIZE | GroundingBenchmarkType.UITARS_LOCALIZE | GroundingBenchmarkType.QWEN3_LOCALIZE_MULTI:
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

def evaluate_response_bbox(ground_truth: BoundingBox, response: BoundingBox, check_label=True):
    # calc IoU for bbox and compare labels
    if check_label and ground_truth.label != response.label:
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

def evaluate_response_bboxes_distance(ground_truth: list[BoundingBox], response: list[BoundingBox]):
    # calc average distance for multiple bboxes, matching by label
    if not ground_truth or not response:
        return None
    total_distance = 0.0
    matched = 0
    for gt in ground_truth:
        for resp in response:
            # Match by label before calculating distance
            if gt.label == resp.label:
                distance = gt.center().euclidian_distance_to(resp.center())
                total_distance += distance
                matched += 1
                break
    if matched == 0:
        return None
    average_distance = total_distance / matched
    return average_distance

def evaluate_response_bbox_distance(ground_truth: BoundingBox, response: BoundingBox):
    # calc distance for bbox centers
    if not ground_truth or not response:
        return None
    distance = ground_truth.center().euclidian_distance_to(response.center())
    return distance

if __name__ == "__main__":
    cli = BenchmarkCli(name="benchmark1_grounding", benchmark_types=list(GroundingBenchmarkType))
    benchmark = GroundingBenchmarkType(cli.benchmark)
    benchmark_files = benchmark.get_relevant_files()

    if not cli.model:
        model_name = benchmark.get_model_name()
    system_prompt = benchmark.get_system_prompt()

    pbar = tqdm(benchmark_files, desc="Processing files", unit="file")
    for file_path in pbar:
        pbar.set_description(f"Processing: {file_path.name}")
        input_png = file_path.with_suffix(".png")
        input_json = file_path.with_suffix(".json")

        image_width, image_height = get_image_dimensions(input_png)

        ground_truth = parse_ground_truth(input_json, benchmark)

        response = None
        score = None
        result: SingleTaskResult

        match benchmark:
            case GroundingBenchmarkType.QWEN3_CLASSIFY:
                assert isinstance(ground_truth, str)
                if not cli.model:
                    cli.model = Qwen3VLLLMWrapper(api_key=cli.API_KEY, base_url=cli.BASE_URL, model_name=model_name)
                response = cli.model.generate_model_response(input_png, system_prompt=system_prompt) or ""
                response = cli.model.parse_response_text(response)
                score = evaluate_response(ground_truth, response)

                result = SingleTaskResult(
                    benchmark_type=benchmark.value,
                    model=cli.model.model_name,
                    final_score=score,
                    iou=None,
                    classification_correct=score,
                    distance=None,
                    score_formula="exact match",
                    input_file=str(file_path),
                    ground_truth=ground_truth,
                    user_prompt=None,
                    response=response
                )

            case GroundingBenchmarkType.QWEN3_LOCALIZE:
                assert isinstance(ground_truth, list)
                assert len(ground_truth) >= 1
                ground_truth_bbox = ground_truth[0]
                additional_user_prompt = f"Locate the {ground_truth_bbox.label}"
                if not cli.model:
                    cli.model = Qwen3VLLLMWrapper(api_key=cli.API_KEY, base_url=cli.BASE_URL, model_name=model_name)
                response = cli.model.generate_model_response(input_png, system_prompt=system_prompt, additional_user_prompt=additional_user_prompt) or ""

                parsed_response = cli.model.parse_response_bbox(response, image_height=image_height, image_width=image_width)
                iou = 0 if parsed_response is None else evaluate_response_bbox(ground_truth_bbox, parsed_response)
                distance = 0 if parsed_response is None else evaluate_response_bbox_distance(ground_truth_bbox, parsed_response)

                result = SingleTaskResult(
                    benchmark_type=benchmark.value,
                    model=cli.model.model_name,
                    final_score=-1,
                    iou=iou,
                    classification_correct=None,
                    distance=distance,
                    score_formula="IoU",
                    input_file=str(file_path),
                    ground_truth=ground_truth_bbox,
                    user_prompt=additional_user_prompt,
                    response=parsed_response if parsed_response is not None else response
                )

            case GroundingBenchmarkType.QWEN3_LOCALIZE_MULTI:
                assert isinstance(ground_truth, list)
                ground_truth_bbox = ground_truth
                assert len(ground_truth_bbox) >= 1
                
                # Generate combinations: single objects, pairs, and all objects
                n_objects = len(ground_truth_bbox)
                all_combos = []
                
                # Single objects
                all_combos.extend([(1, combo) for combo in combinations(range(n_objects), 1)])
                
                # Pairs (only if n_objects >= 2)
                if n_objects >= 2:
                    all_combos.extend([(2, combo) for combo in combinations(range(n_objects), 2)])
                
                # All objects together (only if n_objects > 2, otherwise it's already covered by pairs)
                if n_objects > 2:
                    all_combos.append((n_objects, tuple(range(n_objects))))
                
                combo_pbar = tqdm(all_combos, desc=f"Testing combinations for {file_path.name}", unit="combo", leave=False)
                for r, combo in combo_pbar:
                    # Get the subset of ground truth bboxes for this combination
                    combo_ground_truth = [ground_truth_bbox[i] for i in combo]
                    object_list = ", ".join([gt.label for gt in combo_ground_truth if gt.label is not None])
                    additional_user_prompt = f"Locate the following objects: {object_list}"
                    combo_pbar.set_description(f"{file_path.name} - {len(combo_ground_truth)} obj(s): {object_list[:30]}")
                    if not cli.model:
                        cli.model = Qwen3VLLLMWrapper(api_key=cli.API_KEY, base_url=cli.BASE_URL, model_name=model_name)
                    response = cli.model.generate_model_response(input_png, system_prompt=system_prompt, additional_user_prompt=additional_user_prompt) or ""
                    parsed_response = cli.model.parse_response_bboxes(response, image_height=image_height, image_width=image_width)
                    iou = evaluate_response_bboxes(combo_ground_truth, parsed_response)
                    distance = evaluate_response_bboxes_distance(combo_ground_truth, parsed_response)

                    result = SingleTaskResult(
                        benchmark_type=benchmark.value,
                        model=cli.model.model_name,
                        final_score=-1,
                        iou=iou,
                        classification_correct=None,
                        distance=distance,
                        score_formula="IoU and distance",
                        input_file=str(file_path),
                        ground_truth=combo_ground_truth,
                        user_prompt=additional_user_prompt,
                        response=parsed_response
                    )
                    if cli.save:
                        cli.save_result(result)
                
                # Skip the default result append at the end for this case
                continue

            case GroundingBenchmarkType.UITARS_LOCALIZE:
                assert isinstance(ground_truth, list)
                assert len(ground_truth) >= 1
                ground_truth_bbox = ground_truth[0]
                additional_user_prompt = f"Click the {ground_truth_bbox.label}"
                if not cli.model:
                    cli.model = UiTarsLLMWrapper(api_key=cli.API_KEY, base_url=cli.BASE_URL, model_name=model_name)
                response = cli.model.generate_model_response(input_png, system_prompt=system_prompt, additional_user_prompt=additional_user_prompt) or ""
                parsed_response_tuple = cli.model.parse_response_point(response)
                parsed_response = Point(x=parsed_response_tuple[0], y=parsed_response_tuple[1])
                if parsed_response.x < 0 or parsed_response.y < 0:
                    distance = None
                else:
                    distance = ground_truth_bbox.center().euclidian_distance_to(parsed_response)

                result = SingleTaskResult(
                    benchmark_type=benchmark.value,
                    model=cli.model.model_name,
                    final_score=distance if distance is not None else -1,
                    iou=None,
                    classification_correct=None,
                    distance=distance,
                    score_formula="euclidean distance",
                    input_file=str(file_path),
                    ground_truth=ground_truth_bbox,
                    user_prompt=additional_user_prompt,
                    response=parsed_response
                )

            case _:
                raise ValueError(f"Benchmark type not implemented: {benchmark}")

        if cli.save:
            cli.save_result(result)
