import json
from pathlib import Path
from typing import override

from tqdm import tqdm

from benchmark3_event.system_prompts.qwen3vl_outcome_visual import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_OUTCOME_VISUAL
from benchmark3_event.system_prompts.qwen3vl_cause_visual import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_CAUSE_VISUAL
from benchmark3_event.system_prompts.qwen3vl_effect_visual import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_EFFECT_VISUAL

from main_benchmark1_grounding import evaluate_response_bbox, evaluate_response_bbox_distance
from src.benchmark_base import BenchmarkBase, BenchmarkCli
from src.image_processing import get_image_dimensions
from src.llm_wrapper import BoundingBox, Qwen3VLLLMWrapper
from src.results_model import SingleTaskResult
from utils import draw_bounding_box

def load_json_bench3visual(json_path: Path) -> tuple[str, dict]:
    with open(json_path, "r") as f:
        data = json.load(f)
    return data['TASK_DESCRIPTION'], data['solution']

class EventVisualBenchmarkType(BenchmarkBase):
    OUTCOME_VISUAL = 'outcome_visual'
    EFFECT_VISUAL = 'effect_visual'
    CAUSE_VISUAL = 'cause_visual'
    
    @override
    def get_model_name(self) -> str:
        match self:
            case _:
                return "qwen/qwen3-vl-235b-a22b-instruct"
            
    @override
    def get_system_prompt(self) -> str:
        match self:
            case EventVisualBenchmarkType.OUTCOME_VISUAL:
                return SYSTEM_PROMPT_OUTCOME_VISUAL
            case EventVisualBenchmarkType.EFFECT_VISUAL:
                return SYSTEM_PROMPT_EFFECT_VISUAL
            case EventVisualBenchmarkType.CAUSE_VISUAL:
                return SYSTEM_PROMPT_CAUSE_VISUAL
            case _:
                raise ValueError(f"Benchmark type not implemented: {self}")
            
    @override
    def get_relevant_files(self) -> list[Path]:
        base_path = Path("benchmark3_event/examples")
        outcome_visual = base_path / "outcome_visual"
        effect_visual = base_path / "effect_visual"
        cause_visual = base_path / "cause_visual"
        
        match self:
            case EventVisualBenchmarkType.OUTCOME_VISUAL:
                folders = [outcome_visual]
            case EventVisualBenchmarkType.EFFECT_VISUAL:
                folders = [effect_visual]
            case EventVisualBenchmarkType.CAUSE_VISUAL:
                folders = [cause_visual]
            case _:
                raise ValueError(f"Benchmark type not implemented: {self}")
        
        file_paths = []
        for folder in folders:
            file_paths.extend([p for p in folder.glob("*.json")])
        return sorted(file_paths)


if __name__ == "__main__":
    cli = BenchmarkCli(name="benchmark3_event_visual", benchmark_types=list(EventVisualBenchmarkType))
    benchmark = EventVisualBenchmarkType(cli.benchmark)
    benchmark_files = benchmark.get_relevant_files()

    if cli.model is None:
        model_name = benchmark.get_model_name()
    system_prompt = benchmark.get_system_prompt()

    pbar = tqdm(benchmark_files, desc="Processing files", unit="file")
    for file_path in pbar:
        pbar.set_description(f"Processing: {file_path.name}")
        input_png = file_path.with_suffix(".png").resolve()
        input_json = file_path.with_suffix(".json").resolve()

        if cli.model is None:
            cli.model = Qwen3VLLLMWrapper(api_key=cli.API_KEY, base_url=cli.BASE_URL, model_name=model_name)

        user_prompt, ground_truth = load_json_bench3visual(input_json)
        ground_truth_bbox = None if ground_truth is None else BoundingBox(ground_truth['label'], ground_truth['bbox'][0], ground_truth['bbox'][1], ground_truth['bbox'][2], ground_truth['bbox'][3])

        image_width, image_height = get_image_dimensions(input_png)

        response = cli.model.generate_model_response(input_png, system_prompt, user_prompt) or ""
        response = cli.model.parse_response_bbox(response, image_width, image_height)
        if response is not None:
            image_with_bbox_path = draw_bounding_box(input_png, response.bbox_list())
        iou = 0
        if response is not None and ground_truth_bbox is not None:
            iou = evaluate_response_bbox(ground_truth_bbox, response, check_label=False)
        distance = 0
        if response is not None and ground_truth_bbox is not None:
            distance = evaluate_response_bbox_distance(ground_truth_bbox, response)
        classification_correct = False
        if response is None and ground_truth_bbox is None:
            classification_correct = True
        # print(f"Task: {user_prompt}")
        # print(f"Ground Truth: {ground_truth_bbox}")
        # print(f"Parsed Response: {response}")
        # print(f"Evaluation Score: {score}")

        result = SingleTaskResult(
            benchmark_type=benchmark.value,
            model=cli.model.model_name,
            final_score=-1,
            iou=iou,
            classification_correct=classification_correct,
            distance=distance,
            score_formula="iou and distance",
            input_file=str(file_path),
            ground_truth=ground_truth_bbox or 'None',
            user_prompt=user_prompt,
            response=response or 'None',
        )

        if cli.save:
            cli.save_result(result)
