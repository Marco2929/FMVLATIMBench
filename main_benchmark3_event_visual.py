from pathlib import Path
from typing import override

from tqdm import tqdm


from benchmark3_event.system_prompts.qwen3vl_outcome_visual import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_OUTCOME_VISUAL
from benchmark3_event.system_prompts.qwen3vl_cause_visual import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_CAUSE_VISUAL
from benchmark3_event.system_prompts.qwen3vl_effect_visual import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_EFFECT_VISUAL

from src.bechmark_base import BenchmarkBase, BenchmarkCli
from src.image_processing import get_image_dimensions
from src.llm_wrapper import Qwen3VLLLMWrapper
from utils import draw_bounding_box, load_json

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

    model_name = benchmark.get_model_name()
    system_prompt = benchmark.get_system_prompt()

    pbar = tqdm(benchmark_files, desc="Processing files", unit="file")
    for file_path in pbar:
        pbar.set_description(f"Processing: {file_path.name}")
        input_png = file_path.with_suffix(".png").resolve()
        input_json = file_path.with_suffix(".json").resolve()

        model = cli.model or Qwen3VLLLMWrapper(api_key=cli.API_KEY, base_url=cli.BASE_URL, model_name=model_name)

        user_prompt, ground_truth = load_json(input_json)

        image_width, image_height = get_image_dimensions(input_png)

        response = model.generate_model_response(input_png, system_prompt, user_prompt) or ""
        response = model.parse_response_bbox(response, image_width, image_height)
        if response is None:
            print(f"Could not parse response for file: {file_path}")
            continue
        image_with_bbox_path = draw_bounding_box(input_png.with_suffix('.g.png'), response.bbox_list())
        print(f"Task: {user_prompt}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Parsed Response: {response}")
        # print(f"Evaluation Score: {score}")

        # TODO: calculate IoU etc.
        
        # result = SingleTaskResult(
        #     benchmark_type=benchmark.value,
        #     model=benchmark.get_model_name(),
        #     final_score=score,
        #     iou=None,
        #     classification_correct=score,
        #     distance=None,
        #     score_formula="exact match",
        #     input_file=str(file_path),
        #     ground_truth=ground_truth,
        #     user_prompt=user_prompt,
        #     response=response,
        # )

        # cli.results.append(result)
        
    if cli.save:
        cli.save_results()
