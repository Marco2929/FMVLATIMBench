from pathlib import Path
from typing import override

from benchmark2_understanding.system_prompts.qwen3vl_object_property_ident_with_instruct import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_WITH
from benchmark2_understanding.system_prompts.qwen3vl_object_property_ident_without_instruct import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_WITHOUT
from benchmark2_understanding.system_prompts.qwen3vl_object_state_ident import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_STATE_IDENT
from src.bechmark_base import BenchmarkBase, BenchmarkCli
from src.llm_wrapper import Qwen3VLLLMWrapper
from src.results_model import SingleTaskResult
from utils import parse_ground_truth, parse_response, evaluate_response

class UnderstandingBenchmarkType(BenchmarkBase):
    WITH_INSTRUCT = 'with_instruct'
    WITHOUT_INSTRUCT = 'without_instruct'
    STATE_IDENT = 'state_ident'

    @override
    def get_model_name(self) -> str:
        match self:
            case _:
                return "qwen/qwen3-vl-235b-a22b-instruct"

    @override
    def get_system_prompt(self) -> str:
        match self:
            case UnderstandingBenchmarkType.WITH_INSTRUCT:
                return SYSTEM_PROMPT_WITH
            case UnderstandingBenchmarkType.WITHOUT_INSTRUCT:
                return SYSTEM_PROMPT_WITHOUT
            case UnderstandingBenchmarkType.STATE_IDENT:
                return SYSTEM_PROMPT_STATE_IDENT
            case _:
                raise ValueError(f"Benchmark type not implemented: {self}")

    @override
    def get_relevant_files(self) -> list[Path]:
        base_path = Path("benchmark2_understanding/examples")
        proptery_indent = base_path / "object_property_ident"
        state_indent = base_path / "object_state_ident"
        
        match self:
            case UnderstandingBenchmarkType.WITH_INSTRUCT:
                folders = [proptery_indent]
            case UnderstandingBenchmarkType.WITHOUT_INSTRUCT:
                folders = [proptery_indent]
            case UnderstandingBenchmarkType.STATE_IDENT:
                folders = [state_indent]
            case _:
                raise ValueError(f"Benchmark type not implemented: {self}")
        
        file_paths = []
        for folder in folders:
            file_paths.extend([p for p in folder.glob("*.txt")])
        return sorted(file_paths)

if __name__ == "__main__":
    cli = BenchmarkCli(name="benchmark2_understanding", benchmark_types=list(UnderstandingBenchmarkType))
    benchmark = UnderstandingBenchmarkType(cli.benchmark)
    benchmark_files = benchmark.get_relevant_files()

    model_name = benchmark.get_model_name()
    system_prompt = benchmark.get_system_prompt()

    for file_path in benchmark_files:
        input_png = file_path.with_suffix(".png").resolve()
        input_json = file_path.with_suffix(".json").resolve()
        input_prompt = file_path.with_suffix(".txt").resolve()

        with open(input_prompt, 'r') as f:
            user_prompt = f.read().strip()
            
        model = cli.model or Qwen3VLLLMWrapper(api_key=cli.API_KEY, base_url=cli.BASE_URL, model_name=model_name)

        ground_truth = parse_response(parse_ground_truth(input_json))
        response = model.generate_model_response(input_png, system_prompt, additional_user_prompt=user_prompt) or ""
        response = parse_response(response)
        score = evaluate_response(ground_truth, response)
        print(f"Ground Truth: {ground_truth}")
        print(f"Parsed Response: {response}")
        print(f"Evaluation Score: {score}")

        result = SingleTaskResult(
            benchmark_type=benchmark.value,
            model=benchmark.get_model_name(),
            final_score=score,
            iou=None,
            classification_correct=score,
            distance=None,
            score_formula="exact match",
            input_file=str(file_path),
            ground_truth=ground_truth,
            user_prompt=user_prompt,
            response=response,
        )

        cli.results.append(result)
    
    if cli.save:
        cli.save_results()