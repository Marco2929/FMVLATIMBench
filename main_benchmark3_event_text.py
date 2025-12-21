import time
from pathlib import Path
from typing import List, override

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.models.base_model import DeepEvalBaseLLM

from llm_evaluator import llm_evaluate
from benchmark3_event.system_prompts.qwen3vl_outcome_text import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_OUTCOME_TEXT
from benchmark3_event.system_prompts.qwen3vl_cause_text import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_CAUSE_TEXT
from benchmark3_event.system_prompts.qwen3vl_effect_text import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_EFFECT_TEXT

from src.benchmark_base import BenchmarkBase, BenchmarkCli
from src.llm_wrapper import Qwen3VLLLMWrapper
from src.results_model import SingleTaskResult
from utils import load_json


class EventBenchmarkType(BenchmarkBase):
    OUTCOME_TEXT = 'outcome_text'
    EFFECT_TEXT = 'effect_text'
    CAUSE_TEXT = 'cause_text'
    
    @override
    def get_model_name(self) -> str:
        pass
            
    @override
    def get_system_prompt(self) -> str:
        match self:
            case EventBenchmarkType.OUTCOME_TEXT:
                return SYSTEM_PROMPT_OUTCOME_TEXT
            case EventBenchmarkType.EFFECT_TEXT:
                return SYSTEM_PROMPT_EFFECT_TEXT
            case EventBenchmarkType.CAUSE_TEXT:
                return SYSTEM_PROMPT_CAUSE_TEXT
            case _:
                raise ValueError(f"Benchmark type not implemented: {self}")
            
    @override
    def get_relevant_files(self) -> list[Path]:
        base_path = Path("benchmark3_event/examples")
        outcome_text = base_path / "outcome_text"
        effect_text = base_path / "effect_text"
        cause_text = base_path / "cause_text"
        
        match self:
            case EventBenchmarkType.OUTCOME_TEXT:
                folders = [outcome_text]
            case EventBenchmarkType.EFFECT_TEXT:
                folders = [effect_text]
            case EventBenchmarkType.CAUSE_TEXT:
                folders = [cause_text]
            case _:
                raise ValueError(f"Benchmark type not implemented: {self}")
        
        file_paths = []
        for folder in folders:
            file_paths.extend([p for p in folder.glob("*.json")])
        return sorted(file_paths)


if __name__ == "__main__":
    cli = BenchmarkCli(name="benchmark3_event_text", benchmark_types=list(EventBenchmarkType))
    benchmark = EventBenchmarkType(cli.benchmark)
    benchmark_files = benchmark.get_relevant_files()

    model_name = benchmark.get_model_name()
    system_prompt = benchmark.get_system_prompt()

    model = cli.model

    for i, file_path in enumerate(benchmark_files):
        input_png = file_path.with_suffix(".png").resolve()
        input_json = file_path.with_suffix(".json").resolve()

        user_prompt, ground_truth = load_json(input_json)

        # response = generate_model_response(input_png, model_name="qwen/qwen3-vl-30b-a3b-instruct") or ""
        response = model.generate_model_response(input_png, system_prompt=system_prompt, additional_user_prompt=user_prompt)

        score = llm_evaluate(input=user_prompt, actual_output=response, expected_output=ground_truth, context=system_prompt)

        print(f"{i+1} Task: {user_prompt}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Parsed Response: {response}")
        print(f"Evaluation Score: {score}")

        result = SingleTaskResult(
            benchmark_type=benchmark.value,
            model=cli.model_name,
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
        # time.sleep(30)
    if cli.save:
        cli.save_results()
