from pathlib import Path
from typing import List, override

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.models.base_model import DeepEvalBaseLLM

from deepeval_openrouter_utils import OpenRouterLLM
from benchmark3_event.system_prompts.qwen3vl_outcome_text import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_OUTCOME_TEXT
from benchmark3_event.system_prompts.qwen3vl_cause_text import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_CAUSE_TEXT
from benchmark3_event.system_prompts.qwen3vl_effect_text import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_EFFECT_TEXT

from src.bechmark_base import BenchmarkBase, BenchmarkCli
from src.llm_wrapper import Qwen3VLLLMWrapper
from src.results_model import SingleTaskResult
from utils import parse_ground_truth

class EventBenchmarkType(BenchmarkBase):
    OUTCOME_TEXT = 'outcome_text'
    EFFECT_TEXT = 'effect_text'
    CAUSE_TEXT = 'cause_text'
    
    @override
    def get_model_name(self) -> str:
        match self:
            case _:
                return "qwen/qwen3-vl-235b-a22b-instruct"
            
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
            file_paths.extend([p for p in folder.glob("*.txt")])
        return sorted(file_paths)

def evaluate_response_deep_eval(openrouter_llm: DeepEvalBaseLLM, input: str, expected_output: str, actual_output: str,
                                retrieval_context: List[str]):
    # Initialize Metric with the custom model
    metric = ContextualPrecisionMetric(
        threshold=0.9,
        model=openrouter_llm,
        include_reason=True,
        strict_mode = True
    )

    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context
    )

    results = evaluate(test_cases=[test_case], metrics=[metric])

    return results.test_results[0].success


if __name__ == "__main__":
    cli = BenchmarkCli(name="benchmark3_event_visual", benchmark_types=list(EventBenchmarkType))
    benchmark = EventBenchmarkType(cli.benchmark)
    benchmark_files = benchmark.get_relevant_files()

    model_name = benchmark.get_model_name()
    system_prompt = benchmark.get_system_prompt()

    openrouter_llm = OpenRouterLLM(
        model_name=model_name,
        api_key=cli.API_KEY,
    )

    model = cli.model or Qwen3VLLLMWrapper(api_key=cli.API_KEY, base_url=cli.BASE_URL, model_name=model_name)

    for file_path in benchmark_files:
        input_png = file_path.with_suffix(".png").resolve()
        input_json = file_path.with_suffix(".json").resolve()
        input_prompt = file_path.with_suffix(".txt").resolve()

        with open(input_prompt, 'r') as f:
            user_prompt = f.read().strip()


        ground_truth = parse_ground_truth(input_json)

        # response = generate_model_response(input_png, model_name="qwen/qwen3-vl-30b-a3b-instruct") or ""
        response = model.generate_model_response(input_png, system_prompt=system_prompt, additional_user_prompt=user_prompt) or ""
        score = evaluate_response_deep_eval(openrouter_llm=openrouter_llm, input=system_prompt, expected_output=ground_truth,
                                            actual_output=response, retrieval_context=[user_prompt])
        print(f"Task: {user_prompt}")
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
