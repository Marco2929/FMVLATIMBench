from abc import abstractmethod
import argparse
import csv
from enum import Enum
import os
from pathlib import Path
import time
from typing import Dict, Optional

from src.llm_wrapper import GrokLLMWrapper, LLMWrapperBase, Qwen25VLLLMWrapper, Qwen3VLLLMWrapper, OpenAILLMWrapper, GeminiLLMWrapper, UiTarsLLMWrapper
from src.results_model import SingleTaskResult

class BenchmarkBase(Enum):
    @abstractmethod
    def get_model_name(self) -> str:
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def get_system_prompt(self) -> str:
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def get_relevant_files(self) -> list[Path]:
        raise NotImplementedError("This method should be implemented by subclasses.")


class BenchmarkCli:
    benchmark: str
    save: bool
    results: list[SingleTaskResult] = []
    model: LLMWrapperBase|None = None

    def __init__(self, name: str, benchmark_types: list[BenchmarkBase]):
        assert ' ' not in name, "name should not contain spaces"
        self.openrouter_model_list = ['qwen/qwen3-vl-235b-a22b-instruct',
                      'qwen/qwen3-vl-8b-instruct',
                      'bytedance/ui-tars-1.5-7b',
                      'qwen/qwen-2.5-vl-7b-instruct']
        self.openai_model_list = ['gpt-5-mini']
        self.gemini_model_list = ['gemini-2.5-flash']
        self.hyperbolic_model_list = ['Qwen/Qwen2.5-VL-7B-Instruct']
        self.xai_model_list = ['grok-4-1-fast-non-reasoning']
        self.name = name
        self.parser = argparse.ArgumentParser(description=name)
        self.parser.add_argument("--nosave", action="store_true", help="Flag to not save the results in results/*.csv at the end.")
        self.parser.add_argument(
            "--benchmark",
            type=str,
            choices=[bt.value for bt in benchmark_types],
            required=True,
            help="Benchmark type with combination of model and task.",
        )
        self.parser.add_argument(
            "--model",
            type=str,
            choices=self.openrouter_model_list + self.gemini_model_list + self.openai_model_list + self.hyperbolic_model_list + self.xai_model_list,
            required=False,
            help="Optional model name override.",
        )
        args = self.parser.parse_args()
        self.args = args
        self.benchmark = args.benchmark
        assert isinstance(self.benchmark, str)
        self.save = not args.nosave
        assert isinstance(self.save, bool)

        self.model_name = args.model

        if args.model:
            print(f"Using model: {args.model}")

            if args.model in self.openrouter_model_list:
                self.API_KEY = get_api_keys('OPENROUTER_API_KEY')
                self.BASE_URL = get_base_url('BASE_URL')
                if 'ui-tars' in args.model:
                    self.model = UiTarsLLMWrapper(
                        api_key=self.API_KEY,
                        base_url=self.BASE_URL,
                        model_name=args.model
                    )
                elif 'qwen3' in args.model:
                    self.model = Qwen3VLLLMWrapper(
                        api_key=self.API_KEY,
                        base_url=self.BASE_URL,
                        model_name=args.model
                    )
                elif 'qwen-2.5' in args.model:
                    self.model = Qwen25VLLLMWrapper(
                        api_key=self.API_KEY,
                        base_url=self.BASE_URL,
                        model_name=args.model
                    )
                else:
                    raise ValueError(f"Model not implemented: {args.model}")
            elif args.model in self.openai_model_list:
                self.API_KEY = get_api_keys('OPENAI_API_KEY')
                self.model = OpenAILLMWrapper(
                    api_key=self.API_KEY,
                    base_url=None,
                    model_name=args.model
                )
            elif args.model in self.gemini_model_list:
                self.API_KEY = get_api_keys('GEMINI_API_KEY')
                self.BASE_URL = get_base_url('GEMINI_BASE_URL')
                self.model = GeminiLLMWrapper(
                    api_key=self.API_KEY,
                    base_url=self.BASE_URL,
                    model_name=args.model
                )
            elif args.model in self.hyperbolic_model_list:
                self.API_KEY = get_api_keys('HYPERBOLIC_API_KEY')
                self.BASE_URL = get_base_url('HYPERBOLIC_BASE_URL')
                if 'Qwen2.5' in args.model:
                    self.model = Qwen25VLLLMWrapper(
                        api_key=self.API_KEY,
                        base_url=self.BASE_URL,
                        model_name=args.model
                    )
                else:
                    raise ValueError(f"Model not implemented: {args.model}")
            elif args.model in self.xai_model_list:
                self.API_KEY = get_api_keys('XAI_API_KEY')
                self.BASE_URL = get_base_url('XAI_BASE_URL')
                if 'grok' in args.model:
                    self.model = GrokLLMWrapper(
                        api_key=self.API_KEY,
                        base_url=self.BASE_URL,
                        model_name=args.model
                    )
                else:
                    raise ValueError(f"Model not implemented: {args.model}")
            else:
                raise ValueError(f"Model not implemented: {args.model}")

        if self.save:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"{self.name}_results_{timestamp}.csv")
            results_dir = Path(f'{self.name}/results/{self.benchmark}')
            if args.model:
                results_dir = results_dir / args.model.split('/')[-1]
            results_dir.mkdir(parents=True, exist_ok=True)
            self.output_path = results_dir / output_path

    def save_result(self, result: SingleTaskResult):
        self.results.append(result)
        if not self.save:
            return
        file_exists = self.output_path.exists()
        with open(self.output_path, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=SingleTaskResult.get_fieldnames())
            if not file_exists:
                writer.writeheader()
            writer.writerow(result.to_dict())

    def save_results(self):
        if not self.save or not self.results:
            return
        with open(self.output_path, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=SingleTaskResult.get_fieldnames())
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.to_dict())
        print(f"\nAll results saved to {self.output_path}")


def get_api_keys(key_adress) -> str:
    API_KEY = os.getenv(key_adress)
    if API_KEY:
        return API_KEY
    else:
        try:
            with open('.env', 'r') as f:
                for line in f:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
            API_KEY = os.getenv(key_adress)
            if API_KEY is None:
                raise ValueError(f"Please set the {key_adress} environment variable (e.g. in .env)")
        except FileNotFoundError:
            raise ValueError(f"Please set the {key_adress} environment variable (e.g. in .env)")
    return API_KEY

def get_base_url(base_url) -> str:
    BASE_URL = os.getenv(base_url)
    if BASE_URL:
        return BASE_URL
    else:
        try:
            with open('.env', 'r') as f:
                for line in f:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
            BASE_URL = os.getenv(base_url)
            if BASE_URL is None:
                raise ValueError(f"Please set the {base_url} environment variable (e.g. in .env)")
        except FileNotFoundError:
            raise ValueError(f"Please set the {base_url} environment variable (e.g. in .env)")
    return BASE_URL