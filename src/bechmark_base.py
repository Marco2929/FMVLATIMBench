from abc import abstractmethod
import argparse
import csv
from enum import Enum
import os
from pathlib import Path
import time

from src.llm_wrapper import LLMWrapperBase, Qwen3VLLLMWrapper
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
            choices=['qwen/qwen3-vl-235b-a22b-instruct'],
            required=False,
            help="Optional model name override.",
        )
        args = self.parser.parse_args()
        self.args = args
        self.benchmark = args.benchmark
        assert isinstance(self.benchmark, str)
        self.save = not args.nosave
        assert isinstance(self.save, bool)

        self.API_KEY = get_api_key()
        self.BASE_URL = get_base_url()

        if args.model:
            print(f"Overriding model name to: {args.model}")
            match args.model:
                case 'qwen/qwen3-vl-235b-a22b-instruct':
                    self.model = Qwen3VLLLMWrapper(api_key=self.API_KEY, base_url=self.BASE_URL, model_name=args.model)
                case _:
                    raise ValueError(f"Model not implemented: {args.model}")

    def save_results(self):
        if not self.save or not self.results:
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"{self.name}_results_{timestamp}.csv")
        results_dir = Path(f'{self.name}/results/{self.benchmark}')
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / output_path
        with open(output_path, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=SingleTaskResult.get_fieldnames())
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.to_dict())
        print(f"\nAll results saved to {output_path}")

def get_api_key() -> str:
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    if API_KEY:
        return API_KEY
    else:
        try:
            with open('.env', 'r') as f:
                for line in f:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
            API_KEY = os.getenv("OPENROUTER_API_KEY")
            if API_KEY is None:
                raise ValueError("Please set the OPENROUTER_API_KEY environment variable (e.g. in .env)")
        except FileNotFoundError:
            raise ValueError("Please set the OPENROUTER_API_KEY environment variable (e.g. in .env)")
    return API_KEY

def get_base_url() -> str:
    BASE_URL = os.getenv("BASE_URL")
    if BASE_URL:
        return BASE_URL
    else:
        try:
            with open('.env', 'r') as f:
                for line in f:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
            BASE_URL = os.getenv("BASE_URL")
            if BASE_URL is None:
                raise ValueError("Please set the OPENROUTER_BASE_URL environment variable (e.g. in .env)")
        except FileNotFoundError:
            raise ValueError("Please set the OPENROUTER_BASE_URL environment variable (e.g. in .env)")
    return BASE_URL