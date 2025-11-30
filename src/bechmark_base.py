from abc import abstractmethod
import argparse
from enum import Enum
import os
from pathlib import Path

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

    def __init__(self, description: str, benchmark_types: list[BenchmarkBase]):
        self.parser = argparse.ArgumentParser(description=description)
        self.parser.add_argument("--save", action="store_true", help="Flag to save the results in <input>.txt")
        self.parser.add_argument(
            "--benchmark",
            type=str,
            choices=[bt.value for bt in benchmark_types],
            required=True,
            help="Benchmark type with combination of model and task.",
        )
        args = self.parser.parse_args()
        self.benchmark = args.benchmark
        assert isinstance(self.benchmark, str)
        self.save = args.save
        assert isinstance(self.save, bool)

        self.API_KEY = get_api_key()
        self.BASE_URL = get_base_url()


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