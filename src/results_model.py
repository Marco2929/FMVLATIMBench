from dataclasses import dataclass, field, asdict
from datetime import datetime

from .llm_wrapper import BoundingBox, Point

@dataclass
class SingleTaskResult:
    # Metadata
    benchmark_type: str
    model: str

    # Scores
    final_score: float
    iou: float | None
    classification_correct: bool | None
    distance: float | None

    # Details
    score_formula: str
    input_file: str
    ground_truth: str | BoundingBox | list[BoundingBox]
    user_prompt: str | None
    response: str | BoundingBox | list[BoundingBox] | Point

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert the result to a dictionary suitable for CSV writing."""
        data = asdict(self)
        data['ground_truth'] = str(data['ground_truth'])
        data['response'] = str(data['response'])
        return data

    @staticmethod
    def get_fieldnames() -> list[str]:
        """Return the field names for CSV headers."""
        return [
            'timestamp', 'benchmark_type', 'model', 'final_score',
            'iou', 'classification_correct', 'distance', 'score_formula',
            'input_file', 'ground_truth', 'response', 'user_prompt'
        ]
