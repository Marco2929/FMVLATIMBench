import json
import os
from pathlib import Path

folder = "/home/mm/dev/git/FoundationModelsVLA/benchmark3_event/examples/outcome_text"

def extract_number(name):
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 999999

def combine_json(folder, output):
    folder = Path(folder)
    result = {}

    paths = sorted(folder.glob("*.json"), key=lambda p: extract_number(p.stem))

    for path in paths:
        with path.open("r", encoding="utf8") as f:
            data = json.load(f)
        result[path.stem] = data

    with open(output, "w", encoding="utf8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    output = os.path.join(folder, "combined.json")
    combine_json(folder, output)
