import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def benchmark_textual(folder: str = "results/", benchmarks: list[str] = None):
    """
    Visualize benchmark results from CSV files in a folder.

    Args:
        folder (str): Path to folder containing CSV files.
        benchmarks (list[str], optional): List of benchmark types to include. Defaults to None (all).
    """
    folder_path = Path(folder)
    csv_files = list(folder_path.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in folder: {folder}")
        return

    # Load and concatenate all CSV files
    data_list = [pd.read_csv(f, parse_dates=["timestamp"]) for f in csv_files]
    data = pd.concat(data_list, ignore_index=True)

    # Convert boolean columns to numeric
    data["final_score_numeric"] = data["final_score"].astype(int)
    data["classification_correct_numeric"] = data["classification_correct"].astype(int)

    # Map benchmark types
    benchmark_map = {
        "understanding": ["with_instruct", "without_instruct", "state_ident"],
        "event": ["outcome_text", "cause_text", "effect_text"],
        # add more mappings if needed
    }

    if benchmarks:
        mapped_values = []
        for b in benchmarks:
            mapped_values.extend(benchmark_map.get(b.lower(), []))
        data = data[data["benchmark_type"].isin(mapped_values)]

    reverse_map = {}
    for category, subtypes in benchmark_map.items():
        for subtype in subtypes:
            reverse_map[subtype] = category

    data["benchmark_category"] = data["benchmark_type"].map(reverse_map)

    # 2. Aggregate by BOTH category and specific type
    agg_data = data.groupby(["benchmark_category", "benchmark_type"])[
        "classification_correct_numeric"].mean().reset_index()

    # 3. Apply custom ordering to the main category
    benchmark_order = ["understanding", "event", "manipulation", "planning"]
    agg_data["benchmark_category"] = pd.Categorical(
        agg_data["benchmark_category"],
        categories=benchmark_order,
        ordered=True
    )

    # Sort to ensure the plots appear in the correct order
    agg_data = agg_data.sort_values(["benchmark_category", "benchmark_type"])

    # 4. Plot using catplot for hierarchical visualization
    g = sns.catplot(
        data=agg_data,
        x="benchmark_type",
        y="classification_correct_numeric",
        col="benchmark_category",
        kind="bar",
        col_wrap=2,
        height=4,
        aspect=1.2,
        sharex=False,
        sharey=True
    )

    # Adjust titles and labels
    g.set_titles("{col_name}")
    g.set_axis_labels("", "Accuracy")

    # Fix the Y-axis limit for all subplots
    g.set(ylim=(0, 1))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize benchmark results.")
    parser.add_argument(
        "--folder",

        type=str,
        default="results/",
        help="Folder containing CSV benchmark results (default: results/)"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        help="Benchmark types to visualize (space separated, e.g., understanding event manipulation planning)"
    )
    args = parser.parse_args()
    benchmark_textual(folder=args.folder, benchmarks=args.benchmarks)
