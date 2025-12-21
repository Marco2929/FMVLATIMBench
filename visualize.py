import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys


def benchmark_textual(category: str, folder: str = "results/"):
    """
    Visualize benchmark results for a specific category.
    Automatically plots all models found in the CSV files separately.

    Args:
        category (str): The benchmark category (e.g., 'understanding', 'event').
        folder (str): Path to folder containing CSV files.
    """
    # 1. Define Mappings
    benchmark_map = {
        "understanding": ["with_instruct", "without_instruct", "state_ident"],
        "event": ["outcome_text", "cause_text", "effect_text"],
    }

    if category not in benchmark_map:
        print(f"Error: Category '{category}' not found. Available: {list(benchmark_map.keys())}")
        sys.exit(1)

    # 2. Load Data
    folder_path = Path(folder)
    csv_files = list(folder_path.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in folder: {folder}")
        return

    # Load and concatenate all CSVs
    data_list = [pd.read_csv(f, parse_dates=["timestamp"]) for f in csv_files]
    data = pd.concat(data_list, ignore_index=True)

    # 3. Validation
    # Ensure 'model' column exists to distinguish the files/models
    if "model" not in data.columns:
        print("Error: The column 'model' is missing from the CSV files.")
        print("Cannot map results to specific models without this column.")
        return

    data["classification_correct_numeric"] = data["classification_correct"].astype(int)

    # 4. Filter by Category
    target_types = benchmark_map[category]
    filtered_data = data[data["benchmark_type"].isin(target_types)].copy()

    if filtered_data.empty:
        print(f"No data found for category '{category}' (checked types: {target_types}).")
        return

    # 5. Aggregate
    # Group by BOTH benchmark_type and model.
    # This separates the scores for every model found in the files.
    agg_data = filtered_data.groupby(["benchmark_type", "model"])[
        "classification_correct_numeric"].mean().reset_index()

    # 6. Plot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # hue="model" creates the separate bars for each model automatically
    ax = sns.barplot(
        data=agg_data,
        x="benchmark_type",
        y="classification_correct_numeric",
        hue="model"
    )

    # Formatting
    ax.set_title(f"Benchmark Accuracy: {category.capitalize()}")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Benchmark Type")
    ax.set_ylim(0, 1)

    # Move legend outside to prevent blocking data if there are many models
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize benchmark results by category.")

    parser.add_argument(
        "category",
        type=str,
        choices=["understanding", "event"],
        help="The benchmark category to visualize."
    )

    parser.add_argument(
        "--folder",
        type=str,
        default="results/",
        help="Folder containing CSV benchmark results."
    )

    args = parser.parse_args()
    benchmark_textual(category=args.category, folder=args.folder)