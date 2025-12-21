import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
ROOT_DIR = 'results'
DISTANCE_THRESHOLD = 50.0  # Pixels: Distance < 50px counts as success


def parse_benchmark_results(root_dir):
    results = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(root, file)

            try:
                # newline='' is required for the csv module to handle quotes correctly
                with open(file_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)

                    for row in reader:
                        # 1. Basic Validation
                        # Ensure row has enough columns and starts with a timestamp (skips headers/garbage)
                        if len(row) < 3 or not row[0].startswith("20"):
                            continue

                        # 2. Extract Metadata
                        category_raw = row[1].strip()
                        model_raw = row[2].strip()
                        model_name = model_raw.split('/')[-1]  # Clean model name

                        score = 0.0
                        metric_type = "Unknown"
                        category_label = None

                        # 3. Extract Metrics based on Category

                        # --- CLASSIFY ---
                        if "classify" in category_raw:
                            category_label = "Classify"
                            # Col 3 is boolean string "True"/"False"
                            is_correct = row[3].strip() == "True"
                            score = 100.0 if is_correct else 0.0
                            metric_type = "Accuracy"

                        # --- LOCALIZE MULTI ---
                        elif "localize_multi" in category_raw:
                            category_label = "Localize Multi"
                            try:
                                # Col 4 is IoU
                                score = float(row[4]) * 100.0
                                metric_type = "Mean IoU"
                            except (ValueError, IndexError):
                                continue

                        # --- LOCALIZE (SINGLE) ---
                        elif "localize" in category_raw:
                            category_label = "Localize"

                            # Check if it's Distance-based (UITars style) or IoU-based (Qwen style)
                            # We check if the metric name (usually near the end, col 7) mentions 'euclidean'
                            # Or if the IoU column (col 4) is -1 or empty.

                            is_distance_metric = False
                            if len(row) > 7 and "euclidean" in row[7].lower():
                                is_distance_metric = True

                            if is_distance_metric:
                                try:
                                    # Col 3 is Distance
                                    dist = float(row[3])
                                    score = 100.0 if dist < DISTANCE_THRESHOLD else 0.0
                                    metric_type = f"Success Rate (<{DISTANCE_THRESHOLD}px)"
                                except (ValueError, IndexError):
                                    continue
                            else:
                                try:
                                    # Col 4 is IoU
                                    score = float(row[4]) * 100.0
                                    metric_type = "Mean IoU"
                                except (ValueError, IndexError):
                                    continue

                        if category_label:
                            results.append({
                                "Model": model_name,
                                "Category": category_label,
                                "Score": score,
                                "Metric Type": metric_type
                            })

            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    return pd.DataFrame(results)


# --- EXECUTION ---
df = parse_benchmark_results(ROOT_DIR)

if not df.empty:
    # Aggregate: Mean Score per Model/Category
    df_agg = df.groupby(['Model', 'Category'])['Score'].mean().reset_index()

    # Pivot: Model as Index, Categories as Columns
    df_pivot = df_agg.pivot(index='Model', columns='Category', values='Score')

    # Plot
    ax = df_pivot.plot(kind='bar', figsize=(12, 6), width=0.8)

    plt.title('Model Accuracy Grounding Classify vs Localize vs Localize Multi')
    plt.ylabel('Score (Accuracy % / IoU % / Success Rate %)')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Category')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save
    output_file = 'benchmark_results_plot_grounding.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

    print("\nCorrected Aggregated Results:")
    print(df_agg)
else:
    print("No valid data found.")