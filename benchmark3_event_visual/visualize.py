import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
ROOT_DIR = 'results'
OUTPUT_FILE = 'benchmark_results_plot_visual.png'

# Restore the style you liked
plt.style.use('ggplot')


def parse_benchmark_results(root_dir):
    results = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(root, file)

            try:
                with open(file_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        # 1. Basic Validation
                        if len(row) < 3 or not row[0].startswith("20"):
                            continue

                        # 2. Extract Metadata
                        raw_category = row[1].strip()
                        model_name = row[2].strip().split('/')[-1]

                        # Check for Confusion Task (Ground Truth is None)
                        is_confusion_task = False
                        if len(row) > 9:
                            ground_truth = row[9].strip()
                            if ground_truth == "None":
                                is_confusion_task = True

                        score = np.nan
                        distance = np.nan
                        category_label = None

                        # --- CATEGORY LOGIC ---

                        # Case A: Detect Confusion (Object is missing)
                        if is_confusion_task:
                            # Append "(Confusion)" to the category name so it appears as a separate bar
                            category_label = f"{raw_category} (Confusion)"

                            # Score is Accuracy (Did it predict None?)
                            try:
                                is_correct = row[5].strip() == "True"
                                score = 100.0 if is_correct else 0.0
                            except:
                                score = 0.0

                            # Distance is not applicable for confusion tasks
                            distance = np.nan

                        # Case B: Localization (Object is present)
                        else:
                            # Keep the original category name (e.g., cause_visual)
                            category_label = raw_category

                            # Metric 1: IoU (Score)
                            try:
                                val_iou = float(row[4])
                                score = val_iou * 100.0 if val_iou >= 0 else 0.0
                            except:
                                pass

                            # Metric 2: Distance (Error)
                            try:
                                d_val = -1.0
                                d1 = float(row[3])
                                d2 = float(row[6]) if len(row) > 6 else -1.0
                                if d1 >= 0:
                                    d_val = d1
                                elif d2 >= 0:
                                    d_val = d2

                                # Constraint: Ignore Distance if 0 (invalid/no box)
                                if d_val > 0:
                                    distance = d_val
                                else:
                                    distance = np.nan
                            except:
                                pass

                        if category_label:
                            results.append({
                                "Model": model_name,
                                "Category": category_label,
                                "Score": score,
                                "Distance": distance
                            })
            except Exception as e:
                print(f"Skipping {file}: {e}")

    return pd.DataFrame(results)


def plot_benchmark(df):
    if df.empty:
        print("No data found.")
        return

    # Aggregate Means
    df_agg = df.groupby(['Model', 'Category']).mean(numeric_only=True).reset_index()

    # --- COLOR MAPPING ---
    # Assign a fixed color to each unique category
    default_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    unique_categories = sorted(df_agg['Category'].unique())
    category_colors = {
        cat: default_cycle[i % len(default_cycle)]
        for i, cat in enumerate(unique_categories)
    }

    # Pivot Data
    pivot_score = df_agg.pivot(index='Model', columns='Category', values='Score')
    pivot_dist = df_agg.pivot(index='Model', columns='Category', values='Distance')

    # --- SORTING LOGIC ---
    # Sort by Mean Score Ascending (Low -> High)
    model_order = pivot_score.mean(axis=1).sort_values(ascending=True).index

    pivot_score = pivot_score.reindex(model_order)
    pivot_dist = pivot_dist.reindex(model_order)

    # Setup Subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # --- PLOT 1: SCORES (Accuracy / IoU) ---
    colors_top = [category_colors[col] for col in pivot_score.columns]
    pivot_score.plot(kind='bar', ax=ax1, width=0.7, color=colors_top,
                     edgecolor='white', linewidth=0.7)

    ax1.set_title('Model Performance: Visual Grounding & Confusion', fontsize=12, fontweight='bold', pad=15)
    ax1.set_ylabel('Accuracy (%) / IoU (%)')
    ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=1, title="Task Type")
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f', padding=3, fontsize=9)

    # --- PLOT 2: DISTANCE (Localization Error) ---
    dist_cols = [c for c in pivot_dist.columns if pivot_dist[c].sum() > 0]

    if dist_cols:
        colors_bottom = [category_colors[col] for col in dist_cols]
        pivot_dist[dist_cols].plot(kind='bar', ax=ax2, width=0.7, color=colors_bottom,
                                   edgecolor='white', linewidth=0.7)

        ax2.set_title('Localization Error', fontsize=12, fontweight='bold', pad=15)
        ax2.set_ylabel('Mean Distance (Pixels)')
        ax2.grid(axis='y', linestyle='--', alpha=0.6)

        # Legend consistent with top graph
        ax2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=1, title="Task Type")

        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.1f', padding=3, fontsize=9)
    else:
        ax2.text(0.5, 0.5, "No Distance Metrics Available", ha='center', va='center')

    # Final Layout
    plt.xlabel('Model', fontsize=11, fontweight='bold', labelpad=10)
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Plot saved to {OUTPUT_FILE}")
    print("\nSummary Data (Sorted):")
    print(pivot_score)


# --- EXECUTION ---
df_results = parse_benchmark_results(ROOT_DIR)
plot_benchmark(df_results)