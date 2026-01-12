import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- CONFIGURATION ---
ROOT_DIR = 'results'
OUTPUT_FILE = 'benchmark_results_boxplot.svg'

# 1. Map Raw IDs (from CSV) to Display Names
MODEL_MAPPING = {
    'Qwen2.5-VL-7B-Instruct': 'Qwen2.5 VL',
    'ui-tars-1.5-7b': 'UI-TARS 1.5',
    'qwen3-vl-235b-a22b-instruct': 'Qwen3 VL',
    'gemini-2.5-flash': 'Gemini 2.5 Flash',
    'gpt-5-mini': 'GPT 5 Mini'
}

# 2. Define the exact order for the X-axis
DESIRED_ORDER = [
    'UI-TARS 1.5',
    'Qwen2.5 VL',
    'Qwen3 VL',
    'Gemini 2.5 Flash',
    'GPT 5 Mini'
]

# --- PRESENTATION STYLING ---
# Apply ggplot style first
plt.style.use('ggplot')

# Apply global font scaling for PowerPoint visibility
plt.rcParams.update({
    'font.size': 16,  # Base font size
    'axes.titlesize': 24,  # Main Title
    'axes.labelsize': 20,  # X and Y axis labels
    'xtick.labelsize': 16,  # X-axis tick values
    'ytick.labelsize': 16,  # Y-axis tick values
    'legend.fontsize': 16,  # Legend text
    'figure.titlesize': 26,  # Figure super title
    'font.weight': 'bold',  # Make text bolder generally
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
})


def parse_benchmark_results(root_dir):
    results = []

    # Walk through the directory to read all CSVs
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(root, file)

            try:
                with open(file_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        # Basic validation
                        if len(row) < 3 or not row[0].startswith("20"):
                            continue

                        category_raw = row[1].strip()
                        raw_model_name = row[2].strip().split('/')[-1]

                        # Apply Mapping
                        model_name = MODEL_MAPPING.get(raw_model_name, raw_model_name)

                        score = np.nan
                        distance = np.nan
                        category_label = None

                        # Logic to extract Score (Accuracy/IoU) and Distance (Pixel Error)
                        if "classify" in category_raw:
                            category_label = "Classify"
                            score = 100.0 if row[3].strip() == "True" else 0.0
                        elif "localize" in category_raw:
                            category_label = "Localize Multi" if "multi" in category_raw else "Localize"
                            try:
                                val_iou = float(row[4])
                                score = val_iou * 100.0 if val_iou >= 0 else 0.0
                            except:
                                pass
                            try:
                                d1 = float(row[3])
                                d2 = float(row[6]) if len(row) > 6 else -1.0
                                if d1 >= 0:
                                    distance = d1
                                elif d2 >= 0:
                                    distance = d2
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


def plot_benchmark_boxplot(df):
    if df.empty:
        print("No data found.")
        return

    # Filter Data to only include models in DESIRED_ORDER
    df = df[df['Model'].isin(DESIRED_ORDER)].copy()

    # Define Colors
    unique_categories = sorted(df['Category'].dropna().unique())
    default_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    category_colors = {cat: default_cycle[i % len(default_cycle)] for i, cat in enumerate(unique_categories)}

    # Setup Subplots (2 rows) with slightly more height
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14), sharex=True)

    # --- PLOT 1: SCORES (Accuracy / IoU) - BAR CHART ---
    df_means = df.groupby(['Model', 'Category'], as_index=False)['Score'].mean()

    sns.barplot(
        data=df_means,
        x='Model',
        y='Score',
        hue='Category',
        order=DESIRED_ORDER,
        palette=category_colors,
        ax=ax1,
        edgecolor='white',
        linewidth=1.0,
        errorbar=None
    )

    # Add labels
    for container in ax1.containers:
        # Only show labels > 0
        labels = [f'{v.get_height():.1f}' if v.get_height() > 0 else '' for v in container]
        ax1.bar_label(container, labels=labels, padding=3, fontsize=12, fontweight='bold')

    ax1.set_title('Model Mean Accuracy / IoU', pad=20)
    ax1.set_ylabel('Score (%)', labelpad=15)

    # Legend outside
    ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True, facecolor='white', framealpha=1)

    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax1.set_ylim(0, 110)  # Fix Y axis

    # --- PLOT 2: DISTANCE (Error) - BOXPLOT ---
    df_dist = df[df['Distance'].notna()]

    if not df_dist.empty:
        sns.boxplot(
            data=df_dist,
            x='Model',
            y='Distance',
            hue='Category',
            order=DESIRED_ORDER,
            palette=category_colors,
            ax=ax2,
            linewidth=1.5,  # Thicker lines
            showfliers=False  # Hide outliers for cleaner presentation view
        )

        # Add median values on top of boxplot median lines
        medians_data = df_dist.groupby(['Model', 'Category'])['Distance'].median().reset_index()

        categories = sorted(df_dist['Category'].dropna().unique())
        n_categories = len(categories)

        # Calculate width of a single hue-bar group in seaborn (approx 0.8 total width)
        # Seaborn dodge logic places them centered around integer indices
        total_width = 0.8
        bar_width = total_width / n_categories

        for i, model in enumerate(DESIRED_ORDER):
            for j, category in enumerate(categories):
                median_val = medians_data[(medians_data['Model'] == model) &
                                          (medians_data['Category'] == category)]['Distance']

                if not median_val.empty:
                    val = median_val.values[0]
                    # Calculate offset: j=0 is leftmost, j=n-1 is rightmost relative to center
                    offset = (j - (n_categories - 1) / 2) * bar_width
                    x_pos = i + offset

                    # Place text slightly above the median line
                    ax2.text(x_pos, val, f'{val:.1f}',
                             ha='center', va='bottom',
                             fontsize=11, fontweight='bold',
                             color='black',
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        ax2.set_title('Localization Error Distribution (Lower is Better)', pad=20)
        ax2.set_ylabel('Pixel Distance', labelpad=15)

        # Legend outside
        ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True, facecolor='white', framealpha=1)

        ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
    else:
        ax2.text(0.5, 0.5, "No Distance Data Available", ha='center', va='center')

    # Final Layout
    plt.xlabel('Model', labelpad=15)
    plt.tight_layout()

    plt.savefig(OUTPUT_FILE, format='svg', bbox_inches='tight')
    print(f"Plot saved to {OUTPUT_FILE}")

    # --- CONSOLE TABLE ---
    print("\n" + "=" * 50)
    print("       BENCHMARK RESULTS: MEAN SCORES")
    print("=" * 50)
    pivot_scores = df.groupby(['Model', 'Category'])['Score'].mean().reset_index().pivot(index='Model',
                                                                                         columns='Category',
                                                                                         values='Score')
    pivot_scores = pivot_scores.reindex(DESIRED_ORDER)
    print(pivot_scores.round(2))

    print("\n" + "=" * 50)
    print("       BENCHMARK RESULTS: MEDIAN DISTANCE")
    print("=" * 50)
    pivot_dist = df.groupby(['Model', 'Category'])['Distance'].median().reset_index().pivot(index='Model',
                                                                                            columns='Category',
                                                                                            values='Distance')
    pivot_dist = pivot_dist.reindex(DESIRED_ORDER)
    print(pivot_dist.round(2))
    print("=" * 50 + "\n")


# --- EXECUTION ---
if __name__ == "__main__":
    df_results = parse_benchmark_results(ROOT_DIR)
    plot_benchmark_boxplot(df_results)