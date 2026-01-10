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

# Set style (Seaborn works well with ggplot style too)
plt.style.use('ggplot')


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

    # Filter Data to only include models in DESIRED_ORDER (optional, keeps plot clean)
    df = df[df['Model'].isin(DESIRED_ORDER)].copy()

    # Define Colors to maintain consistency
    unique_categories = sorted(df['Category'].dropna().unique())
    default_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    category_colors = {cat: default_cycle[i % len(default_cycle)] for i, cat in enumerate(unique_categories)}

    # Setup Subplots (2 rows)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14), sharex=True)

    # --- PLOT 1: SCORES (Accuracy / IoU) - BAR CHART WITH MEANS ---
    # Calculate mean scores for each Model-Category combination
    df_means = df.groupby(['Model', 'Category'], as_index=False)['Score'].mean()
    
    sns.barplot(
        data=df_means,
        x='Model',
        y='Score',
        hue='Category',
        order=DESIRED_ORDER,
        palette=category_colors,
        ax=ax1,
        errorbar=None  # No error bars, just show means
    )

    # Add mean values on top of bars
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f', padding=3, fontsize=12)

    ax1.set_title('Model Mean Accuracy', fontsize=18, fontweight='bold', pad=15)
    ax1.set_ylabel('Mean Accuracy (%) / IoU (%)', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.legend(loc='upper left', framealpha=1, facecolor='white', fontsize=14)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    # --- PLOT 2: DISTANCE (Error) ---
    # Filter out Classify for the distance plot (usually has no distance)
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
            linewidth=1.2,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
            showfliers = False
        )

        # Add median values on top of boxplot median lines
        # Get unique categories and models for positioning
        medians_data = df_dist.groupby(['Model', 'Category'])['Distance'].median().reset_index()
        
        # Get the positions of the boxes
        categories = sorted(df_dist['Category'].dropna().unique())
        n_categories = len(categories)
        
        for i, model in enumerate(DESIRED_ORDER):
            for j, category in enumerate(categories):
                median_val = medians_data[(medians_data['Model'] == model) & 
                                         (medians_data['Category'] == category)]['Distance']
                if not median_val.empty:
                    # Calculate x position (adjusted for grouped boxes)
                    x_pos = i + (j - n_categories/2 + 0.5) * (0.8 / n_categories)
                    y_pos = median_val.values[0]
                    ax2.text(x_pos, y_pos, f'{y_pos:.1f}', 
                            ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax2.set_title('Localization Error Distribution', fontsize=18, fontweight='bold', pad=15)
        ax2.set_ylabel('Pixel Distance', fontsize=16)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.legend(loc='upper left', framealpha=1, facecolor='white', fontsize=14)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
    else:
        ax2.text(0.5, 0.5, "No Distance Data Available", ha='center', va='center')

    # Final Layout
    plt.xlabel('Model', fontsize=16, fontweight='bold', labelpad=10)
    plt.tight_layout()

    # Save as SVG
    plt.savefig(OUTPUT_FILE, format='svg', bbox_inches='tight')
    print(f"Plot saved to {OUTPUT_FILE}")


# --- EXECUTION ---
if __name__ == "__main__":
    df_results = parse_benchmark_results(ROOT_DIR)
    plot_benchmark_boxplot(df_results)