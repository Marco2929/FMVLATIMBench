import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- CONFIGURATION ---
ROOT_DIR = 'results'
OUTPUT_FILE_SCORES = 'benchmark_results_scores.svg'
OUTPUT_FILE_DISTANCE = 'benchmark_results_distance.svg'

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
plt.style.use('ggplot')
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 16,
    'font.family': 'sans-serif'
})


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
                        if len(row) < 3 or not row[0].startswith("20"):
                            continue

                        category_raw = row[1].strip()
                        raw_model_name = row[2].strip().split('/')[-1]
                        model_name = MODEL_MAPPING.get(raw_model_name, raw_model_name)

                        score = np.nan
                        distance = np.nan
                        category_label = None

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


def generate_plots(df):
    if df.empty:
        print("No data found.")
        return

    # Filter Data
    df = df[df['Model'].isin(DESIRED_ORDER)].copy()

    # Colors
    unique_categories = sorted(df['Category'].dropna().unique())
    default_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    category_colors = {cat: default_cycle[i % len(default_cycle)] for i, cat in enumerate(unique_categories)}

    # ==========================================
    # 1. PLOT SCORES (Bar Chart)
    # ==========================================
    plt.figure(figsize=(12, 7))

    df_means = df.groupby(['Model', 'Category'], as_index=False)['Score'].mean()

    ax1 = sns.barplot(
        data=df_means,
        x='Model',
        y='Score',
        hue='Category',
        order=DESIRED_ORDER,
        palette=category_colors,
        edgecolor='white',
        linewidth=1.0,
        errorbar=None
    )

    # Annotations
    for container in ax1.containers:
        labels = [f'{v.get_height():.1f}' if v.get_height() > 0 else '' for v in container]
        ax1.bar_label(container, labels=labels, padding=3, fontsize=11, fontweight='bold')

    # Styling
    plt.ylabel('Score (%)')
    plt.xlabel('Model', labelpad=10)
    plt.ylim(0, 115)  # Extra space for legend
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Legend Inside
    plt.legend(title=None, loc='best', frameon=True, facecolor='white', framealpha=0.9, ncol=1)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE_SCORES, format='svg', bbox_inches='tight')
    print(f"Scores plot saved to {OUTPUT_FILE_SCORES}")
    plt.close()

    # ==========================================
    # 2. PLOT DISTANCE (Box Plot)
    # ==========================================
    df_dist = df[df['Distance'].notna()]

    if not df_dist.empty:
        plt.figure(figsize=(12, 7))

        ax2 = sns.boxplot(
            data=df_dist,
            x='Model',
            y='Distance',
            hue='Category',
            order=DESIRED_ORDER,
            palette=category_colors,
            linewidth=1.5,
            showfliers=False
        )

        # Median Annotations Logic
        medians_data = df_dist.groupby(['Model', 'Category'])['Distance'].median().reset_index()
        categories = sorted(df_dist['Category'].dropna().unique())
        n_categories = len(categories)
        total_width = 0.8
        bar_width = total_width / n_categories

        for i, model in enumerate(DESIRED_ORDER):
            for j, category in enumerate(categories):
                median_val = medians_data[(medians_data['Model'] == model) &
                                          (medians_data['Category'] == category)]['Distance']
                if not median_val.empty:
                    val = median_val.values[0]
                    offset = (j - (n_categories - 1) / 2) * bar_width
                    x_pos = i + offset

                    # Text box background for readability on grid lines
                    ax2.text(x_pos, val, f'{val:.1f}',
                             ha='center', va='bottom',
                             fontsize=10, fontweight='bold',
                             color='black',
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))

        # Styling
        plt.ylabel('Pixel Distance (Lower is Better)')
        plt.xlabel('Model', labelpad=10)
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)

        # Legend Inside
        plt.legend(title=None, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

        plt.tight_layout()
        plt.savefig(OUTPUT_FILE_DISTANCE, format='svg', bbox_inches='tight')
        print(f"Distance plot saved to {OUTPUT_FILE_DISTANCE}")
        plt.close()
    else:
        print("No distance data available to plot.")


# --- EXECUTION ---
if __name__ == "__main__":
    df_results = parse_benchmark_results(ROOT_DIR)
    generate_plots(df_results)