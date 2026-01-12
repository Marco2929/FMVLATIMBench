import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
ROOT_DIR = 'results'
OUTPUT_FILE = 'benchmark_results_plot_grounding.png'

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
    'Qwen2.5 VL',
    'UI-TARS 1.5',
    'Qwen3 VL',
    'Gemini 2.5 Flash',
    'GPT 5 Mini'
]

# --- PRESENTATION STYLING ---
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

                        # Apply Mapping - Default to raw name if not found in dict
                        model_name = MODEL_MAPPING.get(raw_model_name, raw_model_name)

                        score = np.nan
                        distance = np.nan
                        category_label = None

                        if "classify" in category_raw:
                            category_label = "Classify"
                            # Handle boolean strings safely
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


def plot_benchmark(df):
    if df.empty:
        print("No data found.")
        return

    # Aggregate Means
    df_agg = df.groupby(['Model', 'Category']).mean(numeric_only=True).reset_index()

    # --- COLOR MAPPING ---
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
    existing_models = [m for m in DESIRED_ORDER if m in pivot_score.index]
    unlisted_models = [m for m in pivot_score.index if m not in existing_models]
    final_order = existing_models + unlisted_models

    pivot_score = pivot_score.reindex(final_order)
    pivot_dist = pivot_dist.reindex(final_order)

    # --- PRINT COMPLETE TABLES TO CONSOLE ---
    print("\n" + "=" * 50)
    print("       BENCHMARK RESULTS: ACCURACY / IoU (%)")
    print("=" * 50)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(pivot_score.round(2))

    print("\n" + "=" * 50)
    print("       BENCHMARK RESULTS: DISTANCE (Pixels)")
    print("=" * 50)
    # Filter out columns that are all NaN/Zero for distance (e.g., 'Classify')
    pivot_dist_clean = pivot_dist.dropna(axis=1, how='all')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        if not pivot_dist_clean.empty:
            print(pivot_dist_clean.round(2))
        else:
            print("No distance metrics available.")
    
    # Calculate distance-based score: 100% - 100 * distance / sqrt(640^2 + 441^2)
    max_distance = np.sqrt(640**2 + 441**2)  # ≈777.1
    pivot_dist_score = 100 - (100 * pivot_dist / max_distance)
    
    # For Localize Multi NaN values, use Localize values
    if 'Localize Multi' in pivot_dist_score.columns and 'Localize' in pivot_dist_score.columns:
        pivot_dist_score['Localize Multi'] = pivot_dist_score['Localize Multi'].fillna(pivot_dist_score['Localize'])
    
    pivot_dist_score_clean = pivot_dist_score.dropna(axis=1, how='all')
    
    print("\n" + "=" * 50)
    print("   BENCHMARK RESULTS: DISTANCE-BASED SCORE (%)")
    print("   (100% - 100 * distance / diagonal)")
    print("=" * 50)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        if not pivot_dist_score_clean.empty:
            print(pivot_dist_score_clean.round(2))
        else:
            print("No distance metrics available.")
    
    # Calculate combined metric: 1/5 * (classify + iou_localize + iou_multi + dist_localize + dist_multi)
    # Only average across available (non-NaN) values for each model
    combined_metric = pd.DataFrame(index=pivot_score.index)
    
    # Add each component (keep NaN for missing data)
    combined_metric['Classify'] = pivot_score.get('Classify', pd.Series(index=pivot_score.index, dtype=float))
    combined_metric['IoU_Localize'] = pivot_score.get('Localize', pd.Series(index=pivot_score.index, dtype=float))
    combined_metric['IoU_Multi'] = pivot_score.get('Localize Multi', pd.Series(index=pivot_score.index, dtype=float))
    
    # For distance scores, use original pivot_dist_score without fillna
    pivot_dist_score_original = 100 - (100 * pivot_dist / max_distance)
    combined_metric['Dist_Localize'] = pivot_dist_score_original.get('Localize', pd.Series(index=pivot_score.index, dtype=float))
    combined_metric['Dist_Multi'] = pivot_dist_score_original.get('Localize Multi', pd.Series(index=pivot_score.index, dtype=float))
    
    # Calculate the average only across non-NaN values for each model
    combined_metric['Overall'] = combined_metric[['Classify', 'IoU_Localize', 'IoU_Multi', 'Dist_Localize', 'Dist_Multi']].mean(axis=1, skipna=True)
    
    print("\n" + "=" * 50)
    print("      COMBINED METRIC (Average of Available Components)")
    print("=" * 50)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(combined_metric[['Overall']].sort_values('Overall', ascending=False).round(2))
    print("=" * 50 + "\n")

    # --- PLOTTING ---
    # Increased height to (16, 12) to accommodate two subplots comfortably
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

    # 1. SCORES PLOT
    colors_top = [category_colors.get(col, 'gray') for col in pivot_score.columns]
    pivot_score.plot(kind='bar', ax=ax1, width=0.8, color=colors_top,
                     edgecolor='white', linewidth=1.0)

    ax1.set_title('Model Accuracy: Grounding Benchmark', pad=20)
    ax1.set_ylabel('Accuracy / IoU (%)', labelpad=15)

    # Place legend
    ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True, facecolor='white', framealpha=1)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    ax1.set_ylim(0, 110)  # Fixed Y-axis for consistency

    for container in ax1.containers:
        # Show label only if value > 0
        labels = [f'{v.get_height():.1f}' if v.get_height() > 0 else '' for v in container]
        ax1.bar_label(container, labels=labels, padding=3, fontsize=12, fontweight='bold')

    # 2. DISTANCE PLOT
    dist_cols = [c for c in pivot_dist.columns if pivot_dist[c].sum() > 0]

    if dist_cols:
        colors_bottom = [category_colors.get(col, 'gray') for col in dist_cols]
        pivot_dist[dist_cols].plot(kind='bar', ax=ax2, width=0.8, color=colors_bottom,
                                   edgecolor='white', linewidth=1.0)

        ax2.set_title('Localization Error (Lower is Better)', pad=20)
        ax2.set_ylabel('Mean Distance (Pixels)', labelpad=15)
        ax2.grid(axis='y', linestyle='--', alpha=0.6)

        # Legend for bottom plot
        ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True, facecolor='white', framealpha=1)

        for container in ax2.containers:
            labels = [f'{v.get_height():.1f}' if v.get_height() > 0 else '' for v in container]
            ax2.bar_label(container, labels=labels, padding=3, fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, "No Distance Metrics Available", ha='center', va='center', fontsize=16)

    # Final Layout
    plt.xlabel('Model', labelpad=15)
    plt.xticks(rotation=0)

    # Tight layout with extra padding for legends
    plt.tight_layout()

    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {OUTPUT_FILE}")


# --- EXECUTION ---
if __name__ == "__main__":
    df_results = parse_benchmark_results(ROOT_DIR)
    plot_benchmark(df_results)