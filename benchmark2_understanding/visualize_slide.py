import os
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
BASE_DIR = "results"
OUTPUT_FILE = "benchmark_results_slides_understanding.svg"

# 1. Map Raw IDs (from CSV) to Display Names
MODEL_MAPPING = {
    'bytedance/ui-tars-1.5-7b': 'UI-TARS 1.5',
    'qwen/qwen-2.5-vl-7b-instruct': 'Qwen2.5 VL',
    'qwen/qwen3-vl-235b-a22b-instruct': 'Qwen3 VL',
    'gemini-2.5-flash': 'Gemini 2.5 Flash',
    'gpt-5-mini': 'GPT 5 Mini'
}

# 2. Map Category Names to Display Names
CATEGORY_MAPPING = {
    'state_ident': 'State Ident',
    'with_instruct': 'Property Ident with instruct',
    'without_instruct': 'Property Ident without instruct'
}

# 3. Define the exact order for the X-axis
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
    'font.size': 16,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 26,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
})

all_data = []

# 4. Load Data
if os.path.exists(BASE_DIR):
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    df_temp = pd.read_csv(file_path, header=None, on_bad_lines='skip')

                    if df_temp.shape[1] > 3:
                        subset = df_temp[[1, 2, 3]].copy()
                        subset.columns = ['Category', 'Model', 'Result']

                        # Filter out header rows found inside data
                        subset = subset[subset['Category'] != 'benchmark_type']
                        subset = subset[subset['Model'] != 'model']

                        all_data.append(subset)
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")
else:
    print(f"Directory {BASE_DIR} does not exist.")

if not all_data:
    print("No data found.")
else:
    full_df = pd.concat(all_data, ignore_index=True)

    # 5. Apply Model and Category Renaming
    full_df['Model'] = full_df['Model'].replace(MODEL_MAPPING)
    full_df['Category'] = full_df['Category'].replace(CATEGORY_MAPPING)

    # 6. Process Data
    full_df['Result'] = full_df['Result'].astype(str).str.strip().str.lower() == 'true'
    accuracy_df = full_df.groupby(['Category', 'Model'])['Result'].mean().reset_index()
    accuracy_df['Accuracy'] = accuracy_df['Result'] * 100.0

    # Pivot (Models as Index, Categories as Columns)
    pivot_df = accuracy_df.pivot(index='Model', columns='Category', values='Accuracy')

    # 7. Apply Desired Order
    pivot_df = pivot_df.reindex(DESIRED_ORDER)
    pivot_df = pivot_df.dropna(how='all')

    # 8. Plotting
    ax = pivot_df.plot(
        kind='bar',
        figsize=(20, 9),
        width=0.8,
        edgecolor='white',
        linewidth=1.0
    )

    # plt.title('Model Accuracy: Understanding Benchmark', pad=20)
    plt.xlabel('Model', labelpad=15)
    plt.ylabel('Accuracy (%)', labelpad=15)

    # Fix Y-axis to 0-110 range
    plt.ylim(0, 110)

    plt.xticks(rotation=0)

    plt.legend(
        title='Category',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        frameon=True,
        facecolor='white',
        framealpha=1
    )

    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Add Value Annotations
    for container in ax.containers:
        # Only show labels > 0
        labels = [f'{v.get_height():.1f}' if v.get_height() > 0 else '' for v in container]
        ax.bar_label(
            container,
            labels=labels,
            padding=4,
            fontsize=12,
            fontweight='bold'
        )

    plt.tight_layout()

    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Sorted plot saved to {OUTPUT_FILE}")
    print("\nResults Table:")
    print(pivot_df.round(2))