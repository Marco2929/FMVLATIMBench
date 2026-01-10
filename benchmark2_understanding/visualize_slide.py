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

# 3. Load Data
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

    # 4. Apply Model Naming Mapping
    # Using replace ensures that if a model isn't in the map, it keeps its original name
    full_df['Model'] = full_df['Model'].replace(MODEL_MAPPING)

    # 5. Process Data
    full_df['Result'] = full_df['Result'].astype(str).str.strip().str.lower() == 'true'
    accuracy_df = full_df.groupby(['Category', 'Model'])['Result'].mean().reset_index()
    accuracy_df['Accuracy'] = accuracy_df['Result'] * 100.0

    # Pivot (Models as Index, Categories as Columns)
    pivot_df = accuracy_df.pivot(index='Model', columns='Category', values='Accuracy')

    # 6. Apply Desired Order
    # reindex forces the order. valid models not in DESIRED_ORDER will be dropped or pushed to end depending on logic.
    # Here we strictly enforce the list. Models in the data but NOT in the list will become NaN rows (we drop them).
    # Models in the list but NOT in the data will be NaN rows (we can fill them with 0 or drop).
    pivot_df = pivot_df.reindex(DESIRED_ORDER)

    # Optional: Drop rows that are completely empty if a model in DESIRED_ORDER wasn't found in data
    pivot_df = pivot_df.dropna(how='all')

    # 7. Plotting
    ax = pivot_df.plot(
        kind='bar',
        figsize=(16, 9),
        width=0.8,
        edgecolor='white',
        linewidth=1.0
    )

    plt.title('Model Accuracy: Understanding Benchmark', pad=20)
    plt.xlabel('Model', labelpad=15)
    plt.ylabel('Accuracy (%)', labelpad=15)

    # REVERTED ROTATION TO NORMAL (0)
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
        ax.bar_label(
            container,
            fmt='%.1f',
            padding=4,
            fontsize=12,
            fontweight='bold'
        )

    plt.tight_layout()

    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Sorted plot saved to {OUTPUT_FILE}")
    print(pivot_df)