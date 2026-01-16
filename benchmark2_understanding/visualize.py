import os
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
BASE_DIR = "results"
OUTPUT_FILE = "benchmark_results_understanding.svg"

MODEL_MAPPING = {
    'bytedance/ui-tars-1.5-7b': 'UI-TARS 1.5',
    'qwen/qwen-2.5-vl-7b-instruct': 'Qwen2.5 VL',
    'qwen/qwen3-vl-235b-a22b-instruct': 'Qwen3 VL',
    'gemini-2.5-flash': 'Gemini 2.5 Flash',
    'gpt-5-mini': 'GPT 5 Mini'
}

CATEGORY_MAPPING = {
    'state_ident': 'State Ident',
    'with_instruct': 'Prop. Ident (w/ instr.)',
    'without_instruct': 'Prop. Ident (no instr.)'
}

DESIRED_ORDER = [
    'UI-TARS 1.5',
    'Qwen2.5 VL',
    'Qwen3 VL',
    'Gemini 2.5 Flash',
    'GPT 5 Mini'
]

# --- REFINED PUBLICATION STYLING ---
plt.style.use('default')

# Balanced settings: Large enough for papers, but small enough to avoid overlap
params = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.weight': 'bold',
    'font.size': 18,  # Reduced from 22 to 18
    'axes.labelsize': 22,  # Y-axis label size
    'xtick.labelsize': 18,  # X-axis model names (Critical for overlap)
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.figsize': (12, 6),  # Widened slightly (10 -> 12) to fit names
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'lines.linewidth': 1.5
}
plt.rcParams.update(params)

all_data = []

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
                        subset = subset[subset['Category'] != 'benchmark_type']
                        subset = subset[subset['Model'] != 'model']
                        all_data.append(subset)
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")

if not all_data:
    print("No data found.")
else:
    full_df = pd.concat(all_data, ignore_index=True)
    full_df['Model'] = full_df['Model'].replace(MODEL_MAPPING)
    full_df['Category'] = full_df['Category'].replace(CATEGORY_MAPPING)
    full_df['Result'] = full_df['Result'].astype(str).str.strip().str.lower() == 'true'

    accuracy_df = full_df.groupby(['Category', 'Model'])['Result'].mean().reset_index()
    accuracy_df['Accuracy'] = accuracy_df['Result'] * 100.0

    pivot_df = accuracy_df.pivot(index='Model', columns='Category', values='Accuracy')
    pivot_df = pivot_df.reindex(DESIRED_ORDER)
    pivot_df = pivot_df.dropna(how='all')

    colors = ['#4E79A7', '#F28E2B', '#76B7B2']

    ax = pivot_df.plot(
        kind='bar',
        figsize=params['figure.figsize'],
        width=0.85,
        color=colors,
        edgecolor='black',
        linewidth=1.2,
        zorder=3
    )

    # REMOVED X-AXIS LABEL
    plt.xlabel('')

    plt.ylabel('Accuracy (%)', labelpad=10, fontweight='bold', fontsize=22)
    plt.ylim(0, 125)

    # Ensure tick labels are horizontal
    plt.xticks(rotation=0)

    plt.legend(
        bbox_to_anchor=(0.5, 1.12),
        loc='upper center',
        ncol=3,  # Back to 3 columns to save vertical space
        frameon=False,
        fontsize=16,  # Slightly smaller legend to prevent wrapping
        columnspacing=1.5
    )

    ax.grid(axis='y', linestyle='--', alpha=0.4, color='gray', zorder=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    for container in ax.containers:
        # Reverted to 1 decimal place if you prefer precision, or keep .0f
        labels = [f'{v.get_height():.0f}' if v.get_height() > 0 else '' for v in container]
        ax.bar_label(
            container,
            labels=labels,
            padding=3,
            fontsize=16,  # Adjusted to match bar width
            fontweight='bold',
            color='black'
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, format='svg', bbox_inches='tight')
    plt.savefig(OUTPUT_FILE.replace('.svg', '.png'), dpi=300, bbox_inches='tight')

    print(f"Refined plot saved to {OUTPUT_FILE}")