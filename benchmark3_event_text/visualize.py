import os
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
BASE_DIR = "results"
OUTPUT_FILE = "benchmark_results_plot_event_text.png"

# Apply the requested style
plt.style.use('ggplot')


def process_benchmark_data(base_dir):
    all_data = []

    # 1. Load Data
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    # Read CSV (Header=None because these logs usually lack headers)
                    df_temp = pd.read_csv(file_path, header=None, on_bad_lines='skip')

                    if df_temp.shape[1] > 3:
                        # Extract relevant columns: 1=Category, 2=Model, 3=Result
                        subset = df_temp[[1, 2, 3]].copy()
                        subset.columns = ['Category', 'Model', 'Result']

                        # Filter out potential header rows re-embedded in data
                        subset = subset[subset['Category'] != 'benchmark_type']
                        subset = subset[subset['Model'] != 'model']

                        # Clean Model Name
                        subset['Model'] = subset['Model'].apply(lambda x: str(x).split('/')[-1])

                        all_data.append(subset)
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")

    if not all_data:
        return pd.DataFrame()

    full_df = pd.concat(all_data, ignore_index=True)

    # Convert result to boolean (Robust string handling)
    full_df['Result'] = full_df['Result'].astype(str).str.strip().str.lower() == 'true'

    # Calculate Accuracy (Mean of booleans)
    accuracy_df = full_df.groupby(['Category', 'Model'])['Result'].mean().reset_index()
    accuracy_df.rename(columns={'Result': 'Accuracy'}, inplace=True)

    # Convert to Percentage
    accuracy_df['Accuracy'] = accuracy_df['Accuracy'] * 100.0

    return accuracy_df


# --- MAIN EXECUTION ---
df = process_benchmark_data(BASE_DIR)

if not df.empty:
    # Pivot: Index=Model, Columns=Category, Values=Accuracy
    pivot_df = df.pivot(index='Model', columns='Category', values='Accuracy')

    # --- SORTING LOGIC ---
    # Sort by Mean Accuracy Ascending (Low -> High)
    # This puts the Best Model on the RIGHT
    pivot_df['Mean_Acc'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values('Mean_Acc', ascending=True)
    pivot_df = pivot_df.drop(columns=['Mean_Acc'])

    # --- COLOR MAPPING ---
    # Assign distinct colors to categories to match your preferred style
    default_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    categories = sorted(pivot_df.columns)
    category_colors = {
        cat: default_cycle[i % len(default_cycle)]
        for i, cat in enumerate(categories)
    }
    plot_colors = [category_colors[col] for col in pivot_df.columns]

    # --- PLOTTING ---
    ax = pivot_df.plot(kind='bar', figsize=(14, 8), width=0.8,
                       color=plot_colors, edgecolor='white', linewidth=0.7)

    plt.title('Model Accuracy: by Event (Textual)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=0)  # Horizontal labels if they fit

    # Legend settings
    plt.legend(title='Event Category', loc='upper left', frameon=True, facecolor='white', framealpha=1)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Annotate Bars with Values
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Sorted plot saved to {OUTPUT_FILE}")
    print("\nSummary Data:")
    print(pivot_df)

else:
    print("No valid data found.")