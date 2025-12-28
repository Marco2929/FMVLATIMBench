import os
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
BASE_DIR = "results"
OUTPUT_FILE = "benchmark_results_plot_understanding.png"

# Set the style to match the previous plot
plt.style.use('ggplot')

all_data = []

# 1. Load Data
for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            try:
                # Read CSV (header=None to handle files without standard headers)
                df_temp = pd.read_csv(file_path, header=None, on_bad_lines='skip')

                if df_temp.shape[1] > 3:
                    # Extract relevant columns: 1=Category, 2=Model, 3=Result
                    subset = df_temp[[1, 2, 3]].copy()
                    subset.columns = ['Category', 'Model', 'Result']

                    # Filter out header rows found inside data
                    subset = subset[subset['Category'] != 'benchmark_type']
                    subset = subset[subset['Model'] != 'model']

                    all_data.append(subset)
            except Exception as e:
                print(f"Skipping {file_path}: {e}")

if not all_data:
    print("No data found.")
else:
    full_df = pd.concat(all_data, ignore_index=True)

    # 2. Process Data
    # Convert result to boolean
    full_df['Result'] = full_df['Result'].astype(str).str.strip().str.lower() == 'true'

    # Calculate Accuracy (Mean) and convert to Percentage (0-100)
    accuracy_df = full_df.groupby(['Category', 'Model'])['Result'].mean().reset_index()
    accuracy_df['Accuracy'] = accuracy_df['Result'] * 100.0  # Convert to percentage

    # Pivot (Models as Index, Categories as Columns)
    pivot_df = accuracy_df.pivot(index='Model', columns='Category', values='Accuracy')

    # --- SORTING LOGIC ---
    # Sort by Average Accuracy across all categories (Best models on the left)
    pivot_df['Mean_Acc'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values('Mean_Acc', ascending=True)  # Ascending for correct left-to-right bar order
    pivot_df = pivot_df.drop(columns=['Mean_Acc'])
    # ---------------------

    # 3. Plotting
    # Use edgecolor='white' for cleaner bar separation
    ax = pivot_df.plot(kind='bar', figsize=(14, 8), width=0.8, edgecolor='white', linewidth=0.7)

    plt.title('Model Accuracy: Understanding Benchmark', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')

    # Rotate x-labels slightly to prevent overlap, or 0 if short names
    plt.xticks(rotation=0)

    # Move legend outside if it's crowded, or keep inside 'best'
    plt.legend(title='Category', bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True, facecolor='white')

    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Add Value Annotations on top of bars
    for container in ax.containers:
        # fmt='%.1f' formats to 1 decimal place
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=9)

    plt.tight_layout()

    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Sorted plot saved to {OUTPUT_FILE}")
    print(pivot_df)