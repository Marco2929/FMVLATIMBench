import os
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
base_dir = "results"
output_file = "benchmark_results_plot_understanding.png"

all_data = []

# 1. Load Data
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            try:
                # Read CSV
                df_temp = pd.read_csv(file_path, header=None, on_bad_lines='skip')

                if df_temp.shape[1] > 3:
                    subset = df_temp[[1, 2, 3]].copy()
                    subset.columns = ['Category', 'Model', 'Result']

                    # Filter out header rows if they exist
                    subset = subset[subset['Category'] != 'benchmark_type']
                    subset = subset[subset['Model'] != 'model']

                    all_data.append(subset)
            except Exception as e:
                print(f"Skipping {file_path}: {e}")

if not all_data:
    print("No data found.")
else:
    full_df = pd.concat(all_data, ignore_index=True)

    # Convert result to boolean
    full_df['Result'] = full_df['Result'].astype(str).str.strip().str.lower() == 'true'

    # Calculate Accuracy
    accuracy_df = full_df.groupby(['Category', 'Model'])['Result'].mean().reset_index()
    accuracy_df.rename(columns={'Result': 'Accuracy'}, inplace=True)

    # Pivot (Models as Index, Categories as Columns)
    pivot_df = accuracy_df.pivot(index='Model', columns='Category', values='Accuracy')

    # --- SORTING LOGIC ---
    # Option 1: Sort by Average Accuracy across all categories (Best models on the left)
    pivot_df['Mean_Acc'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values('Mean_Acc', ascending=True)
    pivot_df = pivot_df.drop(columns=['Mean_Acc'])  # Remove helper col before plotting

    # Option 2: Sort by a specific category (e.g., 'state_ident')
    # pivot_df = pivot_df.sort_values('state_ident', ascending=False)
    # ---------------------

    # Plot
    ax = pivot_df.plot(kind='bar', figsize=(14, 8), width=0.8)

    plt.title('Model Accuracy by Understanding', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Category')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(output_file)
    print(f"Sorted plot saved to {output_file}")
    print(pivot_df)