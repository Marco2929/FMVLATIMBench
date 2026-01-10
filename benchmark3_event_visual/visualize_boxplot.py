"""
Box plot visualization for benchmark3_event_visual results.
Shows distribution of IoU scores across instruction types and models.
Uses benchmark3 styling with box plot approach from benchmark1.
"""

import sys
sys.path.insert(0, '..')

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import shared utilities for consistent styling
from benchmark3_visualization_utils import MODEL_MAPPING, DESIRED_MODEL_ORDER, setup_plot_style

# Configuration
BASE_DIR = "results"
OUTPUT_FILE = "benchmark_results_boxplot_distance_event_visual.svg"

# Instruction Type Mapping
INSTRUCTION_MAPPING = {
    'outcome_visual': 'Baseline',
    'effect_visual': 'Baseline',
    'cause_visual': 'Baseline',
    'outcome_visual_partslist': 'Parts List',
    'effect_visual_partslist': 'Parts List',
    'cause_visual_partslist': 'Parts List',
    'outcome_visual_partsdescriptions': 'Parts Descriptions',
    'effect_visual_partsdescriptions': 'Parts Descriptions',
    'cause_visual_partsdescriptions': 'Parts Descriptions',
}


def load_raw_data(base_dir):
    """Load all CSV data without aggregation for box plot."""
    all_rows = []
    csv_count = 0
    
    for csv_file in Path(base_dir).rglob("*.csv"):
        csv_count += 1
        try:
            # Read CSV with pandas for proper parsing
            df = pd.read_csv(csv_file)
            
            for _, row in df.iterrows():
                # Skip rows where ground_truth or response is None
                if (pd.isna(row.get('ground_truth')) or str(row.get('ground_truth')).strip().lower() == 'none' or
                    pd.isna(row.get('response')) or str(row.get('response')).strip().lower() == 'none'):
                    continue
                
                benchmark_type = row['benchmark_type']
                model_id = row['model'].strip()  # Keep full model path for mapping
                
                # Extract distance score
                try:
                    distance = float(row['distance'])
                    if distance < 0:
                        continue
                except (ValueError, KeyError):
                    continue
                
                # Map to instruction type
                if benchmark_type not in INSTRUCTION_MAPPING:
                    continue
                
                instruction_type = INSTRUCTION_MAPPING[benchmark_type]
                
                # Map model name
                model_name = MODEL_MAPPING.get(model_id, model_id)
                if model_name not in DESIRED_MODEL_ORDER:
                    continue
                
                all_rows.append({
                    'model': model_name,
                    'instruction_type': instruction_type,
                    'distance': distance  # Keep as pixels
                })
        
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
    print(f"Loaded {len(all_rows)} data points from {csv_count} CSV files")
    return pd.DataFrame(all_rows)


def create_boxplot(df, output_file):
    """Create box plot visualization."""
    if df.empty:
        print("No data to plot.")
        return
    
    # Setup plot style
    setup_plot_style()
    
    # Define instruction type order
    instruction_order = ['Baseline', 'Parts List', 'Parts Descriptions']
    
    # Define colors for instruction types
    colors = {
        'Baseline': '#1f77b4',
        'Parts List': '#ff7f0e',
        'Parts Descriptions': '#2ca02c'
    }
    
    # Filter DESIRED_MODEL_ORDER to only include models present in the data
    models_in_data = set(df['model'].unique())
    model_order = [m for m in DESIRED_MODEL_ORDER if m in models_in_data]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Create box plot
    sns.boxplot(
        data=df,
        x='model',
        y='distance',
        hue='instruction_type',
        order=model_order,
        hue_order=instruction_order,
        palette=colors,
        ax=ax,
        linewidth=1.5,
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.3},
        showfliers=True
    )
    
    # Add median values on top of boxes
    medians_data = df.groupby(['model', 'instruction_type'])['distance'].median().reset_index()
    
    n_instructions = len(instruction_order)
    box_width = 0.8 / n_instructions
    
    for i, model in enumerate(model_order):
        for j, instruction in enumerate(instruction_order):
            median_val = medians_data[
                (medians_data['model'] == model) & 
                (medians_data['instruction_type'] == instruction)
            ]['distance']
            
            if not median_val.empty:
                # Calculate x position for grouped boxes
                x_pos = i + (j - n_instructions/2 + 0.5) * box_width
                y_pos = median_val.values[0]
                
                # Only show label if median > 0
                if y_pos > 0:
                    ax.text(
                        x_pos, y_pos, f'{y_pos:.1f}',
                        ha='center', va='bottom',
                        fontsize=11, fontweight='bold'
                    )
    
    # Styling
    ax.set_title(
        'Localization Error Distribution: Event Visual Benchmark\n(Across Cause, Effect, and Outcome)',
        pad=20
    )
    ax.set_xlabel('Model', labelpad=15)
    ax.set_ylabel('Distance (Pixels)', labelpad=15)
    ax.legend(
        title='Instruction Type',
        bbox_to_anchor=(1.01, 1),
        loc='upper left',
        frameon=True,
        facecolor='white',
        framealpha=1
    )
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_file, format='svg', bbox_inches='tight')
    print(f"Box plot saved to {output_file}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS: Distance Distribution (Pixels)")
    print("=" * 70)
    summary = df.groupby(['model', 'instruction_type'])['distance'].describe()
    print(summary.round(2))
    print("=" * 70 + "\n")


def main():
    # Load raw data (all individual measurements)
    df = load_raw_data(BASE_DIR)
    
    if df.empty:
        print("No data found.")
        return
    
    # Create box plot
    create_boxplot(df, OUTPUT_FILE)


if __name__ == "__main__":
    main()
