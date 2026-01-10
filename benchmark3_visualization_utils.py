"""
Shared visualization utilities for benchmark3 event results (text and visual).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Common model mapping
MODEL_MAPPING = {
    'bytedance/ui-tars-1.5-7b': 'UI-TARS 1.5',
    'Qwen/Qwen2.5-VL-7B-Instruct': 'Qwen2.5 VL',
    'qwen/qwen-2.5-vl-7b-instruct': 'Qwen2.5 VL',
    'qwen/qwen3-vl-235b-a22b-instruct': 'Qwen3 VL',
    'qwen/qwen3-vl-8b-instruct': 'Qwen3 VL 8B',
    'gemini-2.5-flash': 'Gemini 2.5 Flash',
    'gpt-5-mini': 'GPT 5 Mini'
}

DESIRED_MODEL_ORDER = [
    'UI-TARS 1.5',
    'Qwen2.5 VL',
    'Qwen3 VL 8B',
    'Qwen3 VL',
    'Gemini 2.5 Flash',
    'GPT 5 Mini'
]

# Presentation styling
def setup_plot_style():
    """Configure matplotlib for presentation-quality plots."""
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


def load_csv_data(base_dir, score_column='final_score'):
    """
    Load CSV data from benchmark results directory.
    
    Args:
        base_dir: Path to results directory
        score_column: Name of the score column to extract
        
    Returns:
        tuple: (success_count, total_count, data_dict)
    """
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return 0, 0, None
    
    csv_count = 0
    success_count = 0
    
    data_dict = {'benchmark_type': [], 'model': [], 'score': []}
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".csv"):
                csv_count += 1
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        # Skip header, process data lines
                        for line in lines[1:]:
                            parts = line.strip().split(',')
                            if len(parts) >= 4:
                                benchmark_type = parts[1]
                                model = parts[2]
                                
                                # Handle different score formats
                                if score_column == 'final_score':
                                    score_str = parts[3].strip().lower()
                                    score = 1.0 if score_str == 'true' else 0.0
                                elif score_column == 'iou':
                                    try:
                                        score = float(parts[4])
                                    except:
                                        score = 0.0
                                else:
                                    score = 0.0
                                
                                data_dict['benchmark_type'].append(benchmark_type)
                                data_dict['model'].append(model)
                                data_dict['score'].append(score)
                    success_count += 1
                except Exception as e:
                    pass
    
    return success_count, csv_count, data_dict


def create_instruction_plot(data_dict, instruction_mapping, title, output_file, 
                            model_mapping=None, model_order=None):
    """
    Create a grouped bar plot comparing instruction types across models.
    
    Args:
        data_dict: Dictionary with 'benchmark_type', 'model', 'score' keys
        instruction_mapping: Dict mapping benchmark types to instruction categories
        title: Plot title
        output_file: Output filename
        model_mapping: Optional model name mapping
        model_order: Optional desired model order
    """
    if model_mapping is None:
        model_mapping = MODEL_MAPPING
    if model_order is None:
        model_order = DESIRED_MODEL_ORDER
    
    # Create DataFrame
    full_df = pd.DataFrame({
        'benchmark_type': pd.Series(data_dict['benchmark_type']),
        'model': pd.Series(data_dict['model']),
        'score': pd.Series(data_dict['score'])
    })
    
    # Apply model name mapping
    full_df['Model'] = full_df['model'].replace(model_mapping)
    
    # Map benchmark types to instruction categories
    full_df['Instruction'] = full_df['benchmark_type'].map(instruction_mapping)
    
    # Filter out unmapped categories
    full_df = full_df.dropna(subset=['Instruction'])
    
    # Calculate mean accuracy per Model and Instruction Type
    accuracy_df = full_df.groupby(['Model', 'Instruction'])['score'].mean().reset_index()
    accuracy_df['Accuracy'] = accuracy_df['score'] * 100.0
    
    # Pivot (Models as Index, Instructions as Columns)
    pivot_df = accuracy_df.pivot(index='Model', columns='Instruction', values='Accuracy')
    
    # Reorder columns to: Baseline, Parts List, Parts Descriptions
    column_order = ['Baseline', 'Parts List', 'Parts Descriptions']
    pivot_df = pivot_df[[col for col in column_order if col in pivot_df.columns]]
    
    # Apply desired model order
    pivot_df = pivot_df.reindex(model_order)
    pivot_df = pivot_df.dropna(how='all')
    
    # Setup plot style
    setup_plot_style()
    
    # Create plot
    ax = pivot_df.plot(
        kind='bar',
        figsize=(16, 9),
        width=0.8,
        edgecolor='white',
        linewidth=1.0
    )
    
    plt.title(title, pad=20)
    plt.xlabel('Model', labelpad=15)
    plt.ylabel('Accuracy (%) / IoU (%)', labelpad=15)
    
    plt.xticks(rotation=0)
    
    plt.legend(
        title='Instruction Type',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        frameon=True,
        facecolor='white',
        framealpha=1
    )
    
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add value annotations
    for container in ax.containers:
        labels = [f'{v.get_height():.1f}' if v.get_height() != 0 else '' for v in container]
        ax.bar_label(
            container,
            labels=labels,
            padding=4,
            fontsize=12,
            fontweight='bold'
        )
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to {output_file}")
    print("\nAccuracy Summary (%):")
    print(pivot_df)
    
    # Show data counts
    print("\nData counts per category:")
    counts = full_df.groupby(['Model', 'Instruction']).size().unstack(fill_value=0)
    print(counts)
