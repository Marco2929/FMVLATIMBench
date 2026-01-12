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


def load_csv_data(base_dir, score_column='final_score', skip_none_response=True):
    """
    Load CSV data from benchmark results directory.
    
    Args:
        base_dir: Path to results directory
        score_column: Name of the score column to extract
        skip_none_response: If True, skip rows with None response. If False, keep them (for IoU scoring)
        
    Returns:
        tuple: (success_count, total_count, data_dict)
    """
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return 0, 0, None
    
    csv_count = 0
    success_count = 0
    skipped_files = []
    
    # Statistics tracking
    total_rows_read = 0
    rows_skipped_no_ground_truth = 0
    rows_skipped_no_response = 0
    rows_kept = 0
    
    data_dict = {'benchmark_type': [], 'model': [], 'score': []}
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".csv"):
                csv_count += 1
                file_path = os.path.join(root, file)
                try:
                    # Use pandas for proper CSV parsing with quoted fields
                    df = None
                    rows_in_file = 0
                    
                    try:
                        df = pd.read_csv(file_path)
                        rows_in_file = len(df)
                    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
                        # If standard parsing fails, try manual line-by-line parsing
                        print(f"WARNING: Parsing issues in {file_path}, attempting manual recovery...")
                        
                        try:
                            # Read file manually and parse only valid lines
                            import csv
                            valid_rows = []
                            with open(file_path, 'r', encoding='utf-8') as f:
                                # Read header
                                header = f.readline().strip().split(',')
                                
                                # Read data lines
                                for line_num, line in enumerate(f, start=2):
                                    # Only process lines that start with a timestamp
                                    if not line.strip() or not line[0].isdigit():
                                        continue
                                    
                                    # Split and take first 12 columns (up to user_prompt)
                                    parts = line.strip().split(',')
                                    if len(parts) >= 10:  # At least up to response column
                                        valid_rows.append(parts[:12] if len(parts) >= 12 else parts)
                            
                            if valid_rows:
                                # Create DataFrame from valid rows
                                df = pd.DataFrame(valid_rows, columns=header[:len(valid_rows[0])])
                                rows_in_file = len(df)
                                print(f"  Recovered {rows_in_file} rows from {file_path}")
                            else:
                                print(f"ERROR: No valid rows recovered from {file_path}")
                                skipped_files.append((file_path, f"No valid rows after manual parsing"))
                                continue
                                
                        except Exception as e2:
                            print(f"ERROR: Manual parsing failed for {file_path}: {e2}")
                            skipped_files.append((file_path, f"CSV parsing failed: {e}"))
                            continue
                    
                    if df is None or len(df) == 0:
                        continue
                    
                    # Validate required columns exist
                    required_cols = ['benchmark_type', 'model', 'ground_truth', 'response', score_column]
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        print(f"WARNING: {file_path} missing columns: {missing_cols}")
                        skipped_files.append((file_path, f"Missing columns: {missing_cols}"))
                        continue
                    
                    rows_added = 0
                    for idx, row in df.iterrows():
                        total_rows_read += 1
                        
                        # Check if ground_truth or response is None/empty
                        gt = row.get('ground_truth')
                        resp = row.get('response')
                        
                        # Skip if ground_truth is None or empty string
                        if pd.isna(gt) or str(gt).strip() == '' or str(gt).strip().lower() == 'none':
                            rows_skipped_no_ground_truth += 1
                            continue
                        
                        # Skip if response is None or empty string (only if skip_none_response=True)
                        if skip_none_response:
                            if pd.isna(resp) or str(resp).strip() == '' or str(resp).strip().lower() == 'none':
                                rows_skipped_no_response += 1
                                continue
                        
                        rows_kept += 1
                        
                        benchmark_type = row['benchmark_type']
                        model = row['model']
                        
                        # Handle different score formats
                        if score_column == 'final_score':
                            score_str = str(row['final_score']).strip().lower()
                            score = 1.0 if score_str == 'true' else 0.0
                        elif score_column == 'iou':
                            try:
                                score = float(row['iou'])
                            except (ValueError, TypeError):
                                print(f"WARNING: Invalid IoU value in {file_path} row {idx}: {row.get('iou')}")
                                score = 0.0
                        else:
                            score = 0.0
                        
                        data_dict['benchmark_type'].append(benchmark_type)
                        data_dict['model'].append(model)
                        data_dict['score'].append(score)
                        rows_added += 1
                    
                    if rows_added > 0:
                        success_count += 1
                    
                except Exception as e:
                    print(f"ERROR reading {file_path}: {type(e).__name__}: {e}")
                    skipped_files.append((file_path, f"{type(e).__name__}: {e}"))
                    raise  # Re-raise to fail loudly
    
    # Print filtering statistics
    print(f"\n{'='*70}")
    print("DATA FILTERING STATISTICS")
    print(f"{'='*70}")
    print(f"Total rows read from CSVs:              {total_rows_read:,}")
    print(f"Rows skipped (no ground_truth):         {rows_skipped_no_ground_truth:,}")
    print(f"Rows skipped (no response):             {rows_skipped_no_response:,}")
    print(f"Rows kept (valid data):                 {rows_kept:,}")
    print(f"{'='*70}\n")
    
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files:")
        for path, reason in skipped_files[:10]:  # Show first 10
            print(f"  - {path}: {reason}")
        if len(skipped_files) > 10:
            print(f"  ... and {len(skipped_files) - 10} more")
    
    return success_count, csv_count, data_dict


def create_instruction_plot(data_dict, instruction_mapping, title, output_file, 
                            model_mapping=None, model_order=None, ylabel=None):
    """
    Create a grouped bar plot comparing instruction types across models.
    
    Args:
        data_dict: Dictionary with 'benchmark_type', 'model', 'score' keys
        instruction_mapping: Dict mapping benchmark types to instruction categories
        title: Plot title
        output_file: Output filename
        model_mapping: Optional model name mapping
        model_order: Optional desired model order
        ylabel: Optional y-axis label (default: 'Accuracy (%) / IoU (%)')
    """
    if model_mapping is None:
        model_mapping = MODEL_MAPPING
    if model_order is None:
        model_order = DESIRED_MODEL_ORDER
    if ylabel is None:
        ylabel = 'Accuracy (%) / IoU (%)'
    
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
    plt.ylabel(ylabel, labelpad=15)
    
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
