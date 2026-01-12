"""
Visualization for benchmark3_event_visual results.
Compares instruction types (baseline, parts list, parts descriptions) across models.
Averages results across cause, effect, and outcome events.
"""

import sys
sys.path.insert(0, '..')

from benchmark3_visualization_utils import load_csv_data, create_instruction_plot

# Configuration
BASE_DIR = "results"
OUTPUT_FILE = "benchmark_results_slides_event_visual.svg"

# Instruction Type Mapping for visual benchmarks
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

def main():
    # Load CSV data (keep response=None rows for IoU scoring - they count as IoU=0)
    success_count, csv_count, data_dict = load_csv_data(BASE_DIR, score_column='iou', skip_none_response=False)
    
    print(f"Found {csv_count} CSV files, successfully loaded {success_count}")
    print(f"Total data rows: {len(data_dict['benchmark_type'])}")
    
    if len(data_dict['benchmark_type']) == 0:
        print("No data found.")
        return
    
    # Create the plot
    title = 'Model Accuracy: Event Visual Benchmark\n(Averaged across Cause, Effect, and Outcome)'
    create_instruction_plot(
        data_dict,
        INSTRUCTION_MAPPING,
        title,
        OUTPUT_FILE
    )

if __name__ == "__main__":
    main()
