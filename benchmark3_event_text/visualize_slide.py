"""
Visualization for benchmark3_event_text results.
Compares instruction types (baseline, parts list, parts descriptions) across models.
Averages results across cause, effect, and outcome events.
"""

import sys
sys.path.insert(0, '..')

from benchmark3_visualization_utils import load_csv_data, create_instruction_plot

# Configuration
BASE_DIR = "results"
OUTPUT_FILE = "benchmark_results_slides_event_text.svg"

# Instruction Type Mapping for text benchmarks
INSTRUCTION_MAPPING = {
    'outcome_text': 'Baseline',
    'effect_text': 'Baseline',
    'cause_text': 'Baseline',
    'outcome_text_partslist': 'Parts List',
    'effect_text_partslist': 'Parts List',
    'cause_text_partslist': 'Parts List',
    'outcome_text_partsdescriptions': 'Parts Descriptions',
    'effect_text_partsdescriptions': 'Parts Descriptions',
    'cause_text_partsdescriptions': 'Parts Descriptions',
}

def main():
    # Load CSV data
    success_count, csv_count, data_dict = load_csv_data(BASE_DIR, score_column='final_score')
    
    print(f"Found {csv_count} CSV files, successfully loaded {success_count}")
    print(f"Total data rows: {len(data_dict['benchmark_type'])}")
    
    if len(data_dict['benchmark_type']) == 0:
        print("No data found.")
        return
    
    # Create the plot
    title = 'Model Accuracy: Event Text Benchmark\n(Averaged across Cause, Effect, and Outcome)'
    create_instruction_plot(
        data_dict,
        INSTRUCTION_MAPPING,
        title,
        OUTPUT_FILE
    )

if __name__ == "__main__":
    main()
