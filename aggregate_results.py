"""
Aggregate results from multiple experiment runs into a CSV file.

This script reads all run folders within an experiment directory and combines
the evaluation results into a single CSV file for easy analysis.
"""

import json
import csv
from pathlib import Path
import argparse
from datetime import datetime


def find_all_runs(experiment_path: Path):
    """Find all run directories in the experiment folder.
    
    Args:
        experiment_path: Path to the experiment directory
        
    Returns:
        List of tuples (run_dir_path, run_name)
    """
    runs = []
    if not experiment_path.exists():
        print(f"Experiment path does not exist: {experiment_path}")
        return runs
    
    # Find all run_* directories
    for run_dir in sorted(experiment_path.glob("run_*")):
        if run_dir.is_dir():
            runs.append((run_dir, run_dir.name))
    
    return runs


def load_run_results(run_dir: Path):
    """Load evaluation results from a run directory.
    
    Args:
        run_dir: Path to the run directory
        
    Returns:
        Dictionary with evaluation results or None if not found
    """
    results_file = run_dir / "evaluation" / "results.json"
    
    if not results_file.exists():
        print(f"Warning: No results.json found in {run_dir.name}")
        return None
    
    try:
        with open(results_file, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading results from {run_dir.name}: {e}")
        return None


def aggregate_to_csv(experiment_path: Path, output_csv: Path = None):
    """Aggregate all run results into a CSV file.
    
    Args:
        experiment_path: Path to the experiment directory
        output_csv: Path for the output CSV file (optional)
    """
    runs = find_all_runs(experiment_path)
    
    if not runs:
        print(f"No runs found in {experiment_path}")
        return
    
    print(f"Found {len(runs)} run(s) in {experiment_path}")
    
    # Collect all results
    all_results = []
    for run_dir, run_name in runs:
        results = load_run_results(run_dir)
        if results:
            results['run_name'] = run_name
            results['run_dir'] = str(run_dir)
            all_results.append(results)
    
    if not all_results:
        print("No valid results found to aggregate")
        return
    
    # Determine output path
    if output_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = experiment_path / f"aggregated_results_{timestamp}.csv"
    
    # Prepare CSV headers and rows
    headers = [
        'run_name',
        'success',
        'total_distance',
        'threshold',
        'mean_iou',
        'iou_threshold',
        'use_iou',
        'num_targets'
    ]
    
    # Collect all object names from all runs
    all_object_names = set()
    for result in all_results:
        if 'details' in result:
            all_object_names.update(result['details'].keys())
        if 'iou_scores' in result:
            all_object_names.update(result['iou_scores'].keys())
    
    # Add per-object columns
    for obj_name in sorted(all_object_names):
        headers.append(f'{obj_name}_distance')
        headers.append(f'{obj_name}_iou')
    
    # Write CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        for result in all_results:
            row = {
                'run_name': result['run_name'],
                'success': result.get('success', False),
                'total_distance': result.get('total_distance', 'N/A'),
                'threshold': result.get('threshold', 'N/A'),
                'mean_iou': result.get('mean_iou', 'N/A'),
                'iou_threshold': result.get('iou_threshold', 'N/A'),
                'use_iou': result.get('use_iou', False),
                'num_targets': len(result.get('details', {}))
            }
            
            # Add per-object metrics
            details = result.get('details', {})
            iou_scores = result.get('iou_scores', {})
            
            for obj_name in sorted(all_object_names):
                dist_key = f'{obj_name}_distance'
                iou_key = f'{obj_name}_iou'
                
                row[dist_key] = details.get(obj_name, 'N/A')
                row[iou_key] = iou_scores.get(obj_name, 'N/A')
            
            writer.writerow(row)
    
    print(f"\nAggregated results saved to: {output_csv}")
    print(f"Total runs processed: {len(all_results)}")
    
    # Print summary statistics
    successful_runs = sum(1 for r in all_results if r.get('success', False))
    print(f"Successful runs: {successful_runs}/{len(all_results)} ({100*successful_runs/len(all_results):.1f}%)")
    
    if all_results:
        avg_distance = sum(r.get('total_distance', 0) for r in all_results if isinstance(r.get('total_distance'), (int, float))) / len(all_results)
        print(f"Average total distance: {avg_distance:.2f} pixels")
        
        if any(r.get('use_iou', False) for r in all_results):
            iou_results = [r.get('mean_iou', 0) for r in all_results if isinstance(r.get('mean_iou'), (int, float))]
            if iou_results:
                avg_iou = sum(iou_results) / len(iou_results)
                print(f"Average mean IoU: {avg_iou:.4f}")


def aggregate_category(category_path: Path, output_dir: Path = None):
    """Aggregate results for all experiments in a category.
    
    Args:
        category_path: Path to the category directory containing multiple experiments
        output_dir: Directory to save the aggregated CSV files (optional)
    """
    if not category_path.exists():
        print(f"Category path does not exist: {category_path}")
        return
    
    # Find all experiment directories
    experiments = [d for d in category_path.iterdir() if d.is_dir()]
    
    if not experiments:
        print(f"No experiment directories found in {category_path}")
        return
    
    print(f"Found {len(experiments)} experiment(s) in {category_path.name}")
    
    for experiment_dir in sorted(experiments):
        print(f"\n{'='*60}")
        print(f"Processing experiment: {experiment_dir.name}")
        print(f"{'='*60}")
        
        output_csv = None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv = output_dir / f"{experiment_dir.name}_aggregated_{timestamp}.csv"
        
        aggregate_to_csv(experiment_dir, output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark manipulation results into CSV files"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="benchmark4_manipulation",
        help="Benchmark directory name (e.g., 'benchmark1_grounding', 'benchmark4_manipulation'). Default: benchmark4_manipulation"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Path to a specific experiment directory to aggregate"
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Category name (e.g., 'manipul_actions') to aggregate all experiments within"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to specify the results/model/category/experiment folder structure"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file path (optional, defaults to experiment folder with timestamp)"
    )
    
    args = parser.parse_args()
    
    if args.experiment:
        # Aggregate a specific experiment
        experiment_path = Path(args.experiment)
        output_csv = Path(args.output) if args.output else None
        aggregate_to_csv(experiment_path, output_csv)
        
    elif args.category:
        # Aggregate all experiments in a category
        if args.model:
            category_path = Path(args.benchmark) / "results" / args.model / args.category
        else:
            category_path = Path(args.benchmark) / "results" / args.category
        output_dir = Path(args.output) if args.output else None
        aggregate_category(category_path, output_dir)
        
    else:
        # Default: aggregate all categories
        results_root = Path(args.benchmark) / "results"
        
        if not results_root.exists():
            print(f"Results directory not found: {results_root}")
            exit(1)
        
        # Check if model is specified, otherwise iterate through model directories
        if args.model:
            model_path = results_root / args.model
            if not model_path.exists():
                print(f"Model directory not found: {model_path}")
                exit(1)
            
            categories = [d for d in model_path.iterdir() if d.is_dir()]
            
            if not categories:
                print(f"No category directories found in {model_path}")
                exit(1)
            
            print(f"Found {len(categories)} category(ies) for model {args.model}")
            
            for category_dir in sorted(categories):
                print(f"\n{'#'*70}")
                print(f"# Processing category: {category_dir.name}")
                print(f"{'#'*70}")
                aggregate_category(category_dir)
        else:
            # Iterate through all model directories
            model_dirs = [d for d in results_root.iterdir() if d.is_dir()]
            
            if not model_dirs:
                print(f"No model directories found in {results_root}")
                exit(1)
            
            print(f"Found {len(model_dirs)} model(s)")
            
            for model_dir in sorted(model_dirs):
                print(f"\n{'='*70}")
                print(f"= Processing model: {model_dir.name}")
                print(f"{'='*70}")
                
                categories = [d for d in model_dir.iterdir() if d.is_dir()]
                
                for category_dir in sorted(categories):
                    print(f"\n{'#'*70}")
                    print(f"# Processing category: {category_dir.name}")
                    print(f"{'#'*70}")
                    aggregate_category(category_dir)
