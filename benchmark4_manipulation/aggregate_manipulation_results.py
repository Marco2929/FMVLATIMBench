import os
import json
import csv
from pathlib import Path

def main():
    # Use absolute path resolving to find the results directory relative to this script
    script_dir = Path(__file__).parent.absolute()
    base_dir = script_dir / "results"
    output_file = script_dir / "benchmark4_results_summary.json"
    output_csv_file = script_dir / "benchmark4_results_summary.csv"

    print(f"Scanning directory: {base_dir}")

    results_data = {} # (category, model) -> {runs: int, success: int}

    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist.")
        return

    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            if d.startswith("run_"):
                run_path = Path(root) / d
                try:
                    rel_path = run_path.relative_to(base_dir)
                    parts = rel_path.parts
                    
                    # Expected structure relative to results/: 
                    # Category / Subcategory / Model / RunID
                    # This gives 4 parts usually. 
                    
                    if len(parts) < 4:
                        # If structure is shallower, we might need adjustments.
                        # But based on current knowledge:
                        # parts[0] is Category
                        # parts[-2] is Model (parent of RunID)
                        pass
                    
                    category = parts[0]
                    # Model is the parent folder of the run folder
                    model_name = parts[-2]
                            
                    # Check for results.json
                    results_file = run_path / "evaluation" / "results.json"
                    is_success = False
                    
                    if results_file.exists():
                        try:
                            with open(results_file, 'r') as f:
                                data = json.load(f)
                                if isinstance(data, dict) and data.get("user_marked_success") is True:
                                    is_success = True
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON: {results_file}")
                        except Exception as e:
                            print(f"Error reading {results_file}: {e}")
                    else:
                        # No results file means no success
                        pass
                    
                    key = (category, model_name)
                    if key not in results_data:
                        results_data[key] = {"runs": 0, "success": 0}
                    
                    results_data[key]["runs"] += 1
                    if is_success:
                        results_data[key]["success"] += 1

                except Exception as e:
                    print(f"Error processing path {run_path}: {e}")

    # Prepare output list
    output_list = []
    # Sort for consistent output
    sorted_keys = sorted(results_data.keys())
    
    for category, model in sorted_keys:
        stats = results_data[(category, model)]
        num_runs = stats["runs"]
        num_success = stats["success"]
        success_rate = num_success / num_runs if num_runs > 0 else 0.0
        
        output_list.append({
            "model_name": model,
            "category": category,
            "num_runs": num_runs,
            "num_success": num_success,
            "success_rate": success_rate
        })

    # Write to JSON
    try:
        with open(output_file, 'w') as f:
            json.dump(output_list, f, indent=4)
        print(f"Successfully wrote summary to {output_file}")
    except Exception as e:
        print(f"Error writing JSON file: {e}")

    # Write to CSV
    try:
        with open(output_csv_file, 'w', newline='') as f:
            fieldnames = ["model_name", "category", "num_runs", "num_success", "success_rate"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_list)
        print(f"Successfully wrote summary to {output_csv_file}")
    except Exception as e:
        print(f"Error writing CSV file: {e}")

if __name__ == "__main__":
    main()
