#!/usr/bin/env python3
"""
Fix CSV files by removing lines that don't start with timestamps.
"""

import os
import re
from pathlib import Path

# Pattern to match timestamp at start of line
TIMESTAMP_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}')

def fix_csv_file(file_path):
    """Remove lines that don't start with a timestamp (after header)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            return 0  # Nothing to fix
        
        # Keep header
        fixed_lines = [lines[0]]
        removed_count = 0
        
        # Process data lines
        for line in lines[1:]:
            if TIMESTAMP_PATTERN.match(line):
                fixed_lines.append(line)
            else:
                removed_count += 1
        
        if removed_count > 0:
            # Write back the fixed content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)
            return removed_count
        
        return 0
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

def main():
    base_dir = Path("results")
    
    if not base_dir.exists():
        print(f"Directory {base_dir} not found")
        return
    
    total_files = 0
    total_fixed = 0
    total_lines_removed = 0
    
    for csv_file in base_dir.rglob("*.csv"):
        total_files += 1
        removed = fix_csv_file(csv_file)
        if removed > 0:
            total_fixed += 1
            total_lines_removed += removed
            print(f"Fixed {csv_file.name}: removed {removed} lines")
    
    print(f"\n{'='*60}")
    print(f"Processed {total_files} CSV files")
    print(f"Fixed {total_fixed} files")
    print(f"Removed {total_lines_removed} invalid lines")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
