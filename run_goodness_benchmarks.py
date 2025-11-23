"""
Script to benchmark all goodness functions across all datasets.
This automates running experiments with different goodness functions.
"""

import subprocess
import sys
from pathlib import Path

# Import goodness registry to get all available functions
import goodness

# Define datasets to test
DATASETS = ["mnist", "fashionmnist", "cifar10", "stl10"]

# Get all registered goodness functions
GOODNESS_FUNCTIONS = goodness.list_available_goodness_functions()

def run_experiment(dataset, goodness_fn):
    """
    Run a single experiment with specified dataset and goodness function.
    
    Args:
        dataset: Name of the dataset
        goodness_fn: Name of the goodness function
    """
    print("\n" + "="*80)
    print(f"Running: Dataset={dataset}, Goodness Function={goodness_fn}")
    print("="*80 + "\n")
    
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "main.py",
        f"input.dataset={dataset}",
        f"model.goodness_function={goodness_fn}"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ Successfully completed: {dataset} with {goodness_fn}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {dataset} with {goodness_fn}")
        print(f"Error: {e}\n")
        return False
    except KeyboardInterrupt:
        print("\n\nBenchmarking interrupted by user.")
        sys.exit(1)


def main():
    """Run all goodness function benchmarks."""
    print("="*80)
    print("GOODNESS FUNCTION BENCHMARKING")
    print("="*80)
    print(f"\nDatasets to test: {', '.join(DATASETS)}")
    print(f"Goodness functions to test: {', '.join(GOODNESS_FUNCTIONS)}")
    print(f"\nTotal experiments: {len(DATASETS) * len(GOODNESS_FUNCTIONS)}")
    print("\n" + "="*80 + "\n")
    
    # Confirm before starting
    response = input("Do you want to proceed? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Benchmarking cancelled.")
        return
    
    results = {
        "success": [],
        "failed": []
    }
    
    total = len(DATASETS) * len(GOODNESS_FUNCTIONS)
    current = 0
    
    for dataset in DATASETS:
        for goodness_fn in GOODNESS_FUNCTIONS:
            current += 1
            print(f"\n[{current}/{total}] Testing {dataset} with {goodness_fn}...")
            
            success = run_experiment(dataset, goodness_fn)
            
            if success:
                results["success"].append((dataset, goodness_fn))
            else:
                results["failed"].append((dataset, goodness_fn))
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARKING SUMMARY")
    print("="*80)
    print(f"\nTotal experiments: {total}")
    print(f"Successful: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")
    
    if results["failed"]:
        print("\nFailed experiments:")
        for dataset, goodness_fn in results["failed"]:
            print(f"  - {dataset} with {goodness_fn}")
    
    print("\n" + "="*80)
    print("Results saved in:")
    print("  - results/ folder (JSON files)")
    print("  - emissions/ folder (CSV files)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
