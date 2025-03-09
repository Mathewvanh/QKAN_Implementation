#!/usr/bin/env python
"""
Helper script to run the fixed expanded grid search overnight.
This uses our fix to make all optimization methods responsive to hyperparameters.
"""

import os
import subprocess
import sys

def run_overnight():
    """Run the fixed expanded grid search for overnight execution"""
    
    # Default configurations
    methods = ['QUBO', 'IntegerProgramming', 'Evolutionary', 'GreedyHeuristic']
    epochs = 50
    results_dir = 'optimization_results_fixed_full'
    data_path = "~/Interning/Kaggle/jane_street_kaggle/jane-street-real-time-market-data-forecasting/train.parquet/"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        # Check if specific method is requested
        method_arg = sys.argv[1]
        if method_arg in methods:
            methods = [method_arg]
            # Create method-specific results dir
            results_dir = f'optimization_results_fixed_{method_arg.lower()}'
            print(f"Running for method: {method_arg}")
            print(f"Results will be saved in: {results_dir}")
        elif method_arg == "all":
            print("Running all methods")
        else:
            print(f"Unknown method: {method_arg}")
            print(f"Available methods: {methods} or 'all'")
            return
    
    # Check for custom epochs
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        epochs = int(sys.argv[2])
        print(f"Using custom epochs: {epochs}")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Construct the command
    cmd = [
        "python", "run_expanded_grid.py",
        "--methods"] + methods + [
        "--epochs", str(epochs),
        "--results_dir", results_dir,
        "--data_path", data_path,
        "--force"  # Skip confirmation
    ]
    
    # Print summary
    print("\n" + "="*50)
    print(f"Running fixed expanded grid search overnight")
    print(f"Methods: {methods}")
    print(f"Epochs: {epochs}")
    print(f"Results directory: {results_dir}")
    print(f"Data path: {data_path}")
    print("="*50 + "\n")
    
    # Print usage examples
    print("Usage examples:")
    print("  python run_fixed_overnight.py QUBO           # Run only QUBO method")
    print("  python run_fixed_overnight.py QUBO 30        # Run QUBO with 30 epochs")
    print("  python run_fixed_overnight.py all            # Run all methods")
    print("  python run_fixed_overnight.py all 20         # Run all methods with 20 epochs\n")
    
    # Confirm before running
    confirm = input("Ready to start? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted by user")
        return
    
    # Execute the command
    try:
        result = subprocess.run(cmd, check=True)
        print(f"Command completed with exit code: {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")

if __name__ == "__main__":
    run_overnight() 