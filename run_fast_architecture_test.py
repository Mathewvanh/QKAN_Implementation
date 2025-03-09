#!/usr/bin/env python
"""
Script to run a fast architecture comparison test focusing on 
OriginalKAN, Transformer, LSTM, and a few key KAN variants.
"""

import os
import subprocess
import sys
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Run a fast architecture comparison test')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--optimization_method', type=str, default='Evolutionary',
                       choices=['QUBO', 'IntegerProgramming', 'Evolutionary', 'GreedyHeuristic'],
                       help='Optimization method for the original KAN')
    parser.add_argument('--data_path', type=str, 
                       default="~/Interning/Kaggle/jane_street_kaggle/jane-street-real-time-market-data-forecasting/train.parquet/",
                       help='Path to Jane Street dataset')
    parser.add_argument('--min_r2', type=float, default=-5.0, 
                        help='Minimum R² to include in plots (to filter extreme negative values)')
    args = parser.parse_args()

    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"architecture_comparison_quick_{timestamp}"
    
    print("\n" + "="*60)
    print(f"Running Quick Architecture Comparison")
    print(f"Testing OriginalKAN, Transformer, LSTM, and key KAN variants")
    print(f"Optimization method: {args.optimization_method}")
    print(f"Epochs: {args.epochs}")
    print(f"Results will be saved in: {results_dir}")
    print("="*60 + "\n")
    
    # Generate subset parameter - we need a specific list of architectures
    subset = "fast"  # This will include OriginalKAN, TaylorKAN, and RBFKAN
    
    # Run the comparison
    compare_cmd = [
        "python", "compare_architectures.py",
        "--results_dir", results_dir,
        "--epochs", str(args.epochs),
        "--optimization_method", args.optimization_method,
        "--data_path", args.data_path,
        "--subset", subset,
        "--force"
    ]
    
    print("Command to run:")
    print(" ".join(compare_cmd))
    print()
    
    # Ask for confirmation
    confirm = input("Ready to start the quick architecture test? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted")
        return

    # Run the comparison
    try:
        print("\nRunning architecture comparison...")
        subprocess.run(compare_cmd, check=True)
        print("\nComparison completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nError running comparison: {e}")
        return
    
    # Generate plots
    plot_cmd = [
        "python", "plot_architecture_results.py",
        "--results_dir", results_dir,
        "--min_r2", str(args.min_r2)
    ]
    
    try:
        print("\nGenerating plots...")
        subprocess.run(plot_cmd, check=True)
        print("\nPlots generated successfully!")
        print(f"Results and plots are available in: {results_dir}")
        print(f"Summary plots are in: {os.path.join(results_dir, 'plots', 'comparison')}")
    except subprocess.CalledProcessError as e:
        print(f"\nError generating plots: {e}")
    
if __name__ == "__main__":
    main() 