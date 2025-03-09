#!/usr/bin/env python
"""
Script to run the architecture comparison between different KAN implementations.
"""

import os
import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run KAN architecture comparison')
    parser.add_argument('--subset', type=str, default=None, 
                       choices=['polynomial', 'wavelet', 'fast', 'all'],
                       help='Run a subset of architectures: polynomial (Chebyshev, Jacobi, Taylor), wavelet (Wavelet, Fourier), fast (for quick testing), or all (default)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--optimization_method', type=str, default='Evolutionary',
                       choices=['QUBO', 'IntegerProgramming', 'Evolutionary', 'GreedyHeuristic'],
                       help='Optimization method for the original KAN')
    parser.add_argument('--data_path', type=str, 
                       default="~/Interning/Kaggle/jane_street_kaggle/jane-street-real-time-market-data-forecasting/train.parquet/",
                       help='Path to Jane Street dataset')
    args = parser.parse_args()

    # Create results directory with info about what we're running
    subset_name = args.subset if args.subset else "all"
    opt_method = args.optimization_method.lower()
    results_dir = f"architecture_comparison_{subset_name}_{opt_method}_{args.epochs}ep"
    
    print("\n" + "="*60)
    print(f"Running KAN Architecture Comparison")
    print(f"Architecture subset: {subset_name}")
    print(f"Optimization method: {args.optimization_method}")
    print(f"Epochs: {args.epochs}")
    print(f"Results will be saved in: {results_dir}")
    print("="*60 + "\n")
    
    # Construct command with appropriate options
    cmd = [
        "python", "compare_architectures.py",
        "--results_dir", results_dir,
        "--epochs", str(args.epochs),
        "--optimization_method", args.optimization_method,
        "--data_path", args.data_path
    ]
    
    # Add subset parameter if specified
    if args.subset:
        cmd.extend(["--subset", args.subset])
    
    # Add --force to skip confirmation
    cmd.append("--force")
    
    # Ask for confirmation
    confirm = input("Ready to start? This may take several hours. (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted")
        return
    
    # Run the comparison
    try:
        subprocess.run(cmd, check=True)
        print(f"\nArchitecture comparison completed successfully!")
        print(f"Results saved in {results_dir}")
        print(f"To view the plots, check the PNG files in the {results_dir} directory")
    except subprocess.CalledProcessError as e:
        print(f"Error running comparison: {e}")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")

if __name__ == "__main__":
    main() 