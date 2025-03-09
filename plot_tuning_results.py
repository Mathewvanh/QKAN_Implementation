import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from tuning_framework import KANTuner, MLPTuner, plot_timing_comparison

def plot_house_sales_results():
    """Generate plots from saved house sales tuning results."""
    print("\nPlotting House Sales Results")
    print("===========================")
    
    # Load saved results
    kan_results = pd.read_csv('tuning_results/house_sales_kan_tuning.csv')
    mlp_results = pd.read_csv('tuning_results/house_sales_mlp_tuning.csv')
    
    # Map network_shape to hidden_size in KAN results
    kan_results['hidden_size'] = kan_results['network_shape']
    
    # Create dummy tuners with just the results loaded
    kan_tuner = KANTuner(None, None, None, None)
    kan_tuner.results_df = kan_results
    
    mlp_tuner = MLPTuner(None, None, None, None)
    mlp_tuner.results_df = mlp_results
    
    # Generate comparison plots
    print("Generating comparison plots...")
    kan_tuner.plot_comparison(mlp_tuner, 'hidden_size', 'mean_val_metric', 'house_sales_mse_comparison.png')
    plot_timing_comparison(kan_tuner, mlp_tuner, "House Sales")
    
    # Print best configurations
    print("\nBest KAN configuration:")
    best_kan_idx = kan_results['mean_val_metric'].argmin()  # Use argmin for MSE
    best_kan = kan_results.iloc[best_kan_idx]
    for param, value in best_kan.items():
        if param not in ['mean_val_metric', 'std_val_metric', 'total_time', 'avg_epoch_time', 'qubo_time', 'avg_fold_time']:
            print(f"{param}: {value}")
    print(f"Best MSE: {best_kan['mean_val_metric']:.6f} ± {best_kan['std_val_metric']:.6f}")
    print(f"Training Time: {best_kan['total_time']:.2f}s")
    print(f"Avg Epoch Time: {best_kan['avg_epoch_time']:.4f}s")
    print(f"Avg QUBO Time: {best_kan['qubo_time']:.4f}s")
    print(f"Avg Fold Time: {best_kan['avg_fold_time']:.2f}s")
    
    print("\nBest MLP configuration:")
    best_mlp_idx = mlp_results['mean_val_metric'].argmin()  # Use argmin for MSE
    best_mlp = mlp_results.iloc[best_mlp_idx]
    for param, value in best_mlp.items():
        if param not in ['mean_val_metric', 'std_val_metric', 'total_time', 'avg_epoch_time', 'avg_fold_time']:
            print(f"{param}: {value}")
    print(f"Best MSE: {best_mlp['mean_val_metric']:.6f} ± {best_mlp['std_val_metric']:.6f}")
    print(f"Training Time: {best_mlp['total_time']:.2f}s")
    print(f"Avg Epoch Time: {best_mlp['avg_epoch_time']:.4f}s")
    print(f"Avg Fold Time: {best_mlp['avg_fold_time']:.2f}s")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create directories if they don't exist
    os.makedirs("tuning_plots", exist_ok=True)
    
    print("Generating plots from saved tuning results...")
    plot_house_sales_results()
