import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from tuning_framework import KANTuner, MLPTuner, plot_timing_comparison
import re

def extract_score_from_filename(filename):
    """Extract the score from a model filename."""
    # Pattern matches numbers like 0_037141 before .pth
    match = re.search(r'_(\d+_\d+)\.pth?$', filename)
    if match:
        # Convert score like "0_037141" to float "0.037141"
        score_str = match.group(1).replace('_', '.')
        return float(score_str)
    return None

def plot_jane_street_results():
    """Generate plots from saved Jane Street tuning results."""
    print("\nPlotting Jane Street Results")
    print("===========================")
    
    # Load saved results and fix column names
    kan_results = pd.read_csv('tuning_results/jane_street_kan_tuning.csv')
    mlp_results = pd.read_csv('tuning_results/jane_street_mlp_tuning.csv')
    
    # Fix column names by removing any whitespace
    kan_results.columns = [col.strip() for col in kan_results.columns]
    mlp_results.columns = [col.strip() for col in mlp_results.columns]
    
    # Map network_shape to hidden_size in KAN results
    kan_results['hidden_size'] = kan_results['network_shape']
    
    # Get scores from saved models
    kan_models = [f for f in os.listdir('tuning_results') if f.startswith('jane_street_best_kan_')]
    mlp_models = [f for f in os.listdir('tuning_results') if f.startswith('jane_street_best_mlp_')]
    
    kan_scores = [extract_score_from_filename(f) for f in kan_models]
    mlp_scores = [extract_score_from_filename(f) for f in mlp_models]
    
    # Filter out None values
    kan_scores = [s for s in kan_scores if s is not None]
    mlp_scores = [s for s in mlp_scores if s is not None]
    
    # Add best saved model scores to results
    if kan_scores:
        best_kan_score = max(kan_scores)
        # Add a row for the best saved model score
        best_kan_row = kan_results.iloc[0].copy()  # Copy config from best CSV result
        best_kan_row['mean_val_metric'] = best_kan_score
        best_kan_row['std_val_metric'] = 0  # Single score, no std
        kan_results = pd.concat([kan_results, pd.DataFrame([best_kan_row])], ignore_index=True)
        
        best_kan_file = [f for f in kan_models if str(best_kan_score).replace('.', '_') in f][0]
        print(f"\nBest KAN model: {best_kan_file}")
        print(f"Best KAN validation R² score: {best_kan_score:.6f}")
    
    if mlp_scores:
        best_mlp_score = max(mlp_scores)
        # Add a row for the best saved model score
        best_mlp_row = mlp_results.iloc[0].copy()  # Copy config from best CSV result
        best_mlp_row['mean_val_metric'] = best_mlp_score
        best_mlp_row['std_val_metric'] = 0  # Single score, no std
        mlp_results = pd.concat([mlp_results, pd.DataFrame([best_mlp_row])], ignore_index=True)
        
        best_mlp_file = [f for f in mlp_models if str(best_mlp_score).replace('.', '_') in f][0]
        print(f"Best MLP model: {best_mlp_file}")
        print(f"Best MLP validation R² score: {best_mlp_score:.6f}")
    
    # Create dummy tuners with updated results
    kan_tuner = KANTuner(None, None, None, None)
    kan_tuner.results_df = kan_results
    
    mlp_tuner = MLPTuner(None, None, None, None)
    mlp_tuner.results_df = mlp_results
    
    # Generate comparison plots
    print("\nGenerating comparison plots (including best saved model scores)...")
    kan_tuner.plot_comparison(mlp_tuner, 'hidden_size', 'mean_val_metric', 'jane_street_r2_comparison.png')
    plot_timing_comparison(kan_tuner, mlp_tuner, "Jane Street")
    
    # Print best configurations from CSV
    print("\nBest configurations from tuning results:")
    print("\nBest KAN configuration:")
    best_kan_idx = kan_results['mean_val_metric'].argmax()  # Use argmax for R² score
    best_kan = kan_results.iloc[best_kan_idx]
    for param, value in best_kan.items():
        if param not in ['mean_val_metric', 'std_val_metric', 'total_time', 'avg_epoch_time', 'qubo_time', 'avg_fold_time']:
            print(f"{param}: {value}")
    print(f"CSV validation R² Score: {best_kan['mean_val_metric']:.6f} ± {best_kan['std_val_metric']:.6f}")
    print(f"Training Time: {best_kan['total_time']:.2f}s")
    print(f"Avg Epoch Time: {best_kan['avg_epoch_time']:.4f}s")
    print(f"Avg QUBO Time: {best_kan['qubo_time']:.4f}s")
    print(f"Avg Fold Time: {best_kan['avg_fold_time']:.2f}s")
    
    print("\nBest MLP configuration:")
    best_mlp_idx = mlp_results['mean_val_metric'].argmax()  # Use argmax for R² score
    best_mlp = mlp_results.iloc[best_mlp_idx]
    for param, value in best_mlp.items():
        if param not in ['mean_val_metric', 'std_val_metric', 'total_time', 'avg_epoch_time', 'avg_fold_time']:
            print(f"{param}: {value}")
    print(f"CSV validation R² Score: {best_mlp['mean_val_metric']:.6f} ± {best_mlp['std_val_metric']:.6f}")
    print(f"Training Time: {best_mlp['total_time']:.2f}s")
    print(f"Avg Epoch Time: {best_mlp['avg_epoch_time']:.4f}s")
    print(f"Avg Fold Time: {best_mlp['avg_fold_time']:.2f}s")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create directories if they don't exist
    os.makedirs("tuning_plots", exist_ok=True)
    
    print("Generating plots from saved tuning results...")
    plot_jane_street_results()
