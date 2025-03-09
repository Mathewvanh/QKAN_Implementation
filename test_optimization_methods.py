import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import time
import os

from CP_KAN import FixedKAN, FixedKANConfig
from optimization_tuner import OptimizationTuner

def test_methods_with_restricted_grid():
    """Test all optimization methods with a smaller grid to verify behavior."""
    # Create synthetic data for testing
    np.random.seed(42)
    x_data = torch.tensor(np.random.randn(100, 10), dtype=torch.float32)
    y_data = torch.tensor(np.random.randn(100, 1), dtype=torch.float32)
    
    # Define a small grid
    param_grid = {
        'max_degree': [3, 5],  # Only 2 values
        'hidden_size': [16, 32],  # Only 2 values
        'hidden_degree': [2],  # Fixed
        'learning_rate': [0.01]  # Fixed
    }
    
    # Method names
    methods = ['QUBO', 'IntegerProgramming', 'Evolutionary', 'GreedyHeuristic']
    
    # Store results
    results = {}
    
    # For each method, run all configurations and store results
    for method in methods:
        results[method] = []
        print(f"\n=== Testing {method} ===")
        
        for max_degree in param_grid['max_degree']:
            for hidden_size in param_grid['hidden_size']:
                config = {
                    'max_degree': max_degree,
                    'hidden_size': hidden_size,
                    'hidden_degree': param_grid['hidden_degree'][0],
                    'learning_rate': param_grid['learning_rate'][0]
                }
                
                print(f"Configuration: {config}")
                
                # Create KAN model
                kan_config = FixedKANConfig(
                    network_shape=[10, hidden_size, 1],
                    max_degree=max_degree,
                    complexity_weight=0.0,
                    trainable_coefficients=True,
                    skip_qubo_for_hidden=False if method == 'QUBO' else True,
                    default_hidden_degree=param_grid['hidden_degree'][0]
                )
                
                kan = FixedKAN(kan_config)
                
                # Select optimization method
                if method == 'QUBO':
                    optimize_fn = lambda kan, x, y: kan.optimize(x, y)
                elif method == 'IntegerProgramming':
                    optimize_fn = lambda kan, x, y: kan.optimize_integer_programming(x, y)
                elif method == 'Evolutionary':
                    optimize_fn = lambda kan, x, y: kan.optimize_evolutionary(x, y)
                elif method == 'GreedyHeuristic':
                    optimize_fn = lambda kan, x, y: kan.optimize_greedy_heuristic(x, y)
                
                # Run optimization
                start_time = time.time()
                optimize_fn(kan, x_data, y_data)
                opt_time = time.time() - start_time
                
                # Extract and save selected degrees
                selected_degrees = []
                for layer in kan.layers:
                    layer_degrees = []
                    for neuron in layer.neurons:
                        layer_degrees.append(int(neuron.selected_degree.item()))
                    selected_degrees.append(layer_degrees)
                
                result = {
                    'config': config,
                    'opt_time': opt_time,
                    'selected_degrees': selected_degrees
                }
                
                print(f"Selected degrees: {selected_degrees}")
                print(f"Optimization time: {opt_time:.4f} seconds")
                print("-" * 50)
                
                results[method].append(result)
    
    # Save results to file
    os.makedirs('test_results', exist_ok=True)
    with open('test_results/method_behavior_test.json', 'w') as f:
        # Convert to serializable format
        serializable_results = {
            method: [
                {
                    'config': r['config'],
                    'opt_time': float(r['opt_time']),
                    'selected_degrees': [[int(d) for d in layer] for layer in r['selected_degrees']]
                }
                for r in results[method]
            ]
            for method in results
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to test_results/method_behavior_test.json")
    
    # Print summary for quick comparison
    print("\n=== Summary ===")
    for method in methods:
        selected_degree_sets = []
        for result in results[method]:
            # Only look at output layer degrees which is selected_degrees[-1]
            output_degrees = tuple(result['selected_degrees'][-1])
            selected_degree_sets.append(output_degrees)
        
        unique_degree_sets = set(selected_degree_sets)
        print(f"{method}: {len(unique_degree_sets)} unique output layer configurations")
        print(f"   Configurations: {unique_degree_sets}")

if __name__ == "__main__":
    test_methods_with_restricted_grid() 