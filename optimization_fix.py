import torch
import numpy as np
from CP_KAN import FixedKAN, FixedKANConfig
import time
import os
import json
import matplotlib.pyplot as plt

def test_fixed_optimization_methods():
    """Test optimization methods with a fix to make them more responsive to hyperparameters."""
    # Create synthetic data
    np.random.seed(42)
    x_data = torch.tensor(np.random.randn(100, 10), dtype=torch.float32)
    y_data = torch.tensor(np.random.randn(100, 1), dtype=torch.float32)
    
    # Method names
    methods = ['QUBO', 'IntegerProgramming', 'Evolutionary', 'GreedyHeuristic']
    
    # Test configurations
    max_degrees = [3, 5, 7]
    results = {}
    
    # Key fix: Don't use skip_qubo_for_hidden for any method
    for method in methods:
        results[method] = []
        print(f"\n=== Testing {method} with Fix ===")
        
        for max_degree in max_degrees:
            print(f"Configuration: max_degree={max_degree}")
            
            # Create KAN model
            kan_config = FixedKANConfig(
                network_shape=[10, 16, 1],
                max_degree=max_degree,
                complexity_weight=0.0,
                trainable_coefficients=True,
                skip_qubo_for_hidden=False,  # Key fix: don't skip for any method
                default_hidden_degree=3
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
                'max_degree': max_degree,
                'opt_time': opt_time,
                'selected_degrees': selected_degrees
            }
            
            print(f"Selected degrees:")
            for i, layer_degrees in enumerate(selected_degrees):
                if i == 0:
                    print(f"  - Hidden Layer: {layer_degrees[:5]}... (showing first 5 of {len(layer_degrees)})")
                else:
                    print(f"  - Output Layer: {layer_degrees}")
            print(f"Optimization time: {opt_time:.4f} seconds")
            print("-" * 50)
            
            results[method].append(result)
    
    # Save results
    os.makedirs('test_results', exist_ok=True)
    with open('test_results/fixed_method_comparison.json', 'w') as f:
        # Convert to serializable format
        serializable_results = {
            method: [
                {
                    'max_degree': r['max_degree'],
                    'opt_time': float(r['opt_time']),
                    'selected_degrees': [[int(d) for d in layer] for layer in r['selected_degrees']]
                }
                for r in results[method]
            ]
            for method in results
        }
        json.dump(serializable_results, f, indent=2)
    
    # Plot the results
    plt.figure(figsize=(14, 10))
    
    # Set up colors and markers
    colors = {'QUBO': 'blue', 'IntegerProgramming': 'green', 
              'Evolutionary': 'orange', 'GreedyHeuristic': 'red'}
    
    # 1. Plot max_degree vs. average selected degree for output layer
    plt.subplot(2, 2, 1)
    for method in methods:
        x_vals = [r['max_degree'] for r in results[method]]
        # Get output layer average degree (selected_degrees[-1])
        y_vals = [np.mean(r['selected_degrees'][-1]) for r in results[method]]
        plt.plot(x_vals, y_vals, 'o-', label=method, color=colors[method], linewidth=2)
    
    plt.xlabel('max_degree Parameter')
    plt.ylabel('Average Selected Degree (Output Layer)')
    plt.title('Responsiveness to max_degree Parameter')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. Plot optimization times
    plt.subplot(2, 2, 2)
    for method in methods:
        x_vals = [r['max_degree'] for r in results[method]]
        y_vals = [r['opt_time'] for r in results[method]]
        plt.plot(x_vals, y_vals, 'o-', label=method, color=colors[method], linewidth=2)
    
    plt.xlabel('max_degree Parameter')
    plt.ylabel('Optimization Time (seconds)')
    plt.title('Optimization Time vs. max_degree')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 3. Plot distribution of selected degrees (only for max_degree=7)
    plt.subplot(2, 2, 3)
    max_degree_idx = max_degrees.index(max(max_degrees))  # Use the largest max_degree
    
    for i, method in enumerate(methods):
        # Get output layer degrees for max_degree=7
        degrees = results[method][max_degree_idx]['selected_degrees'][-1]
        # Count occurrences
        unique_degrees = sorted(set(degrees))
        counts = [degrees.count(d) for d in unique_degrees]
        
        # Offset the bars for better visibility
        offset = 0.1 * (i - 1.5)
        plt.bar([d + offset for d in unique_degrees], counts, 
                width=0.2, label=method, color=colors[method], alpha=0.7)
    
    plt.xlabel('Selected Degree Value')
    plt.ylabel('Count')
    plt.title(f'Distribution of Selected Degrees (max_degree={max(max_degrees)})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 4. Plot average selected degree for hidden layer
    plt.subplot(2, 2, 4)
    for method in methods:
        x_vals = [r['max_degree'] for r in results[method]]
        # Get hidden layer average degree (selected_degrees[0])
        y_vals = [np.mean(r['selected_degrees'][0]) for r in results[method]]
        plt.plot(x_vals, y_vals, 'o-', label=method, color=colors[method], linewidth=2)
    
    plt.xlabel('max_degree Parameter')
    plt.ylabel('Average Selected Degree (Hidden Layer)')
    plt.title('Hidden Layer Responsiveness')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('test_results/optimization_methods_comparison.png', dpi=300)
    print("Saved plot to test_results/optimization_methods_comparison.png")
    
    # Summary statistics
    print("\n=== Summary ===")
    for method in methods:
        output_degrees = []
        for r in results[method]:
            output_degrees.extend(r['selected_degrees'][-1])
        
        unique_degrees = set(output_degrees)
        print(f"{method}:")
        print(f"  - Unique degree values: {sorted(unique_degrees)}")
        print(f"  - Responds to max_degree: {'Yes' if len(unique_degrees) > 1 else 'No'}")
        print(f"  - Average optimization time: {np.mean([r['opt_time'] for r in results[method]]):.4f} seconds")

if __name__ == "__main__":
    test_fixed_optimization_methods() 