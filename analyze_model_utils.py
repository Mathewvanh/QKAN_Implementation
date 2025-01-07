import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt


def load_model_and_metadata(model_file: str, json_file: str) -> Tuple[Dict, Dict]:
    """Load a saved model state and its metadata"""
    # Load raw model data without constructing model
    model_data = torch.load(model_file)

    # Load metadata
    with open(json_file, 'r') as f:
        metadata = json.load(f)

    return model_data, metadata

def extract_degrees_from_state(state_dict: Dict) -> List[List[int]]:
    """Extract polynomial degrees from model state"""
    degrees_by_layer = []
    layer_idx = 0

    while True:
        layer_degrees = []
        neuron_idx = 0

        # Try to find degrees for current layer - checking both formats
        while True:
            # Try both potential key formats
            degree_keys = [
                f'layers.{layer_idx}.neurons.{neuron_idx}._selected_degree',  # Original format
                f'layers.{layer_idx}.neurons.{neuron_idx}.selected_degree',   # Alternative format
            ]

            # Find first matching key
            degree_key = next((key for key in degree_keys if key in state_dict), None)

            if degree_key is None:
                break

            layer_degrees.append(state_dict[degree_key].item())
            neuron_idx += 1

        if not layer_degrees:
            break

        degrees_by_layer.append(layer_degrees)
        layer_idx += 1

    return degrees_by_layer

def extract_coefficients_from_state(state_dict: Dict) -> List[List[np.ndarray]]:
    """Extract polynomial coefficients from model state"""
    coeffs_by_layer = []
    layer_idx = 0

    while True:
        layer_coeffs = []
        neuron_idx = 0

        # Try to find coefficients for current layer - checking both formats
        while True:
            # Try both potential key formats
            coeff_keys = [
                f'layers.{layer_idx}.neurons.{neuron_idx}._coefficients',  # Original format
                f'layers.{layer_idx}.neurons.{neuron_idx}.coefficients',   # Alternative format
            ]

            # Find first matching key
            coeff_key = next((key for key in coeff_keys if key in state_dict), None)

            if coeff_key is None:
                break

            coeffs = state_dict[coeff_key].numpy()
            layer_coeffs.append(coeffs)
            neuron_idx += 1

        if not layer_coeffs:
            break

        coeffs_by_layer.append(layer_coeffs)
        layer_idx += 1

    return coeffs_by_layer

def analyze_model(model_data: Dict, metadata: Dict) -> Dict:
    """Analyze a single model's structure"""
    # Handle both possible formats for state dict
    state_dict = model_data.get('model_state_dict', model_data)

    # Get degrees and coefficients
    degrees = extract_degrees_from_state(state_dict)
    coefficients = extract_coefficients_from_state(state_dict)

    # Rest of the function remains the same...
    layer_stats = []
    for layer_idx, (layer_degrees, layer_coeffs) in enumerate(zip(degrees, coefficients)):
        all_coeffs = np.concatenate([c.flatten() for c in layer_coeffs])

        layer_stats.append({
            'layer': layer_idx,
            'degrees': {
                'mean': np.mean(layer_degrees),
                'std': np.std(layer_degrees),
                'min': min(layer_degrees),
                'max': max(layer_degrees),
                'counts': np.bincount(layer_degrees),
                'total_neurons': len(layer_degrees)
            },
            'coefficients': {
                'mean': np.mean(all_coeffs),
                'std': np.std(all_coeffs),
                'max_abs': np.max(np.abs(all_coeffs)),
                'sparsity': np.mean(np.abs(all_coeffs) < 1e-6),
                'histogram': np.histogram(all_coeffs, bins=50),
                'abs_histogram': np.histogram(np.abs(all_coeffs), bins=50)
            }
        })

    return {
        'layer_stats': layer_stats,
        'config': model_data.get('config', {}),
        'performance': metadata.get('metrics', {})
    }
def plot_comparisons(analyses: Dict[str, Dict]):
    """Create comparison plots between models"""
    n_models = len(analyses)
    n_layers = len(next(iter(analyses.values()))['layer_stats'])

    # 1. Degree Distribution Plot
    fig, axes = plt.subplots(n_layers, 1, figsize=(15, 5*n_layers))
    if n_layers == 1:
        axes = [axes]

    for layer_idx, ax in enumerate(axes):
        for model_name, analysis in analyses.items():
            stats = analysis['layer_stats'][layer_idx]['degrees']
            degrees = range(len(stats['counts']))
            normalized_counts = stats['counts'] / stats['total_neurons']
            ax.bar(degrees, normalized_counts, alpha=0.5, label=f"{model_name}")

        ax.set_title(f"Layer {layer_idx} Degree Distribution")
        ax.set_xlabel("Polynomial Degree")
        ax.set_ylabel("Fraction of Neurons")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Polynomial Degree Distribution Comparison")
    plt.tight_layout()
    plt.show()

    # 2. Coefficient Distribution Plot
    fig, axes = plt.subplots(n_layers, 2, figsize=(20, 5*n_layers))

    for layer_idx in range(n_layers):
        # Regular coefficient distribution
        for model_name, analysis in analyses.items():
            stats = analysis['layer_stats'][layer_idx]['coefficients']
            counts, bins = stats['histogram']
            axes[layer_idx, 0].hist(bins[:-1], bins, weights=counts, alpha=0.5,
                                  label=f"{model_name}\nSparsity: {stats['sparsity']:.2%}",
                                  density=True)

        axes[layer_idx, 0].set_title(f"Layer {layer_idx} Coefficient Distribution")
        axes[layer_idx, 0].set_xlabel("Coefficient Value")
        axes[layer_idx, 0].set_ylabel("Density")
        axes[layer_idx, 0].legend()
        axes[layer_idx, 0].grid(True, alpha=0.3)

        # Absolute coefficient distribution (log scale)
        for model_name, analysis in analyses.items():
            stats = analysis['layer_stats'][layer_idx]['coefficients']
            counts, bins = stats['abs_histogram']
            axes[layer_idx, 1].hist(bins[:-1], bins, weights=counts, alpha=0.5,
                                  label=f"{model_name}", density=True)

        axes[layer_idx, 1].set_title(f"Layer {layer_idx} Absolute Coefficient Distribution")
        axes[layer_idx, 1].set_xlabel("|Coefficient|")
        axes[layer_idx, 1].set_ylabel("Density")
        axes[layer_idx, 1].set_yscale('log')
        axes[layer_idx, 1].legend()
        axes[layer_idx, 1].grid(True, alpha=0.3)

    plt.suptitle("Coefficient Distribution Comparison")
    plt.tight_layout()
    plt.show()

def analyze_models(model_files: List[str], json_files: List[str]):
    """Main analysis function"""
    analyses = {}

    for model_file, json_file in zip(model_files, json_files):
        # Load data
        model_data, metadata = load_model_and_metadata(model_file, json_file)
        acc = metadata['metrics']['test_accuracy']
        name = f"Acc: {acc:.1%}"

        # Analyze model
        analyses[name] = analyze_model(model_data, metadata)

        # Print summary statistics
        print(f"\nModel: {name}")
        print(f"Network Shape: {metadata['network_shape']}")
        print(f"Max Degree: {metadata['max_degree']}")
        print(f"Complexity Weight: {metadata.get('complexity_weight', 'Not specified')}")

        print("\nDegree Statistics by Layer:")
        for layer_idx, stats in enumerate(analyses[name]['layer_stats']):
            deg_stats = stats['degrees']
            print(f"Layer {layer_idx}:")
            print(f"  Mean Degree: {deg_stats['mean']:.2f}")
            print(f"  Degree Range: {deg_stats['min']} - {deg_stats['max']}")

        print("\nCoefficient Statistics by Layer:")
        for layer_idx, stats in enumerate(analyses[name]['layer_stats']):
            coeff_stats = stats['coefficients']
            print(f"Layer {layer_idx}:")
            print(f"  Mean Coeff: {coeff_stats['mean']:.3f}")
            print(f"  Max Abs Coeff: {coeff_stats['max_abs']:.3f}")
            print(f"  Sparsity: {coeff_stats['sparsity']:.2%}")

    # Create comparison plots
    plot_comparisons(analyses)

    return analyses

if __name__ == "__main__":
    # Example usage
    model_files = [
        'models/mnist_kan_model_0.5127.pt',
        'models/mnist_kan_model_0.2130.pt'
    ]

    json_files = [
        'mnist_kan_results_acc_0.5127_11-37-11.json',
        'mnist_kan_results_acc_0.2130_11-41-15.json'
    ]

    analyze_models(model_files, json_files)