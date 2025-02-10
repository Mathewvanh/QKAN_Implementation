from typing import List, Dict

import numpy as np
import torch
from torchvision import datasets


def analyze_mnist_sample(x_train: torch.Tensor, y_train_labels: torch.Tensor,
                         original_dataset: datasets.MNIST = None):
    """
    Analyze properties of MNIST sample compared to full dataset

    Args:
        x_train: Selected training images [N, 784]
        y_train_labels: Selected labels [N]
        original_dataset: Full MNIST training dataset
    """
    # Class distribution analysis
    class_counts = torch.bincount(y_train_labels, minlength=10)
    class_percentages = class_counts / len(y_train_labels) * 100

    print("\nClass Distribution:")
    for digit in range(10):
        print(f"Digit {digit}: {class_counts[digit]} samples ({class_percentages[digit]:.1f}%)")

    # Compare to full dataset if available
    if original_dataset is not None:
        full_counts = torch.bincount(original_dataset.targets, minlength=10)
        full_percentages = full_counts / len(original_dataset) * 100

        print("\nComparison with Full Dataset:")
        max_diff = 0
        for digit in range(10):
            diff = abs(class_percentages[digit] - full_percentages[digit])
            max_diff = max(max_diff, diff)
            print(f"Digit {digit}: Sample {class_percentages[digit]:.1f}% vs Full {full_percentages[digit]:.1f}% "
                  f"(diff: {diff:.1f}%)")
        print(f"\nMaximum distribution difference: {max_diff:.1f}%")

    # Basic statistics
    print("\nSample Statistics:")
    print(f"Total samples: {len(x_train)}")
    print(f"Min samples per class: {min(class_counts)}")
    print(f"Max samples per class: {max(class_counts)}")
    print(f"Std dev of class counts: {torch.std(class_counts.float()):.1f}")

    # Image statistics
    print("\nImage Statistics:")
    print(f"Mean pixel value: {x_train.mean():.3f}")
    print(f"Std dev pixel value: {x_train.std():.3f}")

    return {
        'class_counts': class_counts,
        'class_percentages': class_percentages,
        'statistics': {
            'total_samples': len(x_train),
            'min_samples': min(class_counts).item(),
            'max_samples': max(class_counts).item(),
            'std_dev': torch.std(class_counts.float()).item()
        }
    }

def compare_multiple_samples(dataset: datasets.MNIST, sample_size: int, num_runs: int = 5):
    """
    Compare multiple random samples to analyze consistency
    Returns dict with results_js and variation statistics
    """
    results = []

    for run in range(num_runs):
        print(f"\n=== Run {run + 1} ===")
        indices = torch.randperm(len(dataset))[:sample_size]
        x = dataset.data[indices].reshape(-1, 784).float() / 255.0
        y = dataset.targets[indices]

        result = analyze_mnist_sample(x, y, dataset)
        results.append(result)

    # Analyze variation across runs
    print("\n=== Cross-run Analysis ===")
    class_variations = torch.zeros(10)
    for digit in range(10):
        percentages = torch.tensor([r['class_percentages'][digit] for r in results])
        variation = torch.std(percentages)
        print(f"Digit {digit} percentage std dev across runs: {variation:.2f}%")
        class_variations[digit] = variation

    print(f"\nAverage class percentage variation: {class_variations.mean():.2f}%")
    print(f"Max class percentage variation: {class_variations.max():.2f}%")

    return {
        'sample_results': results,
        'variations': {
            'per_class': class_variations,
            'mean': class_variations.mean().item(),
            'max': class_variations.max().item()
        }
    }
def plot_sample_distributions(results: List[Dict]):
    """Plot class distributions across multiple runs"""
    import matplotlib.pyplot as plt

    num_runs = len(results)
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(10)  # digit classes
    width = 0.8 / num_runs  # width of bars

    for i, result in enumerate(results):
        percentages = result['class_percentages']
        ax.bar(x + i*width, percentages, width, label=f'Run {i+1}', alpha=0.7)

    ax.set_xlabel('Digit Class')
    ax.set_ylabel('Percentage in Sample')
    ax.set_title('Class Distribution Across Multiple Sampling Runs')
    ax.set_xticks(x + width * (num_runs-1)/2)
    ax.set_xticklabels(range(10))
    ax.legend()
    plt.grid(True, alpha=0.3)
    plt.show()