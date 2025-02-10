import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List, Optional

from TorchDegreeOptimizer import DegreeOptimizer, DegreeOptimizerConfig


def plot_degree_optimization(
    optimizer: DegreeOptimizer,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    title: str = "Degree Optimization Results"
) -> None:
    """Plot degree optimization results_js"""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Input data and approximation
    plt.subplot(1, 3, 1)
    x_np = x_data.detach().cpu().numpy()
    y_np = y_data.detach().cpu().numpy()
    
    plt.scatter(x_np, y_np, alpha=0.5, label='Data')
    plt.title('Data and Approximation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    
    # Plot 2: Degree distribution
    plt.subplot(1, 3, 2)
    degrees = [d for sublist in optimizer.optimal_degrees for d in sublist]
    plt.hist(degrees, bins=range(max(degrees) + 2), alpha=0.7)
    plt.title('Distribution of Selected Degrees')
    plt.xlabel('Degree')
    plt.ylabel('Count')
    
    # Plot 3: Error metrics vs degree
    plt.subplot(1, 3, 3)
    scores, comp_r2 = optimizer.evaluate_degree(x_data, y_data)
    degrees = range(len(scores))
    
    plt.plot(degrees, scores.detach().cpu().numpy(), 'b-', label='MSE')
    plt.plot(degrees, comp_r2.detach().cpu().numpy(), 'r--', label='R²')
    plt.title('Error Metrics vs Degree')
    plt.xlabel('Degree')
    plt.ylabel('Error')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_degree_optimization_improved(
        optimizer: DegreeOptimizer,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        title: str = "Degree Optimization Results"
) -> None:
    """Plot degree optimization results_js with predictions for each degree"""
    plt.figure(figsize=(15, 10))

    # Plot 1: Data and predictions for each degree
    plt.subplot(2, 2, 1)
    x_np = x_data.detach().cpu().numpy()
    y_np = y_data.detach().cpu().numpy()

    # Plot original data
    plt.scatter(x_np, y_np, alpha=0.3, label='Data', color='gray')

    # Get predictions for each degree
    colors = plt.cm.viridis(np.linspace(0, 1, optimizer.max_degree + 1))
    scores = []
    predictions = []

    # Sort x for smooth lines
    sort_idx = torch.argsort(x_data.squeeze())
    x_sorted = x_data[sort_idx]

    for d in range(optimizer.max_degree + 1):
        # Get transforms up to this degree
        transforms = []
        for degree in range(d + 1):
            degree_transform = optimizer._compute_transforms(x_sorted)[degree]
            transforms.append(degree_transform.reshape(len(x_sorted), -1))

        # Stack features and predict
        X = torch.cat(transforms, dim=1) if transforms else torch.zeros((len(y_data), 0), device=x_data.device)
        coeffs = torch.linalg.lstsq(X, y_data[sort_idx]).solution
        y_pred = X @ coeffs

        # Plot prediction
        plt.plot(x_sorted.cpu().numpy(), y_pred.detach().cpu().numpy(),
                 '--', alpha=0.7, color=colors[d], label=f'Degree {d}')

        # Store for metrics
        predictions.append(y_pred)
        scores.append(optimizer._compute_metrics(y_data[sort_idx], y_pred)['mse'])

    plt.title('Data and Predictions by Degree')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot 2: Best fit highlighted
    plt.subplot(2, 2, 2)
    plt.scatter(x_np, y_np, alpha=0.3, label='Data', color='gray')

    # Get best degree
    best_degree = np.argmin(scores)
    plt.plot(x_sorted.cpu().numpy(), predictions[best_degree].detach().cpu().numpy(),
             'r-', linewidth=2, label=f'Best Fit (Degree {best_degree})')

    plt.title(f'Best Fit (Degree {best_degree})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    # Plot 3: Distribution of selected degrees
    plt.subplot(2, 2, 3)
    degrees = [d for sublist in optimizer.optimal_degrees for d in sublist]
    plt.hist(degrees, bins=range(max(degrees) + 2), alpha=0.7)
    plt.axvline(x=best_degree, color='r', linestyle='--',
                label=f'Best Degree ({best_degree})')
    plt.title('Distribution of Selected Degrees')
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.legend()

    # Plot 4: Error metrics vs degree
    plt.subplot(2, 2, 4)
    scores = torch.tensor(scores)
    degrees = range(len(scores))

    plt.plot(degrees, scores.cpu().numpy(), 'b-', label='MSE')
    plt.axvline(x=best_degree, color='r', linestyle='--',
                label=f'Best Degree ({best_degree})')
    plt.title('Error Metrics vs Degree')
    plt.xlabel('Degree')
    plt.ylabel('Error')
    plt.yscale('log')  # Log scale to better see differences
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
def run_interactive_test(
    test_functions: List[callable],
    optimizer: DegreeOptimizer,
    n_samples: int = 1000,
    noise_level: float = 0.1
) -> None:
    """Run interactive tests with visualization"""
    device = next(optimizer.parameters()).device
    
    for i, func in enumerate(test_functions):
        # Generate data
        x = torch.linspace(-1, 1, n_samples, device=device).reshape(-1, 1)
        y = func(x)
        if noise_level > 0:
            y += noise_level * torch.randn_like(y)
            
        # Optimize and plot
        optimizer.fit(x, y)
        results = optimizer.analyze_network(x, y)
        optimizer.visualize_analysis(results,x, y)
        plot_degree_optimization_improved(
            optimizer, x, y,
            title=f'Test Function {i+1}: {func.__name__}'
        )
        
# Example usage:
if __name__ == "__main__":
    config = DegreeOptimizerConfig(
        network_shape=[1, 10,10, 1],
        max_degree=7,
        complexity_weight=0.1,
        significance_weight=0.05
    )
    optimizer = DegreeOptimizer(config)
    
    # Define test functions
    def polynomial(x: torch.Tensor) -> torch.Tensor:
        return -1.23451 * x**7 + torch.exp(x)
        
    def sinusoidal(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(2*torch.pi*x**2) + torch.cos(2*torch.pi*x**2)

    def complex_function(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(2*torch.pi*x) + 0.5*x**2 + torch.exp(x**(10))

    def inverse_sin(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(1 / x ** 2)
    test_functions = [polynomial, sinusoidal, complex_function, inverse_sin]
    
    # Run tests with visualization
    run_interactive_test(test_functions, optimizer)