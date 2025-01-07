from typing import Tuple

import torch
import pennylane as qml
from KAN_w_cumulative_polynomials import KANNeuron
import matplotlib.pyplot as plt

def generate_test_data(num_points: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate simple test data for quadratic function"""
    x = torch.linspace(-1, 1, num_points).reshape(-1, 1)
    y = 0.5 * x**2 + 0.3 * x + 0.1
    return x, y

def plot_comparison(x: torch.Tensor, y_true: torch.Tensor,
                    y_classical: torch.Tensor, y_quantum: torch.Tensor):
    """Plot comparison between classical and quantum results"""
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y_true, label='True', alpha=0.5)
    plt.plot(x, y_classical, label='Classical', linestyle='--')
    plt.plot(x, y_quantum, label='Quantum', linestyle=':')
    plt.legend()
    plt.title('Classical vs Quantum QSVT Solution')
    plt.show()

def test_single_neuron():
    # Create test data
    x_data, y_data = generate_test_data()

    # Initialize neuron
    neuron = KANNeuron(input_dim=1, max_degree=2)

    # Test both classical and quantum methods
    y_classical = neuron.optimize_classical(x_data, y_data)
    y_quantum = neuron.optimize_quantum(x_data, y_data)

    # Compare results
    plot_comparison(x_data, y_data, y_classical, y_quantum)

    # Print errors
    classical_mse = torch.mean((y_data - y_classical) ** 2)
    quantum_mse = torch.mean((y_data - y_quantum) ** 2)
    print(f"Classical MSE: {classical_mse:.6f}")
    print(f"Quantum MSE: {quantum_mse:.6f}")

if __name__ == "__main__":
    test_single_neuron()