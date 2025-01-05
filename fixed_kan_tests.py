import unittest
import torch
import numpy as np
from typing import Tuple

from KAN_w_cumulative_polynomials import FixedKANConfig, FixedKAN, KANNeuron


class TestFixedKAN(unittest.TestCase):
    def setUp(self):
        """Setup basic test configurations"""
        self.config = FixedKANConfig(
            network_shape=[1, 10, 1],  # Single input/output with 10 hidden neurons
            max_degree=7,
            complexity_weight=0.1
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    @staticmethod
    def target_function(x: torch.Tensor) -> torch.Tensor:
        # Initialize output tensor
        y = torch.zeros_like(x)

        # Create masks
        mask1 = x < 0.5
        mask2 = (x >= 0.5) & (x < 1.5)
        mask3 = x >= 1.5

        # Apply functions to each region
        y[mask1] = torch.sin(20 * torch.pi * x[mask1]) + x[mask1].pow(2)
        y[mask2] = 0.5 * x[mask2] * torch.exp(-x[mask2]) + torch.abs(torch.sin(5 * torch.pi * x[mask2]))
        y[mask3] = torch.log(x[mask3] - 1) / torch.log(torch.tensor(2.0)) - torch.cos(2 * torch.pi * x[mask3])

        # Add noise - using PyTorch's normal distribution
        noise = torch.normal(mean=0.0, std=0.2, size=y.shape, device=y.device)
        y += noise

        return y
    def generate_test_data(self, func: callable, n_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate test data for a given function"""
        x = torch.linspace(-1, 1, n_samples, device=self.device).reshape(-1, 1)
        y = func(x).reshape(-1, 1)
        return x, y

    def test_simple_function(self):
        """Test fitting a simple polynomial function"""
        def simple_func(x: torch.Tensor) -> torch.Tensor:
            return 0.5 * x**2 - 0.3 * x + 0.1

        # Generate data
        x_data, y_data = self.generate_test_data(simple_func)

        # Create and optimize network
        kan = FixedKAN(self.config)
        kan.optimize(x_data, y_data)

        # Make predictions
        with torch.no_grad():
            y_pred = kan(x_data)

        # Check MSE
        mse = torch.mean((y_pred - y_data) ** 2)
        print(f"Simple function MSE: {mse.item()}")
        self.assertLess(mse.item(), 0.1)  # Should fit well

        # Analyze network
        analysis = kan.analyze_network(x_data)

        # Verify architecture
        self.assertEqual(len(kan.layers), 2)  # Input->Hidden, Hidden->Output
        self.assertEqual(len(kan.layers[0].neurons), 10)  # 10 hidden neurons

        # Check degrees were assigned
        for layer in kan.layers:
            for neuron in layer.neurons:
                self.assertIsNotNone(neuron.selected_degree)
                self.assertLessEqual(neuron.selected_degree, self.config.max_degree)

    def test_complex_function(self):
        """Test fitting a more complex function"""
        def complex_func(x: torch.Tensor) -> torch.Tensor:
            return torch.sin(2 * np.pi * torch.cos(x**2)) + 0.5 * torch.cos(2 * np.pi * torch.exp(x**2))

        # Generate data
        x_data, y_data = self.generate_test_data(self.target_function, n_samples=1000)

        # Create network with more hidden neurons
        config = FixedKANConfig(
            network_shape=[1, 5, 1],  # More neurons for complex function
            max_degree=5
        )
        kan = FixedKAN(config)

        # Optimize and predict
        kan.optimize(x_data, y_data)
        with torch.no_grad():
            y_pred = kan(x_data)

        # Check MSE
        mse = torch.mean((y_pred - y_data) ** 2)
        print(f"Complex function MSE: {mse.item()}")
        #self.assertLess(mse.item(), 0.1)

        # Analyze and visualize
        analysis = kan.analyze_network(x_data)
        kan.visualize_analysis(analysis, x_data, y_data)



    def test_multi_layer_network(self):
        """Test a deeper network architecture"""
        def complex_func(x: torch.Tensor) -> torch.Tensor:
            return torch.sin(2 * np.pi * torch.cos(x**2)) + 0.5 * torch.cos(2 * np.pi * torch.exp(x**2))

        # Generate data
        x_data, y_data = self.generate_test_data(self.target_function, n_samples=1000)

        # Create deeper network
        config = FixedKANConfig(
            network_shape=[1, 5, 5, 1],  # Three layers
            max_degree=5
        )
        kan = FixedKAN(config)

        # Optimize and predict
        kan.optimize(x_data, y_data)
        with torch.no_grad():
            y_pred = kan(x_data)

        # Check MSE
        mse = torch.mean((y_pred - y_data) ** 2)
        print(f"Complex function MSE: {mse.item()}")
        #self.assertLess(mse.item(), 0.1)
        # Analyze network
        analysis = kan.analyze_network(x_data)
        kan.visualize_analysis(analysis, x_data, y_data)
        # Check layer structure
        self.assertEqual(len(analysis), 3)  # Should have 3 layers
        for layer_idx in range(3):
            layer_data = analysis[f'layer_{layer_idx}']
            # Check neuron outputs
            if layer_idx == 0:
                self.assertEqual(len(layer_data['degrees']), 10)
            elif layer_idx == 1:
                self.assertEqual(len(layer_data['degrees']), 5)
            else:
                self.assertEqual(len(layer_data['degrees']), 1)

    def test_comparison_with_previous(self):
        """Compare results with previous implementation"""
        def test_func(x: torch.Tensor) -> torch.Tensor:
            return torch.sin(2 * np.pi * x) + 0.5 * x**2

        # Generate data
        x_data, y_data = self.generate_test_data(self.target_function)

        # Test new implementation
        kan = FixedKAN(self.config)
        kan.optimize(x_data, y_data)
        with torch.no_grad():
            y_pred_new = kan(x_data)
        mse_new = torch.mean((y_pred_new - y_data) ** 2)

        # Compare with previous implementation
        from TorchDegreeOptimizer import DegreeOptimizer, DegreeOptimizerConfig
        old_config = DegreeOptimizerConfig(
            network_shape=[1, 10, 1],
            max_degree=7
        )
        optimizer = DegreeOptimizer(old_config)
        optimizer.fit(x_data, y_data)
        y_pred_old = optimizer.predict(x_data)
        mse_old = torch.mean((y_pred_old - y_data) ** 2)

        print(f"New implementation MSE: {mse_new.item()}")
        print(f"Old implementation MSE: {mse_old.item()}")

        # Analyze both
        new_analysis = kan.analyze_network(x_data)
        old_analysis = optimizer.analyze_network(x_data, y_data)

        # Plot comparison
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot new implementation
        x_np = x_data.cpu().numpy()
        y_np = y_data.cpu().numpy()
        y_new = y_pred_new.detach().cpu().numpy()
        ax1.scatter(x_np, y_np, alpha=0.5, label='Data')
        ax1.plot(x_np, y_new, 'r-', label='New KAN')
        ax1.set_title('New Implementation')
        ax1.legend()

        # Plot old implementation
        y_old = y_pred_old.detach().cpu().numpy()
        ax2.scatter(x_np, y_np, alpha=0.5, label='Data')
        ax2.plot(x_np, y_old, 'b-', label='Old KAN')
        ax2.set_title('Old Implementation')
        ax2.legend()

        plt.show()
    def test_multivariate_fractal(self):
        """Test fitting a complex multivariate fractal function"""
        @staticmethod
        def fractal_func(x: torch.Tensor) -> torch.Tensor:
            """Fractal-like 2D test function with multiple features"""
            # x shape: [batch_size, 2]
            x_coord = x[:, 0]
            y_coord = x[:, 1]

            # Multiple frequency components
            z = torch.sin(10 * torch.pi * x_coord) * torch.cos(10 * torch.pi * y_coord) + \
                torch.sin(torch.pi * (x_coord**2 + y_coord**2))

            # Non-linear interactions
            z += torch.abs(x_coord - y_coord) + \
                 (torch.sin(5 * x_coord * y_coord) / (0.1 + torch.abs(x_coord + y_coord)))

            # Envelope
            z *= torch.exp(-0.1 * (x_coord**2 + y_coord**2))

            # Add noise
            noise = torch.normal(0, 0.1, z.shape, device=x.device)
            z += noise

            return z.unsqueeze(-1)  # Shape: [batch_size, 1]

        # Generate data in [-1, 1] range for Chebyshev
        n_samples = 50  # 50x50 grid points
        x = torch.linspace(-1, 1, n_samples)
        y = torch.linspace(-1, 1, n_samples)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        x_data = torch.stack([X.flatten(), Y.flatten()], dim=1)  # [2500, 2]
        y_data = fractal_func(x_data)  # [2500, 1]

        # Create network
        config = FixedKANConfig(
            network_shape=[2, 1000, 1],
            max_degree=5
        )
        kan = FixedKAN(config)

        # Optimize and predict
        kan.optimize(x_data, y_data)
        with torch.no_grad():
            y_pred = kan(x_data)

        # Compute error
        mse = torch.mean((y_pred - y_data) ** 2)
        normalize = mse / torch.sum(y_data**2)
        print(f"Fractal function MSE: {normalize.item()}")

        # Analyze network
        analysis = kan.analyze_network(x_data)

        # Plot original vs predicted vs error
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(15, 5))

        # Original function
        ax1 = fig.add_subplot(131, projection='3d')
        Z_true = y_data.reshape(n_samples, n_samples)
        ax1.plot_surface(X.cpu(), Y.cpu(), Z_true.cpu(),
                         cmap='coolwarm', alpha=0.7)
        ax1.set_title('Original Function')

        # Predicted function
        ax2 = fig.add_subplot(132, projection='3d')
        Z_pred = y_pred.reshape(n_samples, n_samples)
        ax2.plot_surface(X.cpu(), Y.cpu(), Z_pred.cpu(),
                         cmap='magma', alpha=0.7)
        ax2.set_title('KAN Prediction')

        # Error plot
        ax3 = fig.add_subplot(133, projection='3d')
        Z_error = torch.abs(Z_true - Z_pred)
        ax3.plot_surface(X.cpu(), Y.cpu(), Z_error.cpu(),
                         cmap='viridis', alpha=0.7)
        ax3.set_title('Absolute Error')

        plt.tight_layout()
        plt.show()

        # Show detailed network analysis
        kan.visualize_analysis(analysis, x_data, y_data)
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)