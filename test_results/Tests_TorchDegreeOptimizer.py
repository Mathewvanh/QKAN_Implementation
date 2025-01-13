import unittest
import torch
from typing import Tuple

from TorchDegreeOptimizer import DegreeOptimizerConfig, DegreeOptimizer


class TestTorchDegreeOptimizer(unittest.TestCase):
    def setUp(self):
        """Initialize common test components"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = DegreeOptimizerConfig(
            network_shape=[1, 10, 1],  # Simple 1D function
            max_degree=10,
            complexity_weight=0.1,
            significance_weight=0.05
        )
        self.optimizer = DegreeOptimizer(self.config)

    def generate_polynomial_data(self, degree: int, n_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate polynomial data for testing"""
        x = torch.linspace(-1, 1, n_samples, device=self.device).reshape(-1, 1)
        coeffs = torch.randn(degree + 1, device=self.device)
        y = sum(coeff * torch.pow(x, i) for i, coeff in enumerate(coeffs))
        # Add some noise
        y += 0.1 * torch.randn_like(y)
        return x, y

    def generate_sin_data(self, n_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sinusoidal data for testing"""
        x = torch.linspace(-torch.pi, torch.pi, n_samples, device=self.device).reshape(-1, 1)
        y = torch.sin(x) + 0.1 * torch.randn_like(x)
        return x, y

    def test_polynomial_fitting(self):
        """Test optimizer on polynomial data"""
        for test_degree in [2, 3, 4]:
            with self.subTest(f"Testing degree {test_degree} polynomial"):
                # Generate test data
                x_data, y_data = self.generate_polynomial_data(test_degree)

                # Optimize degrees
                self.optimizer.fit(x_data, y_data)

                # Check optimal degrees exist
                self.assertIsNotNone(self.optimizer.optimal_degrees)

                # Get MSE scores for each degree
                scores = self.optimizer.evaluate_degree(x_data, y_data)[0]  # Just take MSE scores

                # Verify:
                # 1. MSE decreases with degree (model improves)
                self.assertTrue(torch.all(scores[1:] <= scores[:-1]))

                # 2. Final MSE is good (model fits well)
                final_mse = scores[-1]
                self.assertLess(final_mse, 0.1)  # Reasonable threshold for noisy data

                # 3. Selected degrees allow good fit
                y_pred = self.optimizer.predict(x_data)
                mse = self.optimizer._compute_metrics(y_data, y_pred)['mse']
                self.assertLess(mse, 0.1)

    def test_sin_fitting(self):
        """Test optimizer on sinusoidal data"""
        x_data, y_data = self.generate_sin_data()

        # Optimize degrees
        self.optimizer.fit(x_data, y_data)

        # Check metrics
        scores, comp_r2 = self.optimizer.evaluate_degree(x_data, y_data)

        # Should use higher degrees to approximate sin
        max_selected_degree = max(max(d) for d in self.optimizer.optimal_degrees)
        self.assertGreater(max_selected_degree, 3)

    def test_weighted_optimization(self):
        """Test optimizer with weighted samples"""
        x_data, y_data = self.generate_polynomial_data(degree=2)

        # Create weights emphasizing certain regions
        weights = torch.exp(-5 * torch.abs(x_data))

        # Optimize with weights
        self.optimizer.fit(x_data, y_data, weights)
        weighted_degrees = self.optimizer.optimal_degrees

        # Optimize without weights
        self.optimizer.fit(x_data, y_data)
        unweighted_degrees = self.optimizer.optimal_degrees

        # Degrees should be different with weights
        self.assertNotEqual(weighted_degrees, unweighted_degrees)

    def test_device_handling(self):
        """Test optimizer works with different devices"""
        x_data, y_data = self.generate_polynomial_data(degree=2)

        # Move to CPU
        x_cpu = x_data.cpu()
        y_cpu = y_data.cpu()
        self.optimizer.fit(x_cpu, y_cpu)
        cpu_degrees = self.optimizer.optimal_degrees

        if torch.cuda.is_available():
            # Move to GPU
            x_gpu = x_data.cuda()
            y_gpu = y_data.cuda()
            self.optimizer.fit(x_gpu, y_gpu)
            gpu_degrees = self.optimizer.optimal_degrees

            # Results should be same regardless of device
            self.assertEqual(cpu_degrees, gpu_degrees)