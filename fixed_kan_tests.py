import unittest
from datetime import datetime

import torch
import numpy as np
from typing import Tuple

from matplotlib import pyplot as plt
from torchvision import datasets, transforms

from KAN_w_cumulative_polynomials import FixedKANConfig, FixedKAN
from mnist_sampling_diagnostics import analyze_mnist_sample, plot_sample_distributions, compare_multiple_samples


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
        kan.optimize(x_data, y_data, use_quantum=True)

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
            network_shape=[1, 10, 5, 1],  # Three layers
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
        from first_conversion_torch.TorchDegreeOptimizer import DegreeOptimizer, DegreeOptimizerConfig
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
            network_shape=[2, 10, 1],
            max_degree=5
        )
        kan = FixedKAN(config)

        # Optimize and predict
        kan.optimize(x_data, y_data, use_quantum=True)
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

    def test_mnist_classification(self):
        """Test fitting a complex MNIST classification function with QUBO degree optimization"""
        import time
        import json
        network_shape = [784,32,16,16,10]
        max_degree = 5
        train_size = 1000
        complexity_weight = 0.01
        weight_epochs = 20
        learning_rate = 0.001
        experiment_config = {
            'date': datetime.now().strftime("%b-%d-%Y-%I-%M-%S"),
            'train_size': train_size,
            'network_shape': network_shape,
            'max_degree': max_degree,
            'complexity_weight': complexity_weight,
            'weight_epochs': weight_epochs,
            'learning_rate': learning_rate,
            'test_size': 10000,  # Full MNIST test set
        }

        start_time = time.time()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

        # Get training sample
        train_indices = torch.randperm(len(train_dataset))[:train_size]
        x_train = train_dataset.data[train_indices].reshape(-1, 784).float() / 255.0
        y_train_labels = train_dataset.targets[train_indices]

        # Analyze the training sample distribution
        print("\n=== Analyzing Training Sample Distribution ===")
        sample_stats = analyze_mnist_sample(x_train, y_train_labels, train_dataset)

        # Store sampling statistics
        experiment_config['sampling_stats'] = {
            'class_distribution': sample_stats['class_percentages'].tolist(),
            'min_samples_per_class': sample_stats['statistics']['min_samples'],
            'max_samples_per_class': sample_stats['statistics']['max_samples'],
            'class_std_dev': sample_stats['statistics']['std_dev']
        }

        # Plot the distribution for this run
        plt.figure(figsize=(10, 5))
        plt.bar(range(10), sample_stats['class_percentages'])
        plt.title('Class Distribution in Training Sample')
        plt.xlabel('Digit Class')
        plt.ylabel('Percentage in Sample')
        plt.grid(True, alpha=0.3)
        plt.show()

        # Convert to one-hot
        y_train = torch.zeros((train_size, 10))
        y_train.scatter_(1, y_train_labels.unsqueeze(1), 1)

        # Use full test set for validation
        x_test = test_dataset.data.reshape(-1, 784).float() / 255.0
        y_test_labels = test_dataset.targets

        config = FixedKANConfig(
            network_shape=network_shape,
            max_degree=max_degree,
            complexity_weight=complexity_weight,
        )
        kan = FixedKAN(config)

        # Train on training data
        structure_start = time.time()
        print("Phase 1: Optimizing network structure with QUBO...")
        kan.optimize(x_train, y_train)
        structure_end = time.time()
        structure_time = structure_end - structure_start

        train_data = torch.utils.data.TensorDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

        # Phase 2: Horizontal Weight Training
        # print("Phase 2: Training horizontal weights...")
        # weight_start = time.time()
        # kan.train_horizontal_weights(
        #     train_loader=train_loader,
        #     epochs=weight_epochs,
        #     learning_rate=learning_rate
        # )
        # weight_end = time.time()
        # weight_time = weight_end - weight_start
    # Test on both train and test sets
        with torch.no_grad():
            # Training set accuracy
            y_pred_train = kan(x_train)
            train_predictions = torch.argmax(y_pred_train, dim=1)
            train_accuracy = (train_predictions == y_train_labels).float().mean()

            # Test set accuracy
            y_pred_test = kan(x_test)
            test_predictions = torch.argmax(y_pred_test, dim=1)
            test_accuracy = (test_predictions == y_test_labels).float().mean()

        total_time = time.time() - start_time

        # Compile results
        results = {
            **experiment_config,
            "metrics": {
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy),
                "structure_time_seconds": structure_time,
                #"weight_time_seconds": weight_time,
                "total_time_seconds": total_time
            }
        }

        # Save results with timestamp
        results_filename = f'mnist_kan_results_acc_{test_accuracy:.4f}_{datetime.now().strftime("%H-%M-%S")}.json'
        with open(results_filename, 'w') as f:
            json.dump(results, f, indent=4)

        print("\nExperiment Results:")
        print(f"Training Size: {experiment_config['train_size']}")
        print(f"Network Shape: {experiment_config['network_shape']}")
        print(f"Structure Optimization Time: {structure_time:.2f} seconds")
        #print(f"Weight Training Time: {weight_time:.2f} seconds")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        kan.verify_coefficients()
        # Save model
        torch.save({
            'model_state_dict': kan.state_dict(),
            'config': experiment_config,
            'results': results
        }, f'./models/mnist_kan_model_{test_accuracy:.4f}.pt')

        return results
    def test_mnist_n_times(self, n:int = 5):
        """Run MNIST test n times and analyze sampling distributions"""

        # Store results from each run
        all_results = []
        all_distributions = []
        test_accuracies = []

        for run in range(n):
            print(f"\n=== Run {run + 1}/{n} ===")
            results = self.test_mnist_classification()

            # Store results
            all_results.append(results)
            all_distributions.append(results['sampling_stats']['class_distribution'])
            test_accuracies.append(results['metrics']['test_accuracy'])

        # Analyze distributions across all runs
        print("\n=== Analysis Across All Runs ===")
        distributions = np.array(all_distributions)
        accuracies = np.array(test_accuracies)

        # Plot all distributions together
        plt.figure(figsize=(15, 8))

        # Plot individual runs
        for i, (dist, acc) in enumerate(zip(distributions, accuracies)):
            plt.plot(range(10), dist, 'o-', alpha=0.6,
                     label=f'Run {i+1} (Acc: {acc:.3f})')

        # Plot ideal distribution (10% each)
        plt.axhline(y=10, color='k', linestyle='--', alpha=0.5, label='Ideal (10%)')

        plt.title('Class Distributions Across Runs')
        plt.xlabel('Digit Class')
        plt.ylabel('Percentage in Sample')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # Analysis statistics
        average_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        print(f"\nAccuracy Statistics:")
        print(f"Average Test Accuracy: {average_accuracy:.4f}")
        print(f"Std Dev Accuracy: {std_accuracy:.4f}")

        # Find best and worst runs
        best_run = np.argmax(accuracies)
        worst_run = np.argmin(accuracies)

        print(f"\nBest Run ({accuracies[best_run]:.4f}):")
        print("Class distribution:", distributions[best_run])

        print(f"\nWorst Run ({accuracies[worst_run]:.4f}):")
        print("Class distribution:", distributions[worst_run])

        # Calculate class representation variation
        class_stds = np.std(distributions, axis=0)
        print("\nClass Variation (std dev across runs):")
        for digit in range(10):
            print(f"Digit {digit}: {class_stds[digit]:.2f}%")

        return all_results
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
