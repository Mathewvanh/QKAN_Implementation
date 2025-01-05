from enum import Enum

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from torch import Tensor

class MetricType(Enum):
    """Enum for currently supported metric types"""
    MSE = 'mse'
    R2 = 'r2'
    # TODO: Add more metrics as needed:
    # MAE = "mae"
    # RMSE = "rmse"
    # Custom metrics can be added here

@dataclass
class DegreeOptimizerConfig:
    """Configuration for Degree Optimizer"""
    network_shape: List[int]
    max_degree: int
    complexity_weight: float = 0.1
    significance_weight: float = 0.05

class DegreeOptimizer(nn.Module):
    def __init__(self, config: DegreeOptimizerConfig):
        super().__init__()
        self.config = config
        self.network_shape = config.network_shape
        self.num_layers = 1
        self.max_degree = config.max_degree
        self.complexity_weight = config.complexity_weight
        self.significance_weight = config.significance_weight

        self.dummy = nn.Parameter(torch.zeros(1))
        #State tracking
        self.transform_cache: Dict[str, torch.Tensor] = {}
        self.degree_scores: Dict[str, torch.Tensor] = {}
        self.optimal_degrees: Optional[List[List[int]]] = None

        self.qsvt_encoder = None

    def _compute_transforms(self, feature_data: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Compute Chebyshev transforms using PyTorch"""
        transforms = {}
        n_samples, n_features = feature_data.shape

        for d in range(self.max_degree + 1):
            transformed_features = []
            for feature_idx in range(n_features):
                feature = feature_data[:, feature_idx]
                transform = torch.special.chebyshev_polynomial_t(feature, n=d)
                transformed_features.append(transform)

            transforms[d] = torch.stack(transformed_features, dim=1)

        return transforms
    def is_degree_definitive(self, scores: torch.Tensor) -> tuple[bool, int]:
        """
        Determine if there's a definitively best degree based on scores.
        Args:
            scores: Tensor of scores for each degree
        Returns:
            Tuple of (is_definitive, best_degree)
        """
        best_degree = int(torch.argmin(scores))
        best_score = float(scores[best_degree])

        is_definitive = True
        for d in range(len(scores)):
            if d != best_degree:
                score = float(scores[d])
                # Changed relative improvement calculation for MSE
                relative_improvement = (score - best_score) / (score + 1e-10)
                if relative_improvement < self.significance_weight:
                    is_definitive = False
                    break

        return is_definitive, best_degree

    def optimize_layer(
            self,
            layer_idx: int,
            x_data: torch.Tensor,
            y_data: torch.Tensor,
            weights: Optional[torch.Tensor] = None,
            num_reads: int = 1000
    ) -> List[List[int]]:
        """
        Optimize degrees for a single layer using quantum annealing.
        Args:
            layer_idx: Which layer to optimize
            x_data: Input data
            y_data: Target data
            weights: Optional sample weights
            num_reads: Number of annealing reads
        Returns:
            List of optimal degrees for this layer's functions
        """
        from pyqubo import Array  # Import here to avoid global pyqubo import
        import neal
        from cpp_pyqubo import Constraint

        input_dim = self.network_shape[layer_idx]
        output_dim = self.network_shape[layer_idx + 1]
        num_functions = input_dim * output_dim

        # Create binary variables for degree selection
        q = Array.create('q', shape=(num_functions, self.max_degree + 1), vartype='BINARY')

        # Evaluate degrees
        scores, comp_r2 = self.evaluate_degree(x_data, y_data, weights)
        is_definitive, definitive_degree = self.is_degree_definitive(scores)

        # Convert scores to CPU numpy for QUBO construction
        scores_np = scores.detach().cpu().numpy()

        # Build QUBO
        H = 0.0
        if is_definitive:
            # Force selection of definitive degree
            for i in range(num_functions):
                H += -100.0 * q[i, definitive_degree]
                for d in range(self.max_degree + 1):
                    if d != definitive_degree:
                        H += 100.0 * q[i, d]
        else:
            # Optimize based on scores and complexity
            for i in range(num_functions):
                for d in range(self.max_degree + 1):
                    improvement = scores_np[d] - scores_np[d-1] if d > 0 else scores_np[d]
                    H += -1.0 * improvement * q[i, d]
                    H += self.complexity_weight * (d**2) * q[i, d]

        # Add one-hot constraint: exactly one degree per function
        for i in range(num_functions):
            constraint = (sum(q[i, d] for d in range(self.max_degree + 1)) - 1)**2
            H += 10.0 * Constraint(constraint, label=f'one_degree_{i}')

        # Compile and solve QUBO
        model = H.compile()
        bqm = model.to_bqm()

        # Use quantum annealing solver
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=num_reads)

        # Decode results
        decoded = model.decode_sampleset(sampleset)
        best_sample = min(decoded, key=lambda x: x.energy)

        # Extract optimal degrees
        optimal_degrees = []
        for out_idx in range(output_dim):
            output_connections = []
            for in_idx in range(input_dim):
                qubo_idx = out_idx * input_dim + in_idx
                for d in range(self.max_degree + 1):
                    if best_sample.sample[f'q[{qubo_idx}][{d}]'] == 1:
                        output_connections.append(d)
                        break
            optimal_degrees.append(output_connections)

        return optimal_degrees

    def fit(self,
            x_data: torch.Tensor,
            y_data: torch.Tensor,
            weights: Optional[torch.Tensor] = None) -> None:
        """
        Fit the optimizer by finding optimal degrees and preparing for training
        """
        self.optimal_degrees = self.optimize_layer(
            layer_idx=0,
            x_data=x_data,
            y_data=y_data,
            weights=weights
        )

        # Calculate feature statistics
        self.feature_means = x_data.mean(dim=0)
        self.feature_stds = x_data.std(dim=0) + 1e-8

        N = self.network_shape[0]
        K = self.network_shape[1]

        # Prepare weight vectors for QSVT (to be implemented)
        weight_vectors = []
        for d in range(self.max_degree + 1):
            weights = torch.zeros(N * K, device=x_data.device)
            for out_idx, connections in enumerate(self.optimal_degrees):
                for in_idx, degree in enumerate(connections):
                    if degree == d:
                        idx = out_idx * N + in_idx
                        weights[idx] = 1.0
            weight_vectors.append(weights)

    def evaluate_degree(
            self,
            x_data: torch.Tensor,
            y_data: torch.Tensor,
            weights: Optional[torch.Tensor] = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Calculate scores for each degree"""
        cache_key = str(x_data.shape)
        if cache_key in self.degree_scores:
            return self.degree_scores[cache_key]

        scores = torch.zeros(self.max_degree + 1, device=x_data.device)
        comp_r2 = torch.zeros(self.max_degree + 1, device=x_data.device)

        for d in range(self.max_degree + 1):
            transforms = []
            for degree in range(d + 1):
                degree_transform = self._compute_transforms(x_data)[degree]
                transforms.append(degree_transform.reshape(len(x_data), -1))

            X = torch.cat(transforms, dim=1) if transforms else torch.zeros(
                (len(y_data), 0), device=x_data.device
            )

            coeffs = torch.linalg.lstsq(X, y_data).solution
            y_pred = X @ coeffs

            metrics = self._compute_metrics(y_data, y_pred, weights)
            scores[d] = metrics['mse']
            comp_r2[d] = metrics['r2']
        return scores, comp_r2

    def predict(self, x_data: torch.Tensor) -> torch.Tensor:
        """
        Make a prediction using optimized degrees.
        :param x_data: Input features tensor [batch_size, input_dim]
        :return: tensor [batch_size, 1]
        """
        if self.optimal_degrees is None:
            raise RuntimeError('Model not fitted yet, call fit first')

        degrees_tensor = torch.tensor(self.optimal_degrees, device=x_data.device, dtype=torch.float32)

        max_degree = degrees_tensor.max().item()
        transforms = []
        for degree in range(int(max_degree) + 1):
            degree_transform = self._compute_transforms(x_data)[degree]
            transforms.append(degree_transform.reshape(len(x_data), -1))

        X = torch.cat(transforms, dim=1) if transforms else torch.zeros(
            (len(x_data), 0), device=x_data.device
        )


        # Initialize predictions
        y_pred = torch.zeros(len(x_data), 1, device=x_data.device)

        # Compute predictions using least squares
        coeffs = torch.linalg.lstsq(X, y_pred).solution
        y_pred = X @ coeffs

        return y_pred


    def _compute_single_metric(
            self,
            metric_type: MetricType,
            y_true: torch.Tensor,
            y_pred: torch.Tensor,
            weights: Optional[torch.Tensor]=None
    )-> float:
        """
        Compute single metric
        :param metric_type: Type of metric
        :param y_true: True values [N, 1]
        :param y_pred: Predicted values [N, 1]
        :param weights: Optional sample weights [N, 1]
        :return: Computed metric value

        TODO: In future versions, this could be extended to:
        1. Accepted custom metric functions
        2. Support metric-specific parameters
        3. Handle different input shapes/types
        """
        if metric_type == MetricType.MSE:
            squared_errors = (y_true - y_pred) ** 2
            if weights is not None:
                return float(torch.sum(squared_errors * weights / torch.sum(weights)))
            return float(torch.mean(squared_errors))
        elif metric_type == MetricType.R2:
            if weights is not None:
                ss_tot = torch.sum(weights * (y_true - y_pred) ** 2)
                ss_res = torch.sum(weights * y_true ** 2)
            else:
                y_mean = torch.mean(y_true)
                ss_tot = torch.sum((y_true - y_mean) ** 2)
                ss_res = torch.sum((y_true - y_mean) ** 2)
            eps = torch.finfo(torch.float32).eps
            if ss_tot < eps:
                return 0.0
            return (1 - ss_tot / ss_res).item()
        raise NotImplementedError(f'Metric {metric_type} is not implemented')

    def _compute_metrics(
            self,
            y_true: torch.Tensor,
            y_pred: torch.Tensor,
            weights: Optional[torch.Tensor]=None
    ) -> Dict[str, float]:
        """
        Compute all required metrics.

        TODO: Future improvements could include:
        1. Configuration of which metrics to compute
        2. Parallel computation of metrics
        3. Custom metric registration
        4. Metric-specific weights/parameters
        """

        #Ensure tensors are properly shaped
        y_true = y_true.reshape(-1, 1).to(y_pred.device)
        y_pred = y_pred.reshape(-1, 1)
        if weights is not None:
            weights = weights.reshape(-1, 1).to(y_pred.device)
        return {
            metric.value: self._compute_single_metric(metric, y_true, y_pred, weights)
            for metric in MetricType
        }


    def analyze_network(self, x_data: torch.Tensor, y_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze network behavior showing how neurons work together.

        Args:
            x_data: Input features tensor [batch_size, input_dim]
            y_data: Target values tensor [batch_size, 1]

        Returns:
            Dictionary containing analysis results:
            - 'neuron_contributions': Contribution of each neuron to final fit
            - 'neuron_degrees': Selected degree for each neuron
            - 'combined_fit': How neurons combine to form final fit
        """
        analysis_results = {}

        n_neurons = len(self.optimal_degrees)
        neuron_contributions = torch.zeros((n_neurons, len(x_data)), device=x_data.device)
        neuron_degrees = [max(degrees) for degrees in self.optimal_degrees]

        # Analyze each neuron's contribution with its selected degree
        for neuron_idx, degrees in enumerate(self.optimal_degrees):
            # Get transforms for this neuron's degrees
            transforms = []
            for degree in range(max(degrees) + 1):
                if degree in degrees:
                    degree_transform = self._compute_transforms(x_data)[degree]
                    transforms.append(degree_transform.reshape(len(x_data), -1))

            if transforms:
                X = torch.cat(transforms, dim=1)
                # Get contribution using least squares
                coeffs = torch.linalg.lstsq(X, y_data).solution
                contribution = X @ coeffs
                neuron_contributions[neuron_idx] = contribution.squeeze()

        analysis_results['neuron_contributions'] = neuron_contributions
        analysis_results['neuron_degrees'] = neuron_degrees

        # Calculate combined fit
        combined_fit = torch.sum(neuron_contributions, dim=0)
        analysis_results['combined_fit'] = combined_fit

        return analysis_results

    def visualize_analysis(self, analysis_results: Dict[str, torch.Tensor], x_data: torch.Tensor, y_data: torch.Tensor) -> None:
        """Visualize how neurons work together to fit function.

        Creates plots showing:
        1. Individual neuron contributions with their degrees
        2. How neurons combine to form final fit
        3. Neuron activation patterns with degrees

        Args:
            analysis_results: Dictionary from analyze_network()
            x_data: Original input data for plotting
            y_data: Original target data for plotting
        """
        import matplotlib.pyplot as plt

        contributions = analysis_results['neuron_contributions'].cpu().numpy()
        neuron_degrees = analysis_results['neuron_degrees']
        combined_fit = analysis_results['combined_fit'].cpu().numpy()

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])

        # Plot original data and combined fit
        x_plot = x_data.cpu().numpy().squeeze()
        y_plot = y_data.cpu().numpy().squeeze()
        ax1.scatter(x_plot, y_plot, alpha=0.5, label='Original Data')
        ax1.plot(x_plot, combined_fit, 'r-', label='Combined Fit')

        # Plot individual neuron contributions
        for i, (contrib, degree) in enumerate(zip(contributions, neuron_degrees)):
            if torch.norm(torch.tensor(contrib)) > 1e-6:  # Only plot active neurons
                ax1.plot(x_plot, contrib, '--', alpha=0.5,
                         label=f'Neuron {i} (deg={degree})')

        ax1.set_title('Function Approximation: Individual and Combined Contributions')
        ax1.legend()
        ax1.grid(True)

        # Plot neuron activation pattern
        activations = torch.norm(torch.tensor(contributions), dim=1).numpy()
        colors = ['C' + str(d % 10) for d in neuron_degrees]
        bars = ax2.bar(range(len(activations)), activations, color=colors)

        # Add degree annotations
        for bar, degree in zip(bars, neuron_degrees):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'd={degree}', ha='center', va='bottom')

        ax2.set_title('Neuron Activation Strengths with Selected Degrees')
        ax2.set_xlabel('Neuron Index')
        ax2.set_ylabel('Activation Strength')

        plt.tight_layout()
        plt.show()


    @torch.jit.export
    def save_state(self, filename:str) -> None:
        """Save optimizer state including QSVT parameters"""
        state = {
            'network_shape': self.network_shape,
            'max_degree': self.max_degree,
            'complexity_weight': self.complexity_weight,
            'significance_weight': self.significance_weight,
            'transform_cache': self.transform_cache,
            'degree_scores': self.degree_scores,
            'optimal_degrees': self.optimal_degrees,
        }
        torch.save(state, filename)
