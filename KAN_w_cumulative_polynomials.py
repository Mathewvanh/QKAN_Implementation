import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pennylane as qml
from fable import fable
from matplotlib import pyplot as plt
from qiskit import transpile
from qiskit_aer import AerSimulator
from scipy.interpolate import griddata


@dataclass
class FixedKANConfig:
    """Configuration for Fixed Architecture KAN"""
    network_shape: List[int]  # [input_dim, ..., output_dim]
    max_degree: int  # Maximum polynomial degree
    complexity_weight: float = 0.1

class KANNeuron(nn.Module):
    """
    Individual neuron that uses cumulative polynomials up to selected degree
    """
    def __init__(self, input_dim: int, max_degree: int):
        super().__init__()
        self.input_dim = input_dim
        self.max_degree = max_degree
        self.selected_degree = None  # Will be set by QUBO
        self.coefficients = None #Store coefficients after optimization
    def _compute_cumulative_transform(self, x: torch.Tensor, degree: int) -> torch.Tensor:
        """Compute transforms using all polynomials up to degree"""
        transforms = []
        batch_size, input_dim = x.shape

        for dim in range(input_dim):
            dim_transforms = []
            x_dim = x[:, dim]

            for d in range(degree + 1):
                transform = torch.special.chebyshev_polynomial_t(x_dim, n=d)
                dim_transforms.append(transform)
            transforms.append(torch.stack(dim_transforms, dim=1))
        X = torch.cat(transforms, dim=1)
        return X

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using cumulative polynomials"""
        if self.selected_degree is None:
            raise RuntimeError("Neuron degree not set. Run optimization first.")
        if self.coefficients is None:
            raise RuntimeError("Neuron coefficients not set. Run optimization first.")

        # Get transform
        transform = self._compute_cumulative_transform(x, self.selected_degree)
        # Apply stored coefficients
        return transform @ self.coefficients





class KANLayer(nn.Module):
    """
    Layer of KAN with fixed number of neurons
    """
    def __init__(self, input_dim: int, output_dim: int, max_degree: int, complexity_weight: float = 0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_degree = max_degree
        self.complexity_weight = complexity_weight
        # Create fixed number of neurons
        self.neurons = nn.ModuleList([
            KANNeuron(input_dim, max_degree)
            for _ in range(output_dim)
        ])
        self.last_quantum_resources = None

    def optimize_degrees(self, x_data: torch.Tensor, y_data: torch.Tensor, use_quantum: bool = False) -> None:
        """Use QUBO to select optimal degrees for each neuron
        :param use_quantum:
        """
        from pyqubo import Array
        import neal

        # Create binary variables for degree selection
        q = Array.create('q', shape=(self.output_dim, self.max_degree + 1),
                         vartype='BINARY')

        # Build QUBO
        H = 0.0
        degree_coeffs = {}
        quantum_resources = [] if use_quantum else None
        # Evaluate each degree option
        for neuron_idx in range(self.output_dim):
            neuron = self.neurons[neuron_idx]
            degree_coeffs[neuron_idx] = {}
            scores = []
            for d in range(self.max_degree + 1):
                # Get transform matrix [batch_size, d+1]
                X = neuron._compute_cumulative_transform(x_data, d)

                if use_quantum:
                    coeffs,resources = self._optimize_coefficients_quantum(X, y_data)
                    quantum_resources.append(resources)
                else:
                    # Solve least squares: X @ β = y
                    coeffs = self._optimize_coefficients_classical(X, y_data)  # [(d+1), 1]

                degree_coeffs[neuron_idx][d] = coeffs

                y_pred = X @ coeffs  # [batch_size, 1]
                mse = torch.mean((y_data - y_pred) ** 2)
                scores.append(mse.item())
                # Add terms to QUBO
            for d in range(self.max_degree + 1):
                # Calculate improvement from previous degree
                improvement = scores[d] - scores[d-1] if d > 0 else scores[d]

                # Add improvement term and complexity penalty
                H += -1.0 * improvement * q[neuron_idx, d]
                H += self.complexity_weight * (d**2) * q[neuron_idx, d]  # complexity_weight = 0.1
        # Add one-hot constraint - exactly one degree per neuron
        for i in range(self.output_dim):
            constraint = (sum(q[i, d] for d in range(self.max_degree + 1)) - 1)**2
            H += 10.0 * constraint

        # Solve QUBO
        model = H.compile()
        bqm = model.to_bqm()
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=1000)

        # Extract results
        best_sample = min(model.decode_sampleset(sampleset),
                              key=lambda x: x.energy).sample

        # Set degrees
        for neuron_idx in range(self.output_dim):
            for d in range(self.max_degree + 1):
                if best_sample[f'q[{neuron_idx}][{d}]'] == 1:
                    self.neurons[neuron_idx].selected_degree = d
                    self.neurons[neuron_idx].coefficients = degree_coeffs[neuron_idx][d]
                    break
        if use_quantum:
            self.last_quantum_resources = quantum_resources

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining all neuron outputs"""
        # Get each neuron's contribution
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron(x))

        # Combine outputs (sum as per KAN theorem)
        return torch.stack(outputs).sum(dim=0)

    @staticmethod
    def _optimize_coefficients_classical(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Classical optimization using Least squares"""
        return torch.linalg.lstsq(X, y).solution
    def _optimize_coefficients_quantum(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor,Dict]:
        """
        Quantum Optimization using QSVT
        :param X: Transform matrix [batch_size, d+1]
        :param y: Target vector [batch_size, 1]
        :return:
                coeffs: Optimized coefficients [(d+1), 1]
                resources: Dictionary containing quantum resource usage
        """
        # Convert to numpy
        X_np = X.detach().numpy()
        y_np = y.detach().numpy()

        # Initialize simulator
        simulator = AerSimulator(method='unitary')

        # Get block encoding of X matrix
        circuit_X, alpha_X = fable(X_np, 0)

        circuit_X = transpile(circuit_X, simulator)

        # Track resources from first encoding
        resources = {
            'n_qubits': circuit_X.num_qubits,
            'circuit_depth': circuit_X.depth(),
            'gate_count': len(circuit_X.data),
            'alpha_scaling': alpha_X
        }

        # Get unitary matrix
        result = simulator.run(circuit_X).result()
        unitary = result.get_unitary(circuit_X)

        # Extract encoded matrix
        N = X_np.shape[0]
        encoded_X = alpha_X * N * unitary[:N, :N]

        # Solve linear system classically for now
        # (We can replace this with quantum linear solver later)
        coeffs = np.linalg.solve(encoded_X, y_np)

        return torch.from_numpy(coeffs).float().reshape(-1, 1), resources
    def get_last_quantum_resources(self) -> Optional[List[Dict]]:
        """Get resources from last quantum optimization, if any"""
        return getattr(self, 'last_quantum_resources', None)
class FixedKAN(nn.Module):
    """
    Complete Fixed Architecture KAN
    """
    def __init__(self, config: FixedKANConfig):
        super().__init__()
        self.config = config

        # Create fixed layers
        self.layers = nn.ModuleList([
            KANLayer(
                config.network_shape[i],
                config.network_shape[i+1],
                config.max_degree,
                config.complexity_weight,
            )
            for i in range(len(config.network_shape)-1)
        ])

    def optimize(self, x_data: torch.Tensor, y_data: torch.Tensor, use_quantum:bool =False) -> None:
        """Optimize degrees for all layers"""
        current_input = x_data
        for i, layer in enumerate(self.layers):
            # For last layer use y_data, otherwise use intermediate target
            if i == len(self.layers) - 1:
                target = y_data
            else:
                # TODO: Add intermediate target computation
                target = y_data

            layer.optimize_degrees(current_input, target, use_quantum)
            # Update input for next layer
            with torch.no_grad():
                current_input = layer(current_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers"""
        current = x
        for layer in self.layers:
            current = layer(current)
        return current

    def analyze_network(self, x_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze network showing neuron contributions for multivariate input.
        Args:
            x_data: Input tensor [batch_size, input_dim]
        Returns:
            Dictionary containing analysis for each layer:
            - neuron_outputs: Outputs of each neuron [num_neurons, batch_size, 1]
            - neuron_input_contributions: Per-dimension contributions
            - degrees: Selected degrees for each neuron
            - combined_output: Sum of neuron outputs [batch_size, 1]
            - input_dim: Number of input dimensions
        """
        analysis = {}
        current_input = x_data
        #input_dim = x_data.shape[1]

        for layer_idx, layer in enumerate(self.layers):
            input_dim = current_input.shape[1]
            # Get each neuron's contribution
            neuron_outputs = []
            neuron_input_contributions = []  # Track contribution per input dimension

            # For each neuron in the layer
            for neuron in layer.neurons:
                # Get transform
                transforms = []
                for dim in range(input_dim):
                    dim_transforms = []
                    x_dim = current_input[:, dim]

                    # Compute polynomials 0 to degree for this dimension
                    for d in range(neuron.selected_degree + 1):
                        transform = torch.special.chebyshev_polynomial_t(x_dim, n=d)
                        dim_transforms.append(transform)

                    transforms.append(torch.stack(dim_transforms, dim=1))

                # Combine transforms and get output
                X = torch.cat(transforms, dim=1)  # [batch_size, (degree+1)*input_dim]
                output = X @ neuron.coefficients  # [batch_size, 1]

                neuron_outputs.append(output)
                neuron_input_contributions.append(transforms)  # Store per-dimension contributions

            neuron_outputs = torch.stack(neuron_outputs)  # [num_neurons, batch_size, 1]

            # Store analysis
            analysis[f'layer_{layer_idx}'] = {
                'neuron_outputs': neuron_outputs,
                'neuron_input_contributions': neuron_input_contributions,
                'degrees': [n.selected_degree for n in layer.neurons],
                'combined_output': neuron_outputs.sum(dim=0),  # [batch_size, 1]
                'input_dim': input_dim
            }

            # Update input for next layer
            current_input = analysis[f'layer_{layer_idx}']['combined_output']

        return analysis

    def visualize_analysis(self, analysis: Dict[str, torch.Tensor], x_data: torch.Tensor,
                           y_data: Optional[torch.Tensor] = None) -> None:
        """
        Visualize network analysis for multivariate input.
        Args:
            analysis: Dictionary from analyze_network()
            x_data: Input tensor [batch_size, input_dim]
            y_data: Optional target tensor [batch_size, 1]
        """

        num_layers = len(self.layers)
        input_dim = x_data.shape[1]

        if input_dim == 2:
            # 2D visualization with surface and contour plots
            fig = plt.figure(figsize=(20, 8*num_layers))
            gs = plt.GridSpec(num_layers, 3)

            for layer_idx in range(num_layers):
                layer_data = analysis[f'layer_{layer_idx}']
                x_plot = x_data.detach().cpu().numpy()

                # Sort points for better surface plotting
                sort_idx = np.lexsort((x_plot[:, 1], x_plot[:, 0]))
                x_plot = x_plot[sort_idx]

                # 3D surface plot
                ax1 = fig.add_subplot(gs[layer_idx, 0], projection='3d')

                # Plot neuron outputs
                neuron_outputs = layer_data['neuron_outputs'].detach().cpu().numpy()
                for i, (output, degree) in enumerate(zip(neuron_outputs, layer_data['degrees'])):
                    output = output.squeeze()[sort_idx]
                    ax1.scatter(x_plot[:, 0], x_plot[:, 1], output,
                                alpha=0.3, label=f'Neuron {i} (deg={degree})')

                # Plot combined output
                combined = layer_data['combined_output'].detach().cpu().numpy().squeeze()[sort_idx]
                ax1.scatter(x_plot[:, 0], x_plot[:, 1], combined,
                            c='red', alpha=0.7, label='Layer Output')

                if layer_idx == num_layers-1 and y_data is not None:
                    y_plot = y_data.detach().cpu().numpy().squeeze()[sort_idx]
                    ax1.scatter(x_plot[:, 0], x_plot[:, 1], y_plot,
                                c='black', alpha=0.5, label='Target')

                ax1.set_title(f'Layer {layer_idx+1} Contributions')
                ax1.set_xlabel('Input 1')
                ax1.set_ylabel('Input 2')
                ax1.set_zlabel('Output')
                ax1.legend()

                # Create regular grid for interpolation
                n_grid = 50
                x1_unique = np.linspace(x_plot[:, 0].min(), x_plot[:, 0].max(), n_grid)
                x2_unique = np.linspace(x_plot[:, 1].min(), x_plot[:, 1].max(), n_grid)
                X1, X2 = np.meshgrid(x1_unique, x2_unique)

                # Contour plot of combined output
                ax2 = fig.add_subplot(gs[layer_idx, 1])
                grid_points = np.column_stack([X1.ravel(), X2.ravel()])
                Z = griddata(x_plot, combined, (X1, X2), method='cubic')
                contour = ax2.contourf(X1, X2, Z, levels=20, cmap='viridis')
                plt.colorbar(contour, ax=ax2)

                ax2.set_title(f'Layer {layer_idx+1} Output Contours')
                ax2.set_xlabel('Input 1')
                ax2.set_ylabel('Input 2')

                # Degree distribution
                ax3 = fig.add_subplot(gs[layer_idx, 2])
                ax3.hist(layer_data['degrees'], bins=range(self.config.max_degree + 2),
                         alpha=0.7, rwidth=0.8)
                ax3.set_title(f'Layer {layer_idx+1} Degree Distribution')
                ax3.set_xlabel('Degree')
                ax3.set_ylabel('Count')

        else:
            # For 1D or higher dimensions, show summary plots
            fig = plt.figure(figsize=(15, 5*num_layers))
            gs = plt.GridSpec(num_layers, 2)

            for layer_idx in range(num_layers):
                layer_data = analysis[f'layer_{layer_idx}']

                # Plot combined output against first two dimensions
                ax1 = fig.add_subplot(gs[layer_idx, 0])
                x_plot = x_data.detach().cpu().numpy()
                combined = layer_data['combined_output'].detach().cpu().numpy().squeeze()

                if input_dim == 1:
                    ax1.scatter(x_plot, combined, alpha=0.5)
                    ax1.set_xlabel('Input')
                else:
                    scatter = ax1.scatter(x_plot[:, 0], x_plot[:, 1], c=combined,
                                          cmap='viridis', alpha=0.5)
                    plt.colorbar(scatter, ax=ax1)
                    ax1.set_xlabel('Input 1')
                    ax1.set_ylabel('Input 2')

                ax1.set_title(f'Layer {layer_idx+1} Output')

                # Degree distribution
                ax2 = fig.add_subplot(gs[layer_idx, 1])
                ax2.hist(layer_data['degrees'], bins=range(self.config.max_degree + 2),
                         alpha=0.7, rwidth=0.8)
                ax2.set_title(f'Layer {layer_idx+1} Degree Distribution')
                ax2.set_xlabel('Degree')
                ax2.set_ylabel('Count')

        plt.tight_layout()
        plt.show()