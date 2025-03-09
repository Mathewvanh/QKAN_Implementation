import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from torch import Tensor

@dataclass
class FixedKANConfig:
    """
    Configuration for Fixed Architecture KAN
    """
    network_shape: List[int]             # e.g. [input_dim, hidden_dim, ..., output_dim]
    max_degree: int                      # Maximum polynomial degree for QUBO
    complexity_weight: float = 0.1
    trainable_coefficients: bool = False

    # For partial QUBO
    skip_qubo_for_hidden: bool = False
    default_hidden_degree: int = 4       # default polynomial degree for hidden layers (if skipping QUBO)


class KANNeuron(nn.Module):
    """
    Single neuron that uses cumulative polynomials up to selected degree.
    """
    def __init__(self, input_dim: int, max_degree: int):
        super().__init__()
        self.input_dim = input_dim
        self.max_degree = max_degree

        # Store polynomial-degree
        self.register_buffer('selected_degree', torch.tensor([-1], dtype=torch.long))
        self.coefficients = None  # polynomial coefficients (ParameterList or single param)

        # Projection weights/bias (trainable)
        self.w = nn.Parameter(torch.randn(input_dim))
        self.b = nn.Parameter(torch.zeros(1))

    @property
    def degree(self) -> int:
        d = self.selected_degree.item()
        if d < 0:
            raise RuntimeError("Degree not set. Either run QUBO or assign a default.")
        return d

    def set_coefficients(self, coeffs_list: torch.Tensor, train_coeffs: bool = False):
        """
        For degree d, we expect (d+1) coefficients.
        We'll store them as a ParameterList of scalars if we want them trainable.
        """
        if len(coeffs_list) != self.degree + 1:
            raise ValueError(f"Expected {self.degree + 1} coefficients, got {len(coeffs_list)}")

        self.coefficients = nn.ParameterList([
            nn.Parameter(torch.tensor(coeff.item()), requires_grad=train_coeffs)
            for coeff in coeffs_list
        ])

    def _compute_cumulative_transform(self, x: torch.Tensor, degree: int) -> torch.Tensor:
        """
        Project x => scalar alpha = w·x + b, then compute T_0(alpha),...,T_d(alpha).
        Return shape [batch_size, (d+1)].
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, input_dim]
        alpha = x.matmul(self.w) + self.b  # [batch_size]
        transforms = []
        for d_i in range(degree + 1):
            # Chebyshev T_d: torch.special.chebyshev_polynomial_t
            t_d = torch.special.chebyshev_polynomial_t(alpha, n=d_i)
            transforms.append(t_d.unsqueeze(1))  # [batch_size,1]
        return torch.cat(transforms, dim=1)      # [batch_size, (d+1)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.degree < 0:
            raise RuntimeError("Neuron degree not set.")
        if self.coefficients is None:
            raise RuntimeError("Neuron coefficients not set.")

        transform = self._compute_cumulative_transform(x, self.degree)  # [batch_size, d+1]
        coeffs_tensor = torch.stack([c for c in self.coefficients], dim=0)  # [d+1]
        output = (transform * coeffs_tensor).sum(dim=1, keepdim=True)       # [batch_size,1]
        return output


class KANLayer(nn.Module):
    """
    A layer of KAN with output_dim scalar-output neurons + a combine_W + combine_b.
    """
    def __init__(self, input_dim: int, output_dim: int, max_degree: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_degree = max_degree

        # Neurons
        self.neurons = nn.ModuleList([
            KANNeuron(input_dim, max_degree) for _ in range(output_dim)
        ])
        # Combine step => shape [output_dim, output_dim]
        self.combine_W = nn.Parameter(torch.eye(output_dim))
        self.combine_b = nn.Parameter(torch.zeros(output_dim))

    def optimize_degrees(self, x_data: torch.Tensor, y_data: torch.Tensor, train_coeffs: bool) -> None:
        """
        Use QUBO to pick the degree for each neuron, fitting y_data[:, i] for neuron i.
        y_data must have shape [batch_size, output_dim].
        """
        from pyqubo import Array
        import neal

        # Build QUBO
        q = Array.create('q', shape=(self.output_dim, self.max_degree + 1), vartype='BINARY')
        H = 0.0
        degree_coeffs = {}

        # For each neuron => do least squares vs. y_data[:, i]
        for i, neuron in enumerate(self.neurons):
            degree_coeffs[i] = {}
            y_col = y_data[:, i].unsqueeze(-1)  # [batch_size,1]
            for d_i in range(self.max_degree + 1):
                X = neuron._compute_cumulative_transform(x_data, d_i)  # [batch_size, d_i+1]
                coeffs = torch.linalg.lstsq(X, y_col).solution         # [d_i+1, 1]
                y_pred = X.matmul(coeffs)
                mse = torch.mean((y_col - y_pred)**2)
                degree_coeffs[i][d_i] = coeffs
                H += mse * q[i, d_i]


        # 2) One-hot
        #    automatically figure out a scale or just pick a big number
        penalty_strength = 10000000000
        for i in range(self.output_dim):
            H += penalty_strength * (sum(q[i, d_i] for d_i in range(self.max_degree+1)) - 1)**2

        # Solve QUBO
        model = H.compile()
        bqm = model.to_bqm()
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=1000)
        best_sample = min(model.decode_sampleset(sampleset), key=lambda x: x.energy).sample

        # Assign selected degrees & set coefficients
        for i, neuron in enumerate(self.neurons):
            for d_i in range(self.max_degree + 1):
                found_one = False
                if best_sample[f'q[{i}][{d_i}]'] == 1:
                    found_one=True
                    neuron.selected_degree[0] = d_i
                    coeffs_list = degree_coeffs[i][d_i].squeeze(-1)  # shape [d_i+1]
                    neuron.set_coefficients(coeffs_list, train_coeffs)
                    break
                # if not found_one:
                #     print(f' -> For neuron {i}, no degree was set to 1!')
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each neuron => [batch_size,1], then cat => [batch_size, output_dim]
        outs = [nr(x) for nr in self.neurons]
        stack_out = torch.cat(outs, dim=1)  # [batch_size, output_dim]
        return stack_out.mm(self.combine_W) + self.combine_b


# ----------------------------
# PCA-based dimension alignment
# ----------------------------
def autoencoder_dim_align(x_data: torch.Tensor, out_dim: int) -> torch.Tensor:
    """
    If x_data.shape[1] == out_dim, return as-is.
    If x_data.shape[1] >  out_dim, do PCA to reduce dimensions => [batch_size, out_dim].
    If x_data.shape[1] <  out_dim, replicate columns until we have out_dim.

    Returns a new tensor of shape [batch_size, out_dim].
    """
    B, in_dim = x_data.shape
    if in_dim == out_dim:
        return x_data  # no change needed

    elif in_dim > out_dim:
        # PCA dimension reduction => out_dim
        # For large data, might be heavy in memory/time, but it's more principled
        from sklearn.decomposition import PCA
        x_cpu = x_data.detach().cpu().numpy()   # to NumPy
        pca = PCA(n_components=out_dim)
        x_reduced = pca.fit_transform(x_cpu)    # shape [batch_size, out_dim]
        # Convert back to torch
        x_reduced_t = torch.from_numpy(x_reduced).to(
            device=x_data.device, dtype=x_data.dtype
        )
        return x_reduced_t

    else:
        # in_dim < out_dim => replicate columns
        repeats = (out_dim // in_dim) + 1
        expanded = x_data.repeat(1, repeats)  # shape [batch_size, repeats*in_dim]
        expanded = expanded[:, :out_dim]      # slice first out_dim columns
        return expanded


class FixedKAN(nn.Module):
    """
    Multi-layer KAN with partial QUBO logic and optionally trainable coefficients.
    """
    def __init__(self, config: FixedKANConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            KANLayer(config.network_shape[i], config.network_shape[i+1], config.max_degree)
            for i in range(len(config.network_shape) - 1)
        ])

    def optimize(self, x_data: torch.Tensor, y_data: torch.Tensor):
        """
        For each layer, either skip QUBO or do QUBO.
        - Hidden layer => target is 'current_input' if skip_qubo_for_hidden=False
        - Final layer  => target is 'y_data'
        """
        current = x_data
        # 1) Subsample if dataset is huge
        # max_qubo_samples = 1000
        # if x_data.shape[0] > max_qubo_samples:
        #     # pick a random subset of indices
        #     idx = torch.randperm(x_data.shape[0])[:max_qubo_samples]
        #     current = x_data[idx]
        #     y_data = y_data[idx]

        for i, layer in enumerate(self.layers):
            is_last = (i == len(self.layers) - 1)

            if is_last:
                # Final => target = y_data
                layer.optimize_degrees(current, y_data, train_coeffs=self.config.trainable_coefficients)
            else:
                if self.config.skip_qubo_for_hidden:
                    # Just set each neuron to default_hidden_degree
                    for neuron in layer.neurons:
                        neuron.selected_degree[0] = self.config.default_hidden_degree
                        d_plus_1 = neuron.degree + 1
                        neuron.coefficients = nn.ParameterList([
                            nn.Parameter(torch.zeros(()), requires_grad=self.config.trainable_coefficients)
                            for _ in range(d_plus_1)
                        ])
                else:
                    # QUBO for hidden => target = current layer input => shape [batch_size, layer.output_dim]
                    #  => we do PCA if in_dim > out_dim, replicate if in_dim < out_dim
                    #  => if in_dim == out_dim => pass as is
                    aligned_targets = autoencoder_dim_align(current, layer.output_dim)
                    layer.optimize_degrees(
                        current,
                        aligned_targets,  # not the raw "current" => dimension-aligned
                        train_coeffs=self.config.trainable_coefficients
                    )

            # Forward pass => get new 'current' for the next layer
            with torch.no_grad():
                current = layer(current)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def save_model(self, path: str):
        """
        Save degrees plus entire state dict.
        """
        degrees = {}
        for li, layer in enumerate(self.layers):
            degrees[li] = {}
            for ni, neuron in enumerate(layer.neurons):
                degrees[li][ni] = neuron.degree

        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config,
            'degrees': degrees
        }, path)

    @classmethod
    def load_model(cls, path: str) -> 'FixedKAN':
        checkpoint = torch.load(path)
        model = cls(checkpoint['config'])

        # Rebuild degrees & create placeholder coefficients
        for li, layer_degrees in checkpoint['degrees'].items():
            for ni, degree in layer_degrees.items():
                neuron = model.layers[li].neurons[ni]
                neuron.selected_degree[0] = degree
                neuron.coefficients = nn.ParameterList([
                    nn.Parameter(torch.zeros(()), requires_grad=model.config.trainable_coefficients)
                    for _ in range(degree + 1)
                ])

        model.load_state_dict(checkpoint['state_dict'])
        return model

    def train_model(
        self,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        num_epochs: int = 50,
        lr: float = 1e-3,
        complexity_weight: float = 0.1,
        do_qubo: bool = True
    ):
        """
        1) Optionally run QUBO-based optimize() to set degrees & coefficients.
        2) Gather trainable parameters => (w, b) for each neuron, combine_W/b.
           If config.trainable_coefficients=True, also gather coefficient parameters.
        3) Train with MSE + optional L2 penalty on w.
        """
        if do_qubo:
            self.optimize(x_data, y_data)

        params_to_train = []
        for layer in self.layers:
            # The combine weights/bias
            params_to_train.extend([layer.combine_W, layer.combine_b])
            # Each neuron's (w,b)
            for neuron in layer.neurons:
                params_to_train.extend([neuron.w, neuron.b])
                # If we want trainable coefficients:
                if self.config.trainable_coefficients and neuron.coefficients is not None:
                    params_to_train.extend(list(neuron.coefficients))

        optimizer = torch.optim.Adam(params_to_train, lr=lr)

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            y_pred = self.forward(x_data)
            mse = torch.mean((y_pred - y_data)**2)

            # Optional L2 penalty
            w_norm = 0.0
            # for layer in self.layers:
            #     for neuron in layer.neurons:
            #         w_norm += torch.sum(neuron.w**2)
            loss = mse + complexity_weight*w_norm
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss={loss.item():.6f}, MSE={mse.item():.6f}")
        print("[(train_model)] Done training!")

    def train_model_cross_entropy(
            self,
            x_data: torch.Tensor,
            y_data_int: torch.Tensor,       # integer class labels, shape = [batch_size]
            y_data_onehot: torch.Tensor,    # one-hot, shape = [batch_size, num_classes] (for QUBO)
            num_epochs: int = 50,
            lr: float = 1e-3,
            complexity_weight: float = 0.1,
            do_qubo: bool = True
    ):
        """
        1) (Optionally) run QUBO-based optimize() using y_data_onehot (like before).
        2) Gather trainable parameters (including coefficients if config.trainable_coefficients=True).
        3) Compute cross-entropy loss with the final logits vs. integer labels (y_data_int).

        Typical usage example:
          # y_data_onehot => for QUBO
          # y_data_int    => same labels but as class indices
          qkan.train_model_cross_entropy(x_data, y_data_int, y_data_onehot, ...)
        """
        if do_qubo:
            self.optimize(x_data, y_data_onehot)

        params_to_train = []
        for layer in self.layers:
            params_to_train.extend([layer.combine_W, layer.combine_b])
            for neuron in layer.neurons:
                params_to_train.extend([neuron.w, neuron.b])
                if self.config.trainable_coefficients and (neuron.coefficients is not None):
                    params_to_train.extend(list(neuron.coefficients))

        optimizer = torch.optim.Adam(params_to_train, lr=lr)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            logits = self.forward(x_data)  # [batch_size, num_classes]
            ce_loss = nn.functional.cross_entropy(logits, y_data_int.long())

            w_norm = 0.0
            loss = ce_loss + complexity_weight*w_norm
            loss.backward()
            optimizer.step()

            print(f"[CE Training] Epoch {epoch+1}/{num_epochs}, Loss={loss.item():.6f}, CE={ce_loss.item():.6f}")
        print("[train_model_cross_entropy] Done training!")

    def optimize_integer_programming(self, x_data: torch.Tensor, y_data: torch.Tensor):
        """
        Use OR-Tools (Mixed Integer Linear Programming) to optimize the degree selection.
        This is an alternative to the QUBO approach.
        
        Requires: pip install ortools
        """
        print("Starting Integer Programming optimization...")
        try:
            from ortools.linear_solver import pywraplp
        except ImportError:
            raise ImportError("OR-Tools is required for integer programming. Please install: pip install ortools")
        
        # Handle edge case of empty data
        if x_data.shape[0] == 0 or y_data.shape[0] == 0:
            raise ValueError("Cannot optimize with empty data")
            
        current = x_data
        
        for i, layer in enumerate(self.layers):
            is_last = (i == len(self.layers) - 1)
            print(f"Optimizing layer {i+1}/{len(self.layers)}...")
            
            if is_last:
                # Final layer => target = y_data
                target = y_data
            else:
                if self.config.skip_qubo_for_hidden:
                    # Just set each neuron to default_hidden_degree
                    for neuron in layer.neurons:
                        neuron.selected_degree[0] = self.config.default_hidden_degree
                        d_plus_1 = neuron.degree + 1
                        neuron.coefficients = nn.ParameterList([
                            nn.Parameter(torch.zeros(()), requires_grad=self.config.trainable_coefficients)
                            for _ in range(d_plus_1)
                        ])
                    
                    # Forward pass for next layer
                    with torch.no_grad():
                        current = layer(current)
                    continue
                else:
                    # For hidden layer => use dimension-aligned target
                    target = autoencoder_dim_align(current, layer.output_dim)
            
            # Now optimize each neuron in the layer independently
            for j, neuron in enumerate(layer.neurons):
                # Create the MIP solver with the SCIP backend.
                solver = pywraplp.Solver.CreateSolver('SCIP')
                if not solver:
                    raise RuntimeError("Could not create the MIP solver")
                
                # Variables: x[d] = 1 if we select degree d
                x = {}
                for d in range(self.config.max_degree + 1):
                    x[d] = solver.IntVar(0, 1, f'x_{d}')
                
                # Constraint: exactly one degree is selected
                solver.Add(sum(x[d] for d in range(self.config.max_degree + 1)) == 1)
                
                # Compute MSE for each degree
                mse_values = []
                coeffs_dict = {}
                
                y_col = target[:, j].unsqueeze(-1)  # [batch_size,1]
                for d_i in range(self.config.max_degree + 1):
                    X = neuron._compute_cumulative_transform(current, d_i)  # [batch_size, d_i+1]
                    coeffs = torch.linalg.lstsq(X, y_col).solution         # [d_i+1, 1]
                    y_pred = X.matmul(coeffs)
                    mse = torch.mean((y_col - y_pred)**2).item()  # scalar
                    mse_values.append(mse)
                    coeffs_dict[d_i] = coeffs
                
                # Objective: minimize MSE
                objective = solver.Objective()
                for d in range(self.config.max_degree + 1):
                    objective.SetCoefficient(x[d], mse_values[d])
                objective.SetMinimization()
                
                # Solve the problem
                status = solver.Solve()
                
                if status == pywraplp.Solver.OPTIMAL:
                    # Find which degree was selected
                    selected_degree = -1
                    for d in range(self.config.max_degree + 1):
                        if x[d].solution_value() > 0.5:  # if x[d] == 1
                            selected_degree = d
                            break
                    
                    if selected_degree >= 0:
                        # Set the degree and coefficients
                        neuron.selected_degree[0] = selected_degree
                        coeffs_list = coeffs_dict[selected_degree].squeeze(-1)
                        neuron.set_coefficients(coeffs_list, self.config.trainable_coefficients)
                    else:
                        raise RuntimeError(f"No degree selected for neuron {j} in layer {i}")
                else:
                    raise RuntimeError(f"Failed to find optimal solution for neuron {j} in layer {i}")
            
            # Forward pass for next layer
            with torch.no_grad():
                current = layer(current)
                
        print("Integer Programming optimization completed")

    def optimize_evolutionary(self, x_data: torch.Tensor, y_data: torch.Tensor, 
                              population_size: int = 20, generations: int = 50):
        """
        Use a genetic algorithm to optimize the polynomial degrees.
        This is an alternative to the QUBO approach.
        """
        print("Starting Evolutionary optimization...")
        import random
        
        # Handle edge case of empty data
        if x_data.shape[0] == 0 or y_data.shape[0] == 0:
            raise ValueError("Cannot optimize with empty data")
        
        current = x_data
        
        for i, layer in enumerate(self.layers):
            is_last = (i == len(self.layers) - 1)
            print(f"Optimizing layer {i+1}/{len(self.layers)}...")
            
            if is_last:
                # Final layer => target = y_data
                target = y_data
            else:
                if self.config.skip_qubo_for_hidden:
                    # Just set each neuron to default_hidden_degree
                    for neuron in layer.neurons:
                        neuron.selected_degree[0] = self.config.default_hidden_degree
                        d_plus_1 = neuron.degree + 1
                        neuron.coefficients = nn.ParameterList([
                            nn.Parameter(torch.zeros(()), requires_grad=self.config.trainable_coefficients)
                            for _ in range(d_plus_1)
                        ])
                    
                    # Forward pass for next layer
                    with torch.no_grad():
                        current = layer(current)
                    continue
                else:
                    # For hidden layer => use dimension-aligned target
                    target = autoencoder_dim_align(current, layer.output_dim)
            
            # Number of neurons in this layer
            num_neurons = layer.output_dim
            print(f"  Layer has {num_neurons} neurons to optimize")
            
            # For very small networks, make sure population size is appropriate
            actual_population_size = max(5, min(population_size, 2 ** num_neurons))
            if actual_population_size != population_size:
                print(f"  Adjusted population size to {actual_population_size} for small network")
            
            # Initialize population: each individual is a list of degrees for all neurons
            population = []
            for _ in range(actual_population_size):
                individual = [random.randint(0, self.config.max_degree) for _ in range(num_neurons)]
                population.append(individual)
            
            # Pre-compute MSE for each neuron and each degree
            mse_cache = {}
            coeffs_cache = {}
            
            for j, neuron in enumerate(layer.neurons):
                mse_cache[j] = {}
                coeffs_cache[j] = {}
                y_col = target[:, j].unsqueeze(-1)  # [batch_size,1]
                
                for d_i in range(self.config.max_degree + 1):
                    X = neuron._compute_cumulative_transform(current, d_i)  # [batch_size, d_i+1]
                    coeffs = torch.linalg.lstsq(X, y_col).solution         # [d_i+1, 1]
                    y_pred = X.matmul(coeffs)
                    mse = torch.mean((y_col - y_pred)**2).item()  # scalar
                    mse_cache[j][d_i] = mse
                    coeffs_cache[j][d_i] = coeffs
            
            # Fitness function: sum of MSEs across all neurons
            def fitness(individual):
                return sum(mse_cache[j][degree] for j, degree in enumerate(individual))
            
            # Evolution loop
            for gen in range(generations):
                # Evaluate all individuals
                fitness_scores = [fitness(ind) for ind in population]
                
                # Select parents (tournament selection)
                def tournament_select(k=3):
                    # Adjust k for small populations
                    actual_k = min(k, actual_population_size)
                    indices = random.sample(range(actual_population_size), actual_k)
                    best_idx = min(indices, key=lambda i: fitness_scores[i])
                    return population[best_idx]
                
                # Create next generation
                next_population = []
                
                # Elitism: keep the best individual
                elite_idx = fitness_scores.index(min(fitness_scores))
                next_population.append(population[elite_idx])
                
                # Create rest of population through crossover and mutation
                while len(next_population) < actual_population_size:
                    # Crossover (single point)
                    parent1 = tournament_select()
                    parent2 = tournament_select()
                    
                    # Fix for when num_neurons is too small for traditional crossover
                    if num_neurons <= 2:
                        # For very small networks, just do a probabilistic selection from parents
                        child = []
                        for j in range(num_neurons):
                            # 50% chance to pick from either parent
                            if random.random() < 0.5:
                                child.append(parent1[j])
                            else:
                                child.append(parent2[j])
                    else:
                        # Normal crossover for larger networks
                        crossover_point = random.randint(1, num_neurons - 1)
                        child = parent1[:crossover_point] + parent2[crossover_point:]
                    
                    # Mutation (with low probability)
                    for j in range(num_neurons):
                        if random.random() < 0.1:  # 10% mutation rate
                            child[j] = random.randint(0, self.config.max_degree)
                    
                    next_population.append(child)
                
                # Update population
                population = next_population
                
                if gen % 10 == 0:
                    best_fitness = min(fitness_scores)
                    print(f"  Generation {gen}, Best fitness: {best_fitness:.6f}")
            
            # Get the best individual
            best_idx = min(range(actual_population_size), key=lambda i: fitness(population[i]))
            best_individual = population[best_idx]
            
            # Apply the best configuration
            for j, neuron in enumerate(layer.neurons):
                degree = best_individual[j]
                neuron.selected_degree[0] = degree
                coeffs_list = coeffs_cache[j][degree].squeeze(-1)
                neuron.set_coefficients(coeffs_list, self.config.trainable_coefficients)
            
            # Forward pass for next layer
            with torch.no_grad():
                current = layer(current)
                
        print("Evolutionary optimization completed")

    def optimize_greedy_heuristic(self, x_data: torch.Tensor, y_data: torch.Tensor):
        """
        Use a greedy heuristic to optimize the polynomial degrees.
        This approach tries degrees in order of complexity and stops when 
        improvements fall below a threshold.
        """
        print("Starting Greedy Heuristic optimization...")
        
        # Handle edge case of empty data
        if x_data.shape[0] == 0 or y_data.shape[0] == 0:
            raise ValueError("Cannot optimize with empty data")
            
        current = x_data
        
        for i, layer in enumerate(self.layers):
            is_last = (i == len(self.layers) - 1)
            print(f"Optimizing layer {i+1}/{len(self.layers)}...")
            
            if is_last:
                # Final layer => target = y_data
                target = y_data
            else:
                if self.config.skip_qubo_for_hidden:
                    # Just set each neuron to default_hidden_degree
                    for neuron in layer.neurons:
                        neuron.selected_degree[0] = self.config.default_hidden_degree
                        d_plus_1 = neuron.degree + 1
                        neuron.coefficients = nn.ParameterList([
                            nn.Parameter(torch.zeros(()), requires_grad=self.config.trainable_coefficients)
                            for _ in range(d_plus_1)
                        ])
                    
                    # Forward pass for next layer
                    with torch.no_grad():
                        current = layer(current)
                    continue
                else:
                    # For hidden layer => use dimension-aligned target
                    target = autoencoder_dim_align(current, layer.output_dim)
            
            # Optimize each neuron separately
            for j, neuron in enumerate(layer.neurons):
                y_col = target[:, j].unsqueeze(-1)  # [batch_size,1]
                
                best_degree = 0
                best_mse = float('inf')
                best_coeffs = None
                improvement_threshold = 0.01  # Stop if improvement is less than 1%
                
                # For each degree, compute MSE and check if it's better
                for d_i in range(self.config.max_degree + 1):
                    X = neuron._compute_cumulative_transform(current, d_i)  # [batch_size, d_i+1]
                    coeffs = torch.linalg.lstsq(X, y_col).solution         # [d_i+1, 1]
                    y_pred = X.matmul(coeffs)
                    mse = torch.mean((y_col - y_pred)**2).item()  # scalar
                    
                    # Calculate relative improvement
                    if d_i > 0:
                        rel_improvement = (prev_mse - mse) / prev_mse
                    else:
                        rel_improvement = 1.0  # No previous MSE for first degree
                    
                    prev_mse = mse
                    
                    # Update best if this degree is better
                    if mse < best_mse:
                        best_mse = mse
                        best_degree = d_i
                        best_coeffs = coeffs
                    
                    # Stop if improvement is below threshold
                    if d_i > 0 and rel_improvement < improvement_threshold:
                        # Use the current degree if it's the best so far, otherwise use previous
                        if mse < best_mse:
                            best_mse = mse
                            best_degree = d_i
                            best_coeffs = coeffs
                        break
                
                # Set the best degree and coefficients
                neuron.selected_degree[0] = best_degree
                coeffs_list = best_coeffs.squeeze(-1)
                neuron.set_coefficients(coeffs_list, self.config.trainable_coefficients)
            
            # Forward pass for next layer
            with torch.no_grad():
                current = layer(current)
                
        print("Greedy Heuristic optimization completed")