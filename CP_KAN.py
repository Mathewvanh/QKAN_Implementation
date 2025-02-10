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