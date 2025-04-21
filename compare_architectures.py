import argparse
import logging
import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# Import our existing modules
from data_pipeline_js_config import DataConfig
from data_pipeline import DataPipeline
from optimization_tuner import OptimizationTuner, count_parameters
from CP_KAN import FixedKAN, FixedKANConfig

# Import the EasyTSF KAN architectures
sys.path.append('EasyTSF')
from easytsf.layer.kanlayer import (
    KANLayer, WaveKANLayer, NaiveFourierKANLayer, 
    JacobiKANLayer, ChebyKANLayer, TaylorKANLayer, 
    RBFKANLayer, MoKLayer
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class KANArchitectureTuner:
    """Class to compare different KAN architectures on the Jane Street dataset."""
    
    def __init__(self, results_dir="architecture_comparison_results"):
        """Initialize the tuner with the dataset and setup directories."""
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Create plots directory
        self.plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Create subdirectories for different plot types
        os.makedirs(os.path.join(self.plots_dir, "performance"), exist_ok=True)
        os.makedirs(os.path.join(self.plots_dir, "hyperparams"), exist_ok=True)
        os.makedirs(os.path.join(self.plots_dir, "comparison"), exist_ok=True)
        
        # Setup logging for this class
        self.logger = logging.getLogger(self.__class__.__name__)
        file_handler = logging.FileHandler(os.path.join(results_dir, f"architecture_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        # Results DataFrame
        self.results_df = pd.DataFrame()
        
        # Load Jane Street dataset
        self._load_data()
    
    def _load_data(self):
        """Load Jane Street market data."""
        # Configure Jane Street dataset
        self.data_cfg = DataConfig(
            data_path="~/Interning/Kaggle/jane_street_kaggle/jane-street-real-time-market-data-forecasting/train.parquet/",
            n_rows=200000,  # Use 200k rows
            train_ratio=0.7,
            feature_cols=[f'feature_{i:02d}' for i in range(79)],
            target_col="responder_6",
            weight_col="weight",
            date_col="date_id"
        )

        # Load and preprocess data using the Jane Street pipeline
        pipeline = DataPipeline(self.data_cfg, self.logger)
        train_df, train_target, train_weight, val_df, val_target, val_weight = pipeline.load_and_preprocess_data()

        # Convert to numpy then torch
        self.x_train = torch.tensor(train_df.to_numpy(), dtype=torch.float32)
        self.y_train = torch.tensor(train_target.to_numpy(), dtype=torch.float32).squeeze(-1).unsqueeze(-1)
        self.w_train = torch.tensor(train_weight.to_numpy(), dtype=torch.float32).squeeze(-1)

        self.x_val = torch.tensor(val_df.to_numpy(), dtype=torch.float32)
        self.y_val = torch.tensor(val_target.to_numpy(), dtype=torch.float32).squeeze(-1).unsqueeze(-1)
        self.w_val = torch.tensor(val_weight.to_numpy(), dtype=torch.float32).squeeze(-1)

        self.input_dim = self.x_train.shape[1]
        self.logger.info(f"Loaded Jane Street dataset with {len(self.x_train)} training samples and {len(self.x_val)} validation samples")
        self.logger.info(f"Input dimension: {self.input_dim}")
    
    def weighted_r2(self, y_true, y_pred, weights):
        """Calculate weighted R² score using the Jane Street competition formula."""
        # Convert to numpy if tensors
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
            
        # Flatten if needed
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        weights = weights.ravel()
        
        # Calculate weighted means
        weighted_mean_true = np.sum(weights * y_true) / np.sum(weights)
        
        # Calculate weighted total sum of squares
        weighted_total_ss = np.sum(weights * (y_true - weighted_mean_true) ** 2)
        
        # Calculate weighted residual sum of squares
        weighted_residual_ss = np.sum(weights * (y_true - y_pred) ** 2)
        
        # Calculate weighted R²
        weighted_r2 = 1 - (weighted_residual_ss / weighted_total_ss)
        
        return weighted_r2
        
    def weighted_mse(self, y_true, y_pred, weights):
        """Calculate weighted MSE score."""
        # Convert to numpy if tensors
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
            
        # Flatten if needed
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        weights = weights.ravel()
        
        # Calculate weighted MSE
        weighted_mse = np.sum(weights * (y_true - y_pred) ** 2) / np.sum(weights)
        
        return weighted_mse
    
    def _get_configs_for_architecture(self, arch_name):
        """
        Get all configurations to test for a specific architecture.
        Optimized for laptop execution with fewer configurations.
        
        Args:
            arch_name: Name of the architecture
            
        Returns:
            List of configuration dictionaries
        """
        # Define optimized hyperparameters for laptop execution
        hidden_sizes = [24, 28, 32]  # Three hidden sizes
        learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]  # Five learning rates
        
        # Create configurations based on architecture
        if arch_name == "CP-KAN" or arch_name == "OriginalKAN":
            return [
                {'hidden_size': hs, 'max_degree': md, 'hidden_degree': 5, 'learning_rate': lr}
                for hs in hidden_sizes for md in [3, 5, 7] for lr in learning_rates
            ]
        elif arch_name == "SplineKAN":
            return [
                {'hidden_size': hs, 'k': k, 'learning_rate': lr}
                for hs in hidden_sizes for k in [3, 4, 5] for lr in learning_rates
            ]
        elif arch_name == "WaveletKAN":
            return [
                {'hidden_size': hs, 'wavelet_type': wt, 'learning_rate': lr}
                for hs in hidden_sizes for wt in ['mexican_hat', 'morlet', 'ricker'] for lr in learning_rates
            ]
        elif arch_name == "FourierKAN":
            return [
                {'hidden_size': hs, 'gridsize': gs, 'learning_rate': lr}
                for hs in hidden_sizes for gs in [150, 200, 250] for lr in learning_rates
            ]
        elif arch_name == "JacobiKAN":
            return [
                {'hidden_size': hs, 'degree': d, 'a': 1.0, 'b': 1.0, 'learning_rate': lr}
                for hs in hidden_sizes for d in [3, 4, 5] for lr in learning_rates
            ]
        elif arch_name == "ChebyshevKAN":
            return [
                {'hidden_size': hs, 'degree': d, 'learning_rate': lr}
                for hs in hidden_sizes for d in [3, 4, 5] for lr in learning_rates
            ]
        elif arch_name == "TaylorKAN":
            return [
                {'hidden_size': hs, 'order': o, 'learning_rate': lr}
                for hs in hidden_sizes for o in [3, 4, 5] for lr in learning_rates
            ]
        elif arch_name == "RBFKAN":
            return [
                {'hidden_size': hs, 'num_centers': nc, 'alpha': 1.0, 'learning_rate': lr}
                for hs in hidden_sizes for nc in [20, 30, 40] for lr in learning_rates
            ]
        elif arch_name == "MixtureKAN":
            return [
                {'hidden_size': hs, 'experts_type': et, 'learning_rate': lr}
                for hs in hidden_sizes for et in ["A", "B", "C"] for lr in learning_rates
            ]
        elif arch_name == "Transformer":
            return [
                {'hidden_size': hs, 'num_layers': nl, 'num_heads': nh, 'dropout': 0.1, 'learning_rate': lr}
                for hs in hidden_sizes for nl in [1, 2, 3] for nh in [4, 8] for lr in learning_rates[:3]  # Less combinations for Transformer
            ]
        elif arch_name == "LSTM":
            return [
                {'hidden_size': hs, 'num_layers': nl, 'dropout': 0.1, 'bidirectional': bd, 'learning_rate': lr}
                for hs in hidden_sizes for nl in [1, 2] for bd in [True, False] for lr in learning_rates[:3]  # Less combinations for LSTM
            ]
        else:
            self.logger.warning(f"Unknown architecture: {arch_name}, using default configs")
            return [{'hidden_size': hs, 'learning_rate': lr} for hs in hidden_sizes for lr in learning_rates]
    
    def _train_architecture(self, model, architecture_name, config, num_epochs=50, lr=0.01):
        """Train a model architecture and return validation metrics."""
        start_time = time.time()
        results = {}
        
        # Move model to device
        device = next(model.parameters()).device
        
        # Special handling for CP-KAN (display name)
        display_arch_name = "CP-KAN" if architecture_name == "CP-KAN" else architecture_name
        
        # Set up optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Move data to the appropriate device
        x_train_device = self.x_train.to(device)
        y_train_device = self.y_train.to(device)
        w_train_device = self.w_train.to(device)
        
        x_val_device = self.x_val.to(device)
        y_val_device = self.y_val.to(device)
        w_val_device = self.w_val.to(device)
        
        # Training loop tracking
        best_val_r2 = float('-inf')
        best_val_mse = float('inf')
        val_r2_scores = []
        val_mse_scores = []
        epochs_no_improve = 0
        patience = 7  # Slightly increased early stopping patience
        
        # For storing the best model
        best_model_state = None
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(x_train_device)
            
            # Compute weighted MSE loss
            criterion = nn.MSELoss(reduction='none')
            batch_losses = criterion(outputs, y_train_device)
            
            # Apply sample weights
            if outputs.shape != w_train_device.shape:
                # Handle broadcasting for multi-output models
                weighted_losses = batch_losses * w_train_device.view(-1, 1)
            else:
                weighted_losses = batch_losses * w_train_device
                
            loss = weighted_losses.mean()
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val_device)
                
                # Calculate metrics (keeping them on device)
                val_r2 = self.weighted_r2(y_val_device, val_pred, w_val_device)
                val_mse = self.weighted_mse(y_val_device, val_pred, w_val_device)
                
                val_r2_scores.append(val_r2.item())
                val_mse_scores.append(val_mse.item())
                
                # Log progress
                if epoch % 10 == 0 or epoch == num_epochs - 1:  # Log less frequently to reduce console output
                    self.logger.info(f"[{display_arch_name}] Config={config} Epoch {epoch+1}/{num_epochs}, "
                                   f"Train Loss={loss.item():.4f}, Val R²={val_r2.item():.4f}, Val MSE={val_mse.item():.6f}")
                
                # Save best model
                if val_r2.item() > best_val_r2:
                    best_val_r2 = val_r2.item()
                    best_val_mse = val_mse.item()
                    best_model_state = {k: v.cpu().detach() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            
            # Early stopping
            if epochs_no_improve >= patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final metrics
        training_time = time.time() - start_time
        
        # Return results dictionary
        results = {
            'best_val_r2': best_val_r2,
            'best_val_mse': best_val_mse,
            'training_time': training_time,
            'epochs_trained': epoch + 1,
            'val_r2_history': val_r2_scores,
            'val_mse_history': val_mse_scores
        }
        
        return results
    
    def _create_original_kan(self, hidden_size, max_degree, hidden_degree):
        """Create the original KAN model from our implementation."""
        kan_config = FixedKANConfig(
            network_shape=[self.input_dim, hidden_size, 1],
            max_degree=max_degree,
            complexity_weight=0.0,
            trainable_coefficients=True,
            skip_qubo_for_hidden=False,
            default_hidden_degree=hidden_degree
        )
        
        model = FixedKAN(kan_config)
        return model
    
    def _create_spline_kan(self, hidden_size, k=3):
        """Create a spline-based KAN from EasyTSF."""
        class SplineKANModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, k):
                super().__init__()
                self.layer1 = KANLayer(input_dim, hidden_dim, num=5, k=k, device='cpu')
                self.layer2 = nn.Linear(hidden_dim, output_dim)
                
            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                return x
        
        model = SplineKANModel(self.input_dim, hidden_size, 1, k)
        return model
    
    def _create_wavelet_kan(self, hidden_size, wavelet_type='mexican_hat'):
        """Create a wavelet-based KAN from EasyTSF."""
        class WaveletKANModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, wavelet_type):
                super().__init__()
                self.layer1 = WaveKANLayer(input_dim, hidden_dim, wavelet_type=wavelet_type, with_bn=True, device='cpu')
                self.layer2 = nn.Linear(hidden_dim, output_dim)
                
            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                return x
        
        model = WaveletKANModel(self.input_dim, hidden_size, 1, wavelet_type)
        return model
    
    def _create_fourier_kan(self, hidden_size, gridsize=300):
        """Create a Fourier-based KAN from EasyTSF."""
        class FourierKANModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, gridsize):
                super().__init__()
                self.layer1 = NaiveFourierKANLayer(input_dim, hidden_dim, gridsize=gridsize)
                self.layer2 = nn.Linear(hidden_dim, output_dim)
                
            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                return x
        
        model = FourierKANModel(self.input_dim, hidden_size, 1, gridsize)
        return model
    
    def _create_jacobi_kan(self, hidden_size, degree=5, a=1.0, b=1.0):
        """Create a Jacobi polynomial-based KAN from EasyTSF."""
        class JacobiKANModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, degree, a, b):
                super().__init__()
                self.layer1 = JacobiKANLayer(input_dim, hidden_dim, degree, a=a, b=b)
                self.layer2 = nn.Linear(hidden_dim, output_dim)
                
            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                return x
        
        model = JacobiKANModel(self.input_dim, hidden_size, 1, degree, a, b)
        return model
    
    def _create_chebyshev_kan(self, hidden_size, degree=5):
        """Create a Chebyshev polynomial-based KAN from EasyTSF."""
        class ChebyshevKANModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, degree):
                super().__init__()
                self.layer1 = ChebyKANLayer(input_dim, hidden_dim, degree)
                self.layer2 = nn.Linear(hidden_dim, output_dim)
                
            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                return x
        
        model = ChebyshevKANModel(self.input_dim, hidden_size, 1, degree)
        return model
    
    def _create_taylor_kan(self, hidden_size, order=5):
        """Create a Taylor series-based KAN from EasyTSF."""
        class TaylorKANModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, order):
                super().__init__()
                self.layer1 = TaylorKANLayer(input_dim, hidden_dim, order)
                self.layer2 = nn.Linear(hidden_dim, output_dim)
                
            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                return x
        
        model = TaylorKANModel(self.input_dim, hidden_size, 1, order)
        return model
    
    def _create_rbf_kan(self, hidden_size, num_centers=50, alpha=1.0):
        """Create an RBF-based KAN from EasyTSF."""
        class RBFKANModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_centers, alpha):
                super().__init__()
                self.layer1 = RBFKANLayer(input_dim, hidden_dim, num_centers, alpha)
                self.layer2 = nn.Linear(hidden_dim, output_dim)
                
            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                return x
        
        model = RBFKANModel(self.input_dim, hidden_size, 1, num_centers, alpha)
        return model
    
    def _create_mixture_kan(self, hidden_size, experts_type="A"):
        """Create a Mixture of KANs model from EasyTSF."""
        class MixtureKANModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, experts_type):
                super().__init__()
                self.layer1 = MoKLayer(input_dim, hidden_dim, experts_type=experts_type, gate_type="Linear")
                self.layer2 = nn.Linear(hidden_dim, output_dim)
                
            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                return x
        
        model = MixtureKANModel(self.input_dim, hidden_size, 1, experts_type)
        return model
    
    def _create_transformer_model(self, hidden_size, num_layers=2, num_heads=4, dropout=0.1):
        """Create a Transformer-based model for comparison."""
        class TransformerModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                
                # Input embedding
                self.input_embedding = nn.Linear(input_dim, hidden_dim)
                
                # Positional encoding is not needed since we're treating features
                # as a set rather than a sequence
                
                # Transformer encoder layers
                encoder_layers = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim*4,
                    dropout=dropout,
                    batch_first=True
                )
                self.transformer_encoder = nn.TransformerEncoder(
                    encoder_layers,
                    num_layers=num_layers
                )
                
                # Output layer
                self.output_layer = nn.Linear(hidden_dim, output_dim)
                
            def forward(self, x):
                # Reshape input for transformer: [batch_size, seq_len(=1), features]
                x = x.unsqueeze(1)
                
                # Input embedding
                x = self.input_embedding(x)
                
                # Transformer encoder
                x = self.transformer_encoder(x)
                
                # Take the output of the last position
                x = x.squeeze(1)
                
                # Output layer
                x = self.output_layer(x)
                
                return x
        
        model = TransformerModel(
            input_dim=self.input_dim, 
            hidden_dim=hidden_size, 
            output_dim=1, 
            num_layers=num_layers, 
            num_heads=num_heads, 
            dropout=dropout
        )
        
        return model
    
    def _create_lstm_model(self, hidden_size, num_layers=2, dropout=0.1, bidirectional=True):
        """Create an LSTM-based model for comparison."""
        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, bidirectional):
                super().__init__()
                
                # We'll reshape the features as a sequence
                self.input_dim = 1  # Each feature is processed as a single value
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                self.bidirectional = bidirectional
                
                # LSTM layer
                self.lstm = nn.LSTM(
                    input_size=self.input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=bidirectional
                )
                
                # Output layer
                self.output_layer = nn.Linear(
                    hidden_dim * 2 if bidirectional else hidden_dim, 
                    output_dim
                )
                
            def forward(self, x):
                # Reshape input for LSTM: [batch_size, seq_len, features]
                # For financial data, we treat each feature as a timestep
                batch_size, num_features = x.shape
                x = x.view(batch_size, num_features, 1)
                
                # LSTM forward pass
                lstm_out, _ = self.lstm(x)
                
                # Take the output of the last timestep
                if self.bidirectional:
                    # For bidirectional, we concatenate outputs from both directions
                    lstm_out = lstm_out[:, -1, :]
                else:
                    lstm_out = lstm_out[:, -1, :]
                
                # Output layer
                output = self.output_layer(lstm_out)
                
                return output
        
        model = LSTMModel(
            input_dim=self.input_dim, 
            hidden_dim=hidden_size, 
            output_dim=1, 
            num_layers=num_layers, 
            dropout=dropout, 
            bidirectional=bidirectional
        )
        
        return model
    
    def _optimize_and_train(self, model, architecture_name, config, optimizer_method="Evolutionary", num_epochs=20):
        """Optimize the model architecture and then train it."""
        # Apply the optimization method if it's the OriginalKAN
        start_time = time.time()
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device explicitly
        model = model.to(device)
        
        # Move data to device for optimization
        x_train = self.x_train.to(device)
        y_train = self.y_train.to(device)
        
        if architecture_name == "OriginalKAN" or architecture_name == "CP-KAN":
            self.logger.info(f"Running {optimizer_method} optimization on {architecture_name}...")
            # Optimize using the selected method
            if optimizer_method == "QUBO":
                model.optimize(x_train, y_train)
            elif optimizer_method == "IntegerProgramming":
                model.optimize_integer_programming(x_train, y_train)
            elif optimizer_method == "Evolutionary":
                model.optimize_evolutionary(x_train, y_train)
            elif optimizer_method == "GreedyHeuristic":
                model.optimize_greedy_heuristic(x_train, y_train)
            else:
                raise ValueError(f"Unknown optimization method: {optimizer_method}")
            self.logger.info(f"Optimization completed")
        
        optimization_time = time.time() - start_time
        
        # Train the model
        learning_rate = config.get('learning_rate', 0.01)
        results = self._train_architecture(model, architecture_name, config, num_epochs, learning_rate)
        
        # Count parameters
        param_count = count_parameters(model)
        
        # Add to results
        config_with_results = {
            'architecture': architecture_name,
            'config': config,
            'opt_time': optimization_time,
            'param_count': param_count,
            'best_val_r2': results['best_val_r2'],
            'best_val_mse': results['best_val_mse'],
            'train_losses': results.get('val_r2_history', []),  # Use val_r2_history if train_losses not available
            'val_r2_scores': results.get('val_r2_history', results.get('val_r2_scores', [])),
            'val_mse_scores': results.get('val_mse_history', results.get('val_mse_scores', []))
        }
        
        return config_with_results
    
    def compare_architectures(self, num_epochs=50, optimization_method="Evolutionary", architectures_to_test=None):
        """Compare different KAN architectures."""
        self.logger.info(f"Starting architecture comparison with {optimization_method} optimization")
        
        # Define configurations to test - optimized for laptop
        hidden_sizes = [24, 28, 32]  # Three hidden sizes
        learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]  # Five learning rates
        
        # Store results
        all_results = []
        
        # Architecture tests to run
        architectures = [
            # CP-KAN (from our implementation)
            {
                'name': 'OriginalKAN',
                'configs': [
                    {'hidden_size': hs, 'max_degree': md, 'hidden_degree': 3, 'learning_rate': lr}
                    for hs in hidden_sizes for md in [3, 4, 5] for lr in learning_rates
                ],
                'create_fn': self._create_original_kan
            },
            # Spline KAN
            {
                'name': 'SplineKAN',
                'configs': [
                    {'hidden_size': hs, 'k': k, 'learning_rate': lr}
                    for hs in hidden_sizes for k in [3, 4, 5] for lr in learning_rates
                ],
                'create_fn': self._create_spline_kan
            },
            # Wavelet KAN
            {
                'name': 'WaveletKAN',
                'configs': [
                    {'hidden_size': hs, 'wavelet_type': wt, 'learning_rate': lr}
                    for hs in hidden_sizes for wt in ['mexican_hat', 'morlet', 'ricker'] for lr in learning_rates
                ],
                'create_fn': self._create_wavelet_kan
            },
            # Fourier KAN
            {
                'name': 'FourierKAN',
                'configs': [
                    {'hidden_size': hs, 'gridsize': gs, 'learning_rate': lr}
                    for hs in hidden_sizes for gs in [150, 200, 250] for lr in learning_rates
                ],
                'create_fn': self._create_fourier_kan
            },
            # Jacobi KAN
            {
                'name': 'JacobiKAN',
                'configs': [
                    {'hidden_size': hs, 'degree': d, 'a': 1.0, 'b': 1.0, 'learning_rate': lr}
                    for hs in hidden_sizes for d in [3, 4, 5] for lr in learning_rates
                ],
                'create_fn': self._create_jacobi_kan
            },
            # Chebyshev KAN
            {
                'name': 'ChebyshevKAN',
                'configs': [
                    {'hidden_size': hs, 'degree': d, 'learning_rate': lr}
                    for hs in hidden_sizes for d in [3, 4, 5] for lr in learning_rates
                ],
                'create_fn': self._create_chebyshev_kan
            },
            # Taylor KAN
            {
                'name': 'TaylorKAN',
                'configs': [
                    {'hidden_size': hs, 'order': o, 'learning_rate': lr}
                    for hs in hidden_sizes for o in [3, 4, 5] for lr in learning_rates
                ],
                'create_fn': self._create_taylor_kan
            },
            # RBF KAN
            {
                'name': 'RBFKAN',
                'configs': [
                    {'hidden_size': hs, 'num_centers': nc, 'alpha': 1.0, 'learning_rate': lr}
                    for hs in hidden_sizes for nc in [20, 30, 40] for lr in learning_rates
                ],
                'create_fn': self._create_rbf_kan
            },
            # Mixture of KANs
            {
                'name': 'MixtureKAN',
                'configs': [
                    {'hidden_size': hs, 'experts_type': et, 'learning_rate': lr}
                    for hs in hidden_sizes for et in ["A", "B", "C"] for lr in learning_rates
                ],
                'create_fn': self._create_mixture_kan
            },
            # Transformer
            {
                'name': 'Transformer',
                'configs': [
                    {'hidden_size': hs, 'num_layers': nl, 'num_heads': nh, 'dropout': 0.1, 'learning_rate': lr}
                    for hs in hidden_sizes for nl in [1, 2, 3] for nh in [4, 8] for lr in learning_rates[:3]  # Less combinations for Transformer
                ],
                'create_fn': self._create_transformer_model
            },
            # LSTM
            {
                'name': 'LSTM',
                'configs': [
                    {'hidden_size': hs, 'num_layers': nl, 'dropout': 0.1, 'bidirectional': bd, 'learning_rate': lr}
                    for hs in hidden_sizes for nl in [1, 2] for bd in [True, False] for lr in learning_rates[:3]  # Less combinations for LSTM
                ],
                'create_fn': self._create_lstm_model
            }
        ]
        
        # Filter architectures if specified
        if architectures_to_test:
            architectures = [arch for arch in architectures if arch['name'] in architectures_to_test]
            self.logger.info(f"Testing {len(architectures)} architectures: {[arch['name'] for arch in architectures]}")
        
        # Total number of configurations to test
        total_configs = sum(len(arch['configs']) for arch in architectures)
        self.logger.info(f"Testing {total_configs} configurations across {len(architectures)} architectures")
        
        # Main progress bar
        pbar = tqdm(total=total_configs, desc="Architecture Testing Progress")
        
        # Loop through each architecture
        for arch_spec in architectures:
            arch_name = arch_spec['name']
            create_fn = arch_spec['create_fn']
            configs = arch_spec['configs']
            
            # Track best model for this architecture
            best_config = None
            best_r2 = float('-inf')
            
            self.logger.info(f"\n--- Testing {arch_name} Architecture ---")
            
            # Test each configuration
            for config in configs:
                config_str = ", ".join(f"{k}={v}" for k, v in config.items())
                self.logger.info(f"Configuration: {config_str}")
                
                # Create model based on architecture
                if arch_name == "OriginalKAN":
                    model = create_fn(config['hidden_size'], config['max_degree'], config['hidden_degree'])
                elif arch_name == "SplineKAN":
                    model = create_fn(config['hidden_size'], config['k'])
                elif arch_name == "WaveletKAN":
                    model = create_fn(config['hidden_size'], config['wavelet_type'])
                elif arch_name == "FourierKAN":
                    model = create_fn(config['hidden_size'], config['gridsize'])
                elif arch_name == "JacobiKAN":
                    model = create_fn(config['hidden_size'], config['degree'], config['a'], config['b'])
                elif arch_name == "ChebyshevKAN":
                    model = create_fn(config['hidden_size'], config['degree'])
                elif arch_name == "TaylorKAN":
                    model = create_fn(config['hidden_size'], config['order'])
                elif arch_name == "RBFKAN":
                    model = create_fn(config['hidden_size'], config['num_centers'], config['alpha'])
                elif arch_name == "MixtureKAN":
                    model = create_fn(config['hidden_size'], config['experts_type'])
                elif arch_name == "Transformer":
                    model = create_fn(config['hidden_size'], config['num_layers'], config['num_heads'], config['dropout'])
                elif arch_name == "LSTM":
                    model = create_fn(config['hidden_size'], config['num_layers'], config['dropout'], config['bidirectional'])
                
                # Optimize and train
                result = self._optimize_and_train(model, arch_name, config, optimization_method, num_epochs)
                all_results.append(result)
                
                # Update best config
                if result['best_val_r2'] > best_r2:
                    best_r2 = result['best_val_r2']
                    best_config = config.copy()
                
                # Save model if it's the best so far
                if result['best_val_r2'] == best_r2:
                    model_save_path = os.path.join(self.results_dir, f"best_{arch_name.lower()}_model.pt")
                    torch.save(model.state_dict(), model_save_path)
                    self.logger.info(f"Saved best {arch_name} model to {model_save_path}")
                
                # Add to DataFrame
                new_row = pd.DataFrame({
                    'architecture': [arch_name],
                    'hidden_size': [config.get('hidden_size', None)],
                    'learning_rate': [config.get('learning_rate', None)],
                    'param_count': [result['param_count']],
                    'opt_time': [result['opt_time']],
                    'val_r2': [result['best_val_r2']],
                    'val_mse': [result['best_val_mse']]
                })
                
                # Add architecture-specific parameters
                for key, value in config.items():
                    if key not in ['hidden_size', 'learning_rate']:
                        new_row[key] = [value]
                
                self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
                
                # Save results after each configuration
                self.results_df.to_csv(os.path.join(self.results_dir, "architecture_comparison_results.csv"), index=False)
                
                # Update progress bar
                pbar.update(1)
            
            # Log best config for this architecture
            best_config_str = ", ".join(f"{k}={v}" for k, v in best_config.items())
            self.logger.info(f"Best {arch_name} configuration: {best_config_str} (R²={best_r2:.4f})")
        
        # Close progress bar
        pbar.close()
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_df.to_csv(os.path.join(self.results_dir, f"architecture_comparison_results_{timestamp}.csv"), index=False)
        
        # Generate plots
        self._plot_results()
        
        return all_results
    
    def _plot_results(self):
        """Generate plots to visualize architecture comparison results."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 1. Architecture comparison (R²)
        plt.figure(figsize=(12, 6))
        sns.barplot(x='architecture', y='val_r2', data=self.results_df.groupby('architecture')['val_r2'].max().reset_index())
        plt.title('Best Validation R² by Architecture')
        plt.ylabel('Validation R²')
        plt.xlabel('Architecture')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "comparison", "architecture_r2_comparison.png"))
        
        # 1b. Filter out extreme negative values for better visualization
        filtered_df = self.results_df[self.results_df['val_r2'] > -1.0].copy()
        if len(filtered_df) > 0:
            plt.figure(figsize=(12, 6))
            sns.barplot(x='architecture', y='val_r2', data=filtered_df.groupby('architecture')['val_r2'].max().reset_index())
            plt.title('Best Validation R² by Architecture (Filtered)')
            plt.ylabel('Validation R²')
            plt.xlabel('Architecture')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "comparison", "architecture_r2_comparison_filtered.png"))
        
        # 2. Parameter count vs. performance
        plt.figure(figsize=(12, 6))
        for arch in self.results_df['architecture'].unique():
            arch_data = self.results_df[self.results_df['architecture'] == arch]
            plt.scatter(arch_data['param_count'], arch_data['val_r2'], label=arch, alpha=0.7)
        plt.title('Parameter Count vs. Validation R²')
        plt.xlabel('Parameter Count')
        plt.ylabel('Validation R²')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "performance", "params_vs_performance.png"))
        
        # 2b. Filtered parameter count vs. performance
        if len(filtered_df) > 0:
            plt.figure(figsize=(12, 6))
            for arch in filtered_df['architecture'].unique():
                arch_data = filtered_df[filtered_df['architecture'] == arch]
                plt.scatter(arch_data['param_count'], arch_data['val_r2'], label=arch, alpha=0.7)
            plt.title('Parameter Count vs. Validation R² (Filtered)')
            plt.xlabel('Parameter Count')
            plt.ylabel('Validation R²')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "performance", "params_vs_performance_filtered.png"))
        
        # 3. Optimization time vs. performance
        plt.figure(figsize=(12, 6))
        for arch in self.results_df['architecture'].unique():
            arch_data = self.results_df[self.results_df['architecture'] == arch]
            plt.scatter(arch_data['opt_time'], arch_data['val_r2'], label=arch, alpha=0.7)
        plt.title('Optimization Time vs. Validation R²')
        plt.xlabel('Optimization Time (seconds)')
        plt.ylabel('Validation R²')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "performance", "optimization_time_vs_performance.png"))
        
        # 3b. Filtered optimization time vs. performance
        if len(filtered_df) > 0:
            plt.figure(figsize=(12, 6))
            for arch in filtered_df['architecture'].unique():
                arch_data = filtered_df[filtered_df['architecture'] == arch]
                plt.scatter(arch_data['opt_time'], arch_data['val_r2'], label=arch, alpha=0.7)
            plt.title('Optimization Time vs. Validation R² (Filtered)')
            plt.xlabel('Optimization Time (seconds)')
            plt.ylabel('Validation R²')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "performance", "optimization_time_vs_performance_filtered.png"))
        
        # 4. Learning rate comparison
        if 'learning_rate' in self.results_df.columns:
            plt.figure(figsize=(14, 6))
            sns.boxplot(x='architecture', y='val_r2', hue='learning_rate', data=self.results_df)
            plt.title('Impact of Learning Rate on Performance')
            plt.ylabel('Validation R²')
            plt.xlabel('Architecture')
            plt.xticks(rotation=45)
            plt.legend(title='Learning Rate')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "hyperparams", "learning_rate_comparison.png"))
            
            # 4b. Filtered learning rate comparison
            if len(filtered_df) > 0:
                plt.figure(figsize=(14, 6))
                sns.boxplot(x='architecture', y='val_r2', hue='learning_rate', data=filtered_df)
                plt.title('Impact of Learning Rate on Performance (Filtered)')
                plt.ylabel('Validation R²')
                plt.xlabel('Architecture')
                plt.xticks(rotation=45)
                plt.legend(title='Learning Rate')
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, "hyperparams", "learning_rate_comparison_filtered.png"))
        
        # 5. Hidden size comparison
        if 'hidden_size' in self.results_df.columns:
            plt.figure(figsize=(14, 6))
            sns.boxplot(x='architecture', y='val_r2', hue='hidden_size', data=self.results_df)
            plt.title('Impact of Hidden Size on Performance')
            plt.ylabel('Validation R²')
            plt.xlabel('Architecture')
            plt.xticks(rotation=45)
            plt.legend(title='Hidden Size')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "hyperparams", "hidden_size_comparison.png"))
            
            # 5b. Filtered hidden size comparison
            if len(filtered_df) > 0:
                plt.figure(figsize=(14, 6))
                sns.boxplot(x='architecture', y='val_r2', hue='hidden_size', data=filtered_df)
                plt.title('Impact of Hidden Size on Performance (Filtered)')
                plt.ylabel('Validation R²')
                plt.xlabel('Architecture')
                plt.xticks(rotation=45)
                plt.legend(title='Hidden Size')
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, "hyperparams", "hidden_size_comparison_filtered.png"))
        
        # 6. Architecture-specific hyperparameter plots
        # For each architecture, plot the impact of its specific hyperparameters
        arch_specific_params = {
            'SplineKAN': ['k'],
            'WaveletKAN': ['wavelet_type'],
            'FourierKAN': ['gridsize'],
            'JacobiKAN': ['degree'],
            'ChebyshevKAN': ['degree'],
            'TaylorKAN': ['order'],
            'RBFKAN': ['num_centers'],
            'MixtureKAN': ['experts_type'],
            'Transformer': ['num_heads', 'num_layers'],
            'LSTM': ['num_layers', 'bidirectional']
        }
        
        for arch, params in arch_specific_params.items():
            arch_data = self.results_df[self.results_df['architecture'] == arch]
            if len(arch_data) == 0:
                continue  # Skip if no data for this architecture
                
            for param in params:
                if param in arch_data.columns:
                    # Create heatmap for this parameter vs hidden_size
                    if 'hidden_size' in arch_data.columns and param in arch_data.columns:
                        pivot_data = arch_data.pivot_table(
                            values='val_r2', 
                            index='hidden_size', 
                            columns=param, 
                            aggfunc='mean'
                        )
                        
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.4f')
                        plt.title(f'{arch}: {param} vs Hidden Size (R²)')
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.plots_dir, "hyperparams", f"{arch}_{param}_hidden_size_heatmap.png"))
        
        self.logger.info("Generated plots for architecture comparison")

def main():
    """Main function to run the architecture comparison."""
    parser = argparse.ArgumentParser(description='Compare different KAN architectures on Jane Street dataset')
    parser.add_argument('--results_dir', type=str, default='architecture_comparison_results', help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--optimization_method', type=str, default='Evolutionary', 
                       choices=['QUBO', 'IntegerProgramming', 'Evolutionary', 'GreedyHeuristic'],
                       help='Optimization method to use for OriginalKAN')
    parser.add_argument('--data_path', type=str, 
                       default="~/Interning/Kaggle/jane_street_kaggle/jane-street-real-time-market-data-forecasting/train.parquet/",
                       help='Path to Jane Street dataset')
    parser.add_argument('--subset', type=str, default=None,
                       choices=['polynomial', 'wavelet', 'fast', 'all'],
                       help='Run a subset of architectures: polynomial (Chebyshev, Jacobi, Taylor), wavelet (Wavelet, Fourier), fast (for quick testing), or all')
    parser.add_argument('--force', action='store_true', help='Skip confirmation and run immediately')
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set up logging
    log_file = f"architecture_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(os.path.join(args.results_dir, log_file))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting KAN architecture comparison")
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Optimization method: {args.optimization_method}")
    
    # Create tuner and run comparison
    tuner = KANArchitectureTuner(results_dir=args.results_dir)
    
    # Update data path if provided
    if args.data_path:
        tuner.data_cfg.data_path = args.data_path
        logger.info(f"Using data path: {args.data_path}")
    
    # Run comparison
    try:
        # Get the subset of architectures to test if specified
        architectures_to_test = None
        if args.subset:
            if args.subset == 'polynomial':
                architectures_to_test = ['OriginalKAN', 'ChebyshevKAN', 'JacobiKAN', 'TaylorKAN']
                logger.info("Running polynomial-based architectures subset")
            elif args.subset == 'wavelet':
                architectures_to_test = ['OriginalKAN', 'WaveletKAN', 'FourierKAN']
                logger.info("Running wavelet-based architectures subset")
            elif args.subset == 'fast':
                architectures_to_test = ['OriginalKAN', 'LSTM', 'Transformer', 'RBFKAN']  # Quick for testing
                logger.info("Running fast subset for testing")
            elif args.subset == 'all':
                architectures_to_test = None  # All architectures
                logger.info("Running all architectures")
        
        # If not asking for confirmation, print a message
        if args.force:
            logger.info("Skipping confirmation (--force flag used)")
        
        # Run the comparison
        tuner.compare_architectures(
            num_epochs=args.epochs, 
            optimization_method=args.optimization_method,
            architectures_to_test=architectures_to_test
        )
        logger.info("Architecture comparison completed successfully!")
    except Exception as e:
        logger.error(f"Error during architecture comparison: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 