import math
import os
from datetime import datetime
import gc
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
from collections import defaultdict

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Local imports
from CP_KAN import FixedKANConfig, FixedKAN

def build_mlp(input_dim: int, hidden_size: int, depth: int, dropout_rate: float = 0.1) -> nn.Module:
    """Build MLP with specified depth, dropout, and batch normalization."""
    layers = []
    curr_dim = input_dim
    for _ in range(depth):
        layers.extend([
            nn.Linear(curr_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        curr_dim = hidden_size
    layers.append(nn.Linear(curr_dim, 1))
    return nn.Sequential(*layers)

class ModelTuner:
    def __init__(self, results_dir: str = "tuning_results"):
        """Initialize tuner with results directory."""
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Results tracking
        self.results_df = pd.DataFrame(columns=[
            'model_type', 'config', 'epoch', 'train_mse', 'val_mse',
            'grad_norm', 'weight_change'
        ])
        
        # Load and preprocess data
        self._load_data()

    def _load_data(self):
        """Load and preprocess house sales data."""
        dataset = load_dataset(
            "inria-soda/tabular-benchmark",
            data_files="reg_num/house_sales.csv",
            split="train"
        )
        df = pd.DataFrame(dataset)
        label_col = "target" if "target" in df.columns else df.columns[-1]

        # Get features and target
        y = df[label_col].values.astype(np.float32)
        X = df.drop(columns=[label_col]).values.astype(np.float32)

        # Log transform target
        y = np.log1p(y)  # log(1+y)

        # Normalize features
        X = StandardScaler().fit_transform(X)

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert to torch tensors
        self.x_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
        self.x_val = torch.tensor(X_val, dtype=torch.float32)
        self.y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)
        self.input_dim = X_train.shape[1]

    def _train_and_evaluate(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                          model_type: str, config: Dict[str, Any], num_epochs: int = 200,
                          batch_size: int = 128) -> Tuple[Dict[str, List], float, int]:
        """Train model and track metrics."""
        metrics = defaultdict(list)
        best_val_mse = float('inf')
        peak_epoch = 0
        loss_fn = nn.MSELoss()
        
        # Store initial weights
        prev_weights = {name: param.clone().detach() 
                       for name, param in model.named_parameters()}

        for epoch in range(num_epochs):
            model.train()
            
            # Mini-batch training
            n_batches = math.ceil(len(self.x_train) / batch_size)
            epoch_grad_norm = 0.0
            epoch_weight_change = 0.0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(self.x_train))
                
                x_batch = self.x_train[start_idx:end_idx]
                y_batch = self.y_train[start_idx:end_idx]
                
                optimizer.zero_grad()
                y_pred = model(x_batch)
                
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                
                # Calculate gradient norm
                grad_norm = torch.norm(torch.stack([
                    p.grad.norm() for p in model.parameters() if p.grad is not None
                ]))
                epoch_grad_norm += grad_norm.item()
                
                optimizer.step()
            
            # Calculate weight changes
            for name, param in model.named_parameters():
                weight_change = torch.norm(param.data - prev_weights[name])
                epoch_weight_change += weight_change.item()
                prev_weights[name] = param.clone().detach()
            
            # Compute metrics
            model.eval()
            with torch.no_grad():
                y_pred_train = model(self.x_train)
                train_mse = loss_fn(y_pred_train, self.y_train).item()
                
                y_pred_val = model(self.x_val)
                val_mse = loss_fn(y_pred_val, self.y_val).item()
            
            # Track peak performance
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                peak_epoch = epoch
            
            # Store metrics
            metrics['epoch'].append(epoch)
            metrics['train_mse'].append(train_mse)
            metrics['val_mse'].append(val_mse)
            metrics['grad_norm'].append(epoch_grad_norm / n_batches)
            metrics['weight_change'].append(epoch_weight_change)
            
            # Add to results DataFrame
            new_row = pd.DataFrame({
                'model_type': [model_type],
                'config': [str(config)],
                'epoch': [epoch],
                'train_mse': [train_mse],
                'val_mse': [val_mse],
                'grad_norm': [epoch_grad_norm / n_batches],
                'weight_change': [epoch_weight_change]
            })
            self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
            
            if epoch % 10 == 0:
                print(f"[{model_type}] Config={config} Epoch {epoch}/{num_epochs}, "
                      f"Train MSE={train_mse:.6f}, Val MSE={val_mse:.6f}")
        
        return metrics, best_val_mse, peak_epoch

    def tune_kan(self, param_grid: Dict[str, List[Any]], num_epochs: int = 200):
        """Tune KAN hyperparameters."""
        print("\n=== Tuning KAN ===")
        
        # Track best configuration
        best_mse = float('inf')
        best_config = None
        
        # Grid search
        for max_degree in param_grid['max_degree']:
            for hidden_size in param_grid['hidden_size']:
                for hidden_degree in param_grid['hidden_degree']:
                    for lr in param_grid['learning_rate']:
                        config = {
                            'max_degree': max_degree,
                            'hidden_size': hidden_size,
                            'hidden_degree': hidden_degree,
                            'learning_rate': lr
                        }
                        
                        print(f"\nTrying KAN config: {config}")
                        
                        # Create and configure KAN
                        kan_config = FixedKANConfig(
                            network_shape=[self.input_dim, hidden_size, 1],
                            max_degree=max_degree,
                            complexity_weight=0.0,
                            trainable_coefficients=True,
                            skip_qubo_for_hidden=False,
                            default_hidden_degree=hidden_degree
                        )
                        
                        kan = FixedKAN(kan_config)
                        kan.optimize(self.x_train, self.y_train)
                        
                        # Setup optimizer
                        params_to_train = []
                        for layer in kan.layers:
                            params_to_train.extend([layer.combine_W, layer.combine_b])
                            for neuron in layer.neurons:
                                params_to_train.extend([neuron.w, neuron.b])
                        
                        optimizer = torch.optim.Adam(params_to_train, lr=lr)
                        
                        # Train and evaluate
                        metrics, val_mse, peak_epoch = self._train_and_evaluate(
                            kan, optimizer, 'KAN', config, num_epochs
                        )
                        
                        # Update best if needed
                        if val_mse < best_mse:
                            best_mse = val_mse
                            best_config = config
                            
                            # Save best model
                            torch.save({
                                'model_state': kan.state_dict(),
                                'config': kan_config,
                                'metrics': metrics,
                                'val_mse': val_mse,
                                'peak_epoch': peak_epoch
                            }, f'{self.results_dir}/kan_best.pth')
                        
                        # Cleanup
                        del kan, optimizer
                        torch.cuda.empty_cache()
                        gc.collect()
        
        print(f"\nBest KAN config: {best_config}")
        print(f"Best validation MSE: {best_mse:.6f}")
        
        # Save results
        self.results_df.to_csv(f'{self.results_dir}/tuning_results.csv', index=False)

    def tune_mlp(self, param_grid: Dict[str, List[Any]], num_epochs: int = 200):
        """Tune MLP hyperparameters."""
        print("\n=== Tuning MLP ===")
        
        # Track best configuration
        best_mse = float('inf')
        best_config = None
        
        # Grid search
        for hidden_size in param_grid['hidden_size']:
            for depth in param_grid['depth']:
                for dropout in param_grid['dropout']:
                    for lr in param_grid['learning_rate']:
                        config = {
                            'hidden_size': hidden_size,
                            'depth': depth,
                            'dropout': dropout,
                            'learning_rate': lr
                        }
                        
                        print(f"\nTrying MLP config: {config}")
                        
                        # Create MLP
                        mlp = build_mlp(self.input_dim, hidden_size, depth, dropout)
                        optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
                        
                        # Train and evaluate
                        metrics, val_mse, peak_epoch = self._train_and_evaluate(
                            mlp, optimizer, 'MLP', config, num_epochs
                        )
                        
                        # Update best if needed
                        if val_mse < best_mse:
                            best_mse = val_mse
                            best_config = config
                            
                            # Save best model
                            torch.save({
                                'model_state': mlp.state_dict(),
                                'config': config,
                                'metrics': metrics,
                                'val_mse': val_mse,
                                'peak_epoch': peak_epoch
                            }, f'{self.results_dir}/mlp_best.pt')
                        
                        # Cleanup
                        del mlp, optimizer
                        torch.cuda.empty_cache()
                        gc.collect()
        
        print(f"\nBest MLP config: {best_config}")
        print(f"Best validation MSE: {best_mse:.6f}")
        
        # Save results
        self.results_df.to_csv(f'{self.results_dir}/tuning_results.csv', index=False)

    def plot_results(self):
        """Plot tuning results."""
        if len(self.results_df) == 0:
            print("No results to plot. Run tuning first.")
            return
        
        # Group by model type and config
        grouped = self.results_df.groupby(['model_type', 'config'])
        
        # Create plot
        plt.figure(figsize=(15, 10))
        
        for (model_type, config), data in grouped:
            label = f"{model_type} {config}"
            plt.plot(data['epoch'], data['val_mse'], label=label, alpha=0.7)
        
        plt.title("Validation MSE vs Epoch for Different Configurations")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'{self.results_dir}/tuning_comparison_{datetime.now()}.png',
                   bbox_inches='tight')
        print("Tuning comparison plot saved.")

if __name__ == "__main__":
    # Initialize tuner
    tuner = ModelTuner()
    
    # KAN parameter grid
    kan_params = {
        'max_degree': [5, 7, 9],
        'hidden_size': [16, 20, 24],
        'hidden_degree': [3, 5, 7],
        'learning_rate': [1e-2, 1e-3]
    }
    
    # MLP parameter grid
    mlp_params = {
        'hidden_size': [20, 24, 28],
        'depth': [2, 3, 4],
        'dropout': [0.1, 0.2],
        'learning_rate': [1e-2, 1e-3]
    }
    
    # Run tuning
    tuner.tune_kan(kan_params)
    tuner.tune_mlp(mlp_params)
    
    # Plot results
    tuner.plot_results()
