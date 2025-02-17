import math
import unittest
import logging
import os
from datetime import datetime
import gc

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Local imports
from CP_KAN import FixedKANConfig, FixedKAN

def count_parameters(module: nn.Module) -> int:
    """Count trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def weighted_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MSE."""
    return float(np.mean((y_true - y_pred)**2))

def build_mlp(input_dim: int, hidden_size: int, depth: int, dropout_rate: float = 0.1) -> nn.Module:
    """Build MLP with specified depth, dropout, and batch normalization."""
    layers = []
    curr_dim = input_dim
    for _ in range(depth):
        layers.extend([
            nn.Linear(curr_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),  # Add batch norm
            nn.ReLU(),
            nn.Dropout(dropout_rate)      # Add dropout
        ])
        curr_dim = hidden_size
    layers.append(nn.Linear(curr_dim, 1))
    return nn.Sequential(*layers)

class TestHouseSalesKANvsMLP(unittest.TestCase):
    # Class variable for results path
    results_path = 'results_house_sales/kan_vs_mlp_metrics.csv'

    def setUp(self):
        """Initialize data and configurations."""
        self.logger = logging.getLogger("TestHouseSalesKANvsMLP")
        self.logger.setLevel(logging.INFO)

        # Load house_sales data
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
        
        # Ensure directories exist
        os.makedirs("./models_house_sales", exist_ok=True)
        os.makedirs("results_house_sales", exist_ok=True)

        # Load existing results or create new DataFrame
        if os.path.exists(self.results_path):
            self.results_df = pd.read_csv(self.results_path)
        else:
            self.results_df = pd.DataFrame(columns=[
                'model_type', 'depth', 'epoch', 'train_mse', 'val_mse', 'param_count'
            ])

    def tearDown(self):
        """Cleanup after each test."""
        torch.cuda.empty_cache()
        gc.collect()

    def _save_metrics(self, model_type: str, depth: int, epoch: int, 
                     train_mse: float, val_mse: float, param_count: int):
        """Save metrics to results DataFrame."""
        new_row = pd.DataFrame({
            'model_type': [model_type],
            'depth': [depth],
            'epoch': [epoch],
            'train_mse': [train_mse],
            'val_mse': [val_mse],
            'param_count': [param_count]
        })
        # Ensure consistent dtypes during concatenation
        for col in self.results_df.columns:
            new_row[col] = new_row[col].astype(self.results_df[col].dtype)
        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
        
        # Save after each update
        self.results_df.to_csv(self.results_path, index=False)

    def test_1_kan_training(self):
        """Train KAN with simplified config."""
        # Clear old results at the start of a new run
        if os.path.exists(self.results_path):
            os.remove(self.results_path)
            # Reset DataFrame after deletion
            self.results_df = pd.DataFrame(columns=[
                'model_type', 'depth', 'epoch', 'train_mse', 'val_mse', 'param_count'
            ])

        print("\n==== Training KAN ====")
        
        # Simplified KAN config
        qkan_config = FixedKANConfig(
            network_shape=[self.input_dim, 20, 1],  # Single hidden layer
            max_degree=7,                          # Lower degree for stability
            complexity_weight=0.0,                # No regularization
            trainable_coefficients=True,          # Allow coefficient training
            skip_qubo_for_hidden=False,           # Use QUBO for hidden layer
            default_hidden_degree=5               # Simple hidden polynomials
        )
        
        qkan = FixedKAN(qkan_config)
        param_count = count_parameters(qkan)
        print(f"KAN parameter count: {param_count}")
        
        # Run optimize to set degrees and coefficients
        qkan.optimize(self.x_train, self.y_train)
        
        # Training loop
        num_epochs = 100
        lr = 1e-4  # Lower learning rate for stability
        
        params_to_train = []
        for layer in qkan.layers:
            params_to_train.extend([layer.combine_W, layer.combine_b])
            for neuron in layer.neurons:
                params_to_train.extend([neuron.w, neuron.b])
                if qkan_config.trainable_coefficients and neuron.coefficients is not None:
                    params_to_train.extend(list(neuron.coefficients))
        
        optimizer = torch.optim.Adam(params_to_train, lr=lr)
        loss_mse = nn.MSELoss()
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            y_pred = qkan(self.x_train)
            
            # MSE loss
            loss = loss_mse(y_pred, self.y_train)
            
            loss.backward()
            optimizer.step()
            
            # Compute metrics every epoch
            with torch.no_grad():
                # Train MSE
                y_pred_train = qkan(self.x_train)
                train_mse = loss_mse(y_pred_train, self.y_train).item()
                
                # Val MSE
                y_pred_val = qkan(self.x_val)
                val_mse = loss_mse(y_pred_val, self.y_val).item()
                
                if epoch % 10 == 0:  # Only print every 10 epochs
                    print(f"[KAN] Epoch {epoch}/{num_epochs}, Train MSE={train_mse:.4f}, Val MSE={val_mse:.4f}")
                self._save_metrics('KAN', 0, epoch, train_mse, val_mse, param_count)  # Use depth=0 for KAN
        
        # Save final model
        save_path = f"./models_house_sales/kan_final_mse_{val_mse:.4f}.pth"
        qkan.save_model(save_path)
        print(f"KAN model saved to: {save_path}")

    def test_2_mlp_depths(self):
        """Train MLPs of different depths."""
        hidden_size = 24  # Middle ground between 16 and 32
        depths = [2, 3, 4]  # Different depths to try
        num_epochs = 100
        lr = 1e-3
        weight_decay = 0.001
        batch_size = 128
        patience = 15
        
        for depth in depths:
            print(f"\n==== Training MLP (depth={depth}) ====")
            
            mlp = build_mlp(self.input_dim, hidden_size, depth)
            param_count = count_parameters(mlp)
            print(f"MLP (depth={depth}) parameter count: {param_count}")
            
            optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=weight_decay)
            loss_mse = nn.MSELoss()
            
            # For early stopping
            best_val_mse = float('inf')
            patience_counter = 0
            best_epoch = 0
            
            # Training loop
            for epoch in range(num_epochs):
                mlp.train()
                
                # Mini-batch training
                n_batches = math.ceil(len(self.x_train) / batch_size)
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(self.x_train))
                    
                    x_batch = self.x_train[start_idx:end_idx]
                    y_batch = self.y_train[start_idx:end_idx]
                    
                    optimizer.zero_grad()
                    y_pred = mlp(x_batch)
                    
                    # MSE loss
                    loss = loss_mse(y_pred, y_batch)
                    
                    loss.backward()
                    optimizer.step()
                
                # Compute metrics every epoch
                mlp.eval()
                with torch.no_grad():
                    # Train MSE
                    y_pred_train = mlp(self.x_train)
                    train_mse = loss_mse(y_pred_train, self.y_train).item()
                    
                    # Val MSE
                    y_pred_val = mlp(self.x_val)
                    val_mse = loss_mse(y_pred_val, self.y_val).item()
                    
                # Save metrics every epoch
                self._save_metrics(f'MLP', depth, epoch, train_mse, val_mse, param_count)
                
                # Print progress every 10 epochs
                if epoch % 10 == 0:
                    print(f"[MLP-{depth}] Epoch {epoch}/{num_epochs}, Train MSE={train_mse:.4f}, Val MSE={val_mse:.4f}")
                
                # Early stopping check
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}. Best val MSE={best_val_mse:.4f} at epoch {best_epoch}")
                        break
            
            # Save final model
            save_path = f"./models_house_sales/mlp_depth{depth}_mse_{val_mse:.4f}.pt"
            torch.save(mlp.state_dict(), save_path)
            print(f"MLP (depth={depth}) saved to: {save_path}")
            
            # Cleanup
            del mlp
            torch.cuda.empty_cache()
            gc.collect()

    def test_3_plot_results(self):
        """Create comparison plots from saved metrics."""
        if not os.path.exists(self.results_path):
            self.skipTest("No results file found. Run KAN and MLP tests first.")
        
        results = pd.read_csv(self.results_path)
        
        # Skip plotting if no data
        if len(results) == 0:
            self.skipTest("Results file is empty. Run KAN and MLP tests first.")
            
        # Plot MSE vs Epochs
        plt.figure(figsize=(12, 6))
        
        # Plot validation MSE only
        # KAN
        kan_results = results[results['model_type'] == 'KAN']
        if len(kan_results) > 0:  # Only plot if we have KAN results
            plt.plot(kan_results['epoch'], kan_results['val_mse'], 
                    label=f'KAN [{kan_results.iloc[0]["param_count"]} params]', 
                    color='blue', linewidth=2)
        
        # MLPs
        colors = ['orange', 'green', 'red']
        for depth, color in zip([2, 3, 4], colors):
            mlp_d = results[(results['model_type'] == 'MLP') & (results['depth'] == depth)]
            if len(mlp_d) > 0:  # Only plot if we have results for this depth
                param_count = mlp_d.iloc[0]['param_count']
                plt.plot(mlp_d['epoch'], mlp_d['val_mse'], 
                        label=f'MLP-{depth} [{param_count} params]', 
                        color=color, linewidth=2)
        
        plt.title("KAN vs MLP Depths: Validation MSE vs Epoch\nHouse Sales Regression (log-target)")
        plt.xlabel("Epoch")
        plt.xlim(0, 50)  # Limit x-axis to 50 epochs
        plt.ylabel("MSE (log scale)")
        plt.yscale('log')  # Use log scale for MSE
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'./results_house_sales/kan_vs_mlp_comparison_{datetime.now()}.png', bbox_inches='tight')
        print("Comparison plot saved to: ./results_house_sales/kan_vs_mlp_comparison.png")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
