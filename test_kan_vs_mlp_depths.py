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

# Local imports
from data_pipeline_js_config import DataConfig
from data_pipeline import DataPipeline
from CP_KAN import FixedKANConfig, FixedKAN

def count_parameters(module: nn.Module) -> int:
    """Count trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    """Compute weighted R² score."""
    numerator = np.sum(w * (y_true - y_pred)**2)
    denominator = np.sum(w * (y_true**2))
    if denominator < 1e-12:
        return 0.0
    return float(1.0 - numerator/denominator)

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

class TestKANvsMLPDepths(unittest.TestCase):
    def setUp(self):
        """Initialize data and configurations."""
        self.logger = logging.getLogger("TestKANvsMLPDepths")
        self.logger.setLevel(logging.INFO)

        # Reduced dataset size
        self.data_cfg = DataConfig(
            data_path="~/Interning/Kaggle/jane_street_kaggle/jane-street-real-time-market-data-forecasting/train.parquet/",
            n_rows=200000,  # Reduced from 200k
            train_ratio=0.7,
            feature_cols=[f'feature_{i:02d}' for i in range(79)],
            target_col="responder_6",
            weight_col="weight",
            date_col="date_id"
        )

        # Load and preprocess data
        pipeline = DataPipeline(self.data_cfg, self.logger)
        train_df, train_target, train_weight, val_df, val_target, val_weight = pipeline.load_and_preprocess_data()

        # Convert to numpy then torch
        self.x_train = torch.tensor(train_df.to_numpy(), dtype=torch.float32)
        self.y_train = torch.tensor(train_target.to_numpy(), dtype=torch.float32).squeeze(-1)
        self.w_train = torch.tensor(train_weight.to_numpy(), dtype=torch.float32).squeeze(-1)

        self.x_val = torch.tensor(val_df.to_numpy(), dtype=torch.float32)
        self.y_val = torch.tensor(val_target.to_numpy(), dtype=torch.float32).squeeze(-1)
        self.w_val = torch.tensor(val_weight.to_numpy(), dtype=torch.float32).squeeze(-1)

        self.input_dim = self.x_train.shape[1]
        
        # Ensure directories exist
        os.makedirs("./models_janestreet", exist_ok=True)
        os.makedirs("results_js", exist_ok=True)

        # Load existing results_js or create new DataFrame
        results_path = 'results_js/kan_vs_mlp_metrics.csv'
        if os.path.exists(results_path):
            self.results_df = pd.read_csv(results_path)
        else:
            self.results_df = pd.DataFrame(columns=[
                'model_type', 'depth', 'epoch', 'train_r2', 'val_r2', 'param_count'
            ])

    def tearDown(self):
        """Cleanup after each test."""
        torch.cuda.empty_cache()
        gc.collect()

    def _save_metrics(self, model_type: str, depth: int, epoch: int, 
                     train_r2: float, val_r2: float, param_count: int):
        """Save metrics to results_js DataFrame."""
        new_row = pd.DataFrame({
            'model_type': [model_type],
            'depth': [depth],
            'epoch': [epoch],
            'train_r2': [train_r2],
            'val_r2': [val_r2],
            'param_count': [param_count]
        })
        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
        
        # Save after each update
        self.results_df.to_csv('./results_js/kan_vs_mlp_metrics.csv', index=False)

    def test_1_kan_training(self):
        """Train KAN with simplified config."""
        print("\n==== Training KAN ====")
        
        # Simplified KAN config
        qkan_config = FixedKANConfig(
            network_shape=[self.input_dim, 20, 1],  # Increased hidden layer
            max_degree=5,                          # Lower degree for stability
            complexity_weight=0.0,                # Add some regularization
            trainable_coefficients=True,           # Allow coefficient training
            skip_qubo_for_hidden=False,            # Skip QUBO for faster training
            default_hidden_degree=5               # Simple hidden polynomials
        )
        
        qkan = FixedKAN(qkan_config)
        param_count = count_parameters(qkan)
        print(f"KAN parameter count: {param_count}")
        
        # Run optimize to set degrees and coefficients
        qkan.optimize(self.x_train, self.y_train.unsqueeze(-1))
        
        # Training loop
        num_epochs = 100  # Reduced from 500
        lr = 1e-3
        
        params_to_train = []
        for layer in qkan.layers:
            params_to_train.extend([layer.combine_W, layer.combine_b])
            for neuron in layer.neurons:
                params_to_train.extend([neuron.w, neuron.b])
        
        optimizer = torch.optim.Adam(params_to_train, lr=lr)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            y_pred = qkan(self.x_train).squeeze(-1)
            
            # Weighted MSE loss
            numerator = torch.sum(self.w_train * (self.y_train - y_pred)**2)
            denominator = torch.sum(self.w_train)
            loss = numerator / (denominator + 1e-12)
            
            loss.backward()
            optimizer.step()
            
            # Compute metrics every 10 epochs
            if epoch % 10 == 0:
                with torch.no_grad():
                    # Train R²
                    y_pred_train = qkan(self.x_train).squeeze(-1).cpu().numpy()
                    train_r2 = weighted_r2(
                        self.y_train.cpu().numpy(),
                        y_pred_train,
                        self.w_train.cpu().numpy()
                    )
                    
                    # Val R²
                    y_pred_val = qkan(self.x_val).squeeze(-1).cpu().numpy()
                    val_r2 = weighted_r2(
                        self.y_val.cpu().numpy(),
                        y_pred_val,
                        self.w_val.cpu().numpy()
                    )
                    
                    print(f"[KAN] Epoch {epoch}/{num_epochs}, Train R²={train_r2:.4f}, Val R²={val_r2:.4f}")
                    self._save_metrics('KAN', 0, epoch, train_r2, val_r2, param_count)  # Use depth=0 for KAN
        
        # Save final model
        save_path = f"./models_janestreet/kan_final_valr2_{val_r2:.4f}.pth"
        qkan.save_model(save_path)
        print(f"KAN model saved to: {save_path}")

    def test_2_mlp_depths(self):
        """Train MLPs of different depths."""
        hidden_size = 24  # Middle ground between 16 and 32
        depths = [2, 3, 4]  # Different depths to try
        num_epochs = 100    # Reduced from 500
        lr = 1e-3          # Back to original learning rate
        weight_decay = 0.001 # Reduced L2 regularization
        batch_size = 128
        patience = 15      # Increased patience
        
        for depth in depths:
            print(f"\n==== Training MLP (depth={depth}) ====")
            
            mlp = build_mlp(self.input_dim, hidden_size, depth)
            param_count = count_parameters(mlp)
            print(f"MLP (depth={depth}) parameter count: {param_count}")
            
            optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=weight_decay)
            
            # For early stopping
            best_val_r2 = float('-inf')
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
                    w_batch = self.w_train[start_idx:end_idx]
                    
                    optimizer.zero_grad()
                    y_pred = mlp(x_batch).squeeze(-1)
                    
                    # Weighted MSE
                    numerator = torch.sum(w_batch * (y_batch - y_pred)**2)
                    denominator = torch.sum(w_batch)
                    loss = numerator / (denominator + 1e-12)
                    
                    loss.backward()
                    optimizer.step()
                
                # Compute metrics every 10 epochs
                if epoch % 10 == 0:
                    mlp.eval()
                    with torch.no_grad():
                        # Train R²
                        y_pred_train = mlp(self.x_train).squeeze(-1).cpu().numpy()
                        train_r2 = weighted_r2(
                            self.y_train.cpu().numpy(),
                            y_pred_train,
                            self.w_train.cpu().numpy()
                        )
                        
                        # Val R²
                        y_pred_val = mlp(self.x_val).squeeze(-1).cpu().numpy()
                        val_r2 = weighted_r2(
                            self.y_val.cpu().numpy(),
                            y_pred_val,
                            self.w_val.cpu().numpy()
                        )
                        
                    print(f"[MLP-{depth}] Epoch {epoch}/{num_epochs}, Train R²={train_r2:.4f}, Val R²={val_r2:.4f}")
                    self._save_metrics(f'MLP', depth, epoch, train_r2, val_r2, param_count)
                    
                    # Early stopping check
                    if val_r2 > best_val_r2:
                        best_val_r2 = val_r2
                        best_epoch = epoch
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Early stopping at epoch {epoch}. Best val R²={best_val_r2:.4f} at epoch {best_epoch}")
                            break
            
            # Save final model
            save_path = f"./models_janestreet/mlp_depth{depth}_valr2_{val_r2:.4f}.pt"
            torch.save(mlp.state_dict(), save_path)
            print(f"MLP (depth={depth}) saved to: {save_path}")
            
            # Cleanup
            del mlp
            torch.cuda.empty_cache()
            gc.collect()

    def test_3_plot_results(self):
        """Create comparison plots from saved metrics."""
        if not os.path.exists('results_js/kan_vs_mlp_metrics.csv'):
            self.skipTest("No results_js file found. Run KAN and MLP tests first.")
        
        results = pd.read_csv('results_js/kan_vs_mlp_metrics.csv')
        
        # Skip plotting if no data
        if len(results) == 0:
            self.skipTest("Results file is empty. Run KAN and MLP tests first.")
            
        # Plot R² vs Epochs
        plt.figure(figsize=(12, 6))
        
        # Plot validation R² only
        # KAN
        kan_results = results[results['model_type'] == 'KAN']
        plt.plot(kan_results['epoch'], kan_results['val_r2'], 
                label=f'KAN [{kan_results.iloc[0]["param_count"]} params]', 
                color='blue', linewidth=2)
        
        # MLPs
        colors = ['orange', 'green', 'red']
        for depth, color in zip([2, 3, 4], colors):
            mlp_d = results[(results['model_type'] == 'MLP') & (results['depth'] == depth)]
            param_count = mlp_d.iloc[0]['param_count']
            
            plt.plot(mlp_d['epoch'], mlp_d['val_r2'], 
                    label=f'MLP-{depth} [{param_count} params]', 
                    color=color, linewidth=2)
        
        plt.title("KAN vs MLP Depths: Validation R² vs Epoch\nJane Street Market Prediction")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted R²")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plt.savefig('./results_js/kan_vs_mlp_comparison.png', bbox_inches='tight')
        print("Comparison plot saved to: ./results_js/kan_vs_mlp_comparison.png")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
