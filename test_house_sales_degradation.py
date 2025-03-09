import math
import unittest
import logging
import os
from datetime import datetime
import gc
import time

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
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        curr_dim = hidden_size
    layers.append(nn.Linear(curr_dim, 1))
    return nn.Sequential(*layers)

class TestHouseSalesDegradation(unittest.TestCase):
    def setUp(self):
        """Initialize data and configurations."""
        self.logger = logging.getLogger("TestHouseSalesDegradation")
        self.logger.setLevel(logging.INFO)

        # Ensure directories exist
        os.makedirs("./models_house_sales_degradation", exist_ok=True)
        os.makedirs("results_house_sales_degradation", exist_ok=True)

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

        # Results DataFrame
        self.results_df = pd.DataFrame(columns=[
            'model_type', 'epoch', 'train_mse', 'val_mse',
            'grad_norm', 'weight_change'
        ])

    def _train_and_track_metrics(self, model, optimizer, model_type: str, 
                               num_epochs: int = 500, batch_size: int = 128) -> dict:
        """Train model and track detailed metrics."""
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
                # Train MSE
                y_pred_train = model(self.x_train)
                train_mse = loss_fn(y_pred_train, self.y_train).item()
                
                # Val MSE
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
                'epoch': [epoch],
                'train_mse': [train_mse],
                'val_mse': [val_mse],
                'grad_norm': [epoch_grad_norm / n_batches],
                'weight_change': [epoch_weight_change],
                'degradation_from_peak': [best_val_mse - val_mse if epoch > peak_epoch else 0.0]
            })
            self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
            
            if epoch % 10 == 0:
                print(f"[{model_type}] Epoch {epoch}/{num_epochs}, "
                      f"Train MSE={train_mse:.6f}, Val MSE={val_mse:.6f}")
                
            # Early stopping if performance severely degrades
            if val_mse > best_val_mse * 2 and epoch > peak_epoch + 50:
                print(f"Severe degradation detected. Stopping early at {val_mse:.6f} (peak was {best_val_mse:.6f})")
                
                # Fill remaining epochs with last value
                remaining_epochs = num_epochs - epoch - 1
                if remaining_epochs > 0:
                    for fill_epoch in range(epoch + 1, num_epochs):
                        new_row = pd.DataFrame({
                            'model_type': [model_type],
                            'epoch': [fill_epoch],
                            'train_mse': [train_mse],
                            'val_mse': [val_mse],
                            'grad_norm': [epoch_grad_norm / n_batches],
                            'weight_change': [epoch_weight_change],
                            'degradation_from_peak': [best_val_mse - val_mse]
                        })
                        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
                break
        
        return metrics, best_val_mse, peak_epoch

    def test_degradation_analysis(self):
        """Analyze degradation patterns."""
        print("\n==== Loading Best Models ====")
        
        # Load best KAN (MSE=0.0008)
        kan_checkpoint = torch.load('./models_house_sales/kan_final_mse_0.0008.pth')
        kan_config = kan_checkpoint['config']
        # Ensure coefficients are trainable to match saved model
        kan_config.trainable_coefficients = True
        kan = FixedKAN(kan_config)
        kan.load_state_dict(kan_checkpoint['state_dict'])
        
        # Load best MLP (depth=3, MSE=0.0003)
        mlp = build_mlp(self.input_dim, hidden_size=24, depth=3)
        mlp.load_state_dict(torch.load('./models_house_sales/mlp_depth3_mse_0.0003.pt'))
        
        # Continue training KAN
        print("\n--- Training KAN ---")
        kan_optimizer = torch.optim.Adam(kan.parameters(), lr=1e-3)
        kan_metrics, kan_best_mse, kan_peak_epoch = self._train_and_track_metrics(
            kan, kan_optimizer, 'KAN', num_epochs=500
        )
        
        # Save KAN results
        torch.save({
            'model_state': kan.state_dict(),
            'config': kan_config,
            'metrics': kan_metrics,
            'best_mse': kan_best_mse,
            'peak_epoch': kan_peak_epoch
        }, f'./models_house_sales_degradation/kan_peak{kan_peak_epoch}.pth')
        
        # Cleanup
        del kan, kan_optimizer
        torch.cuda.empty_cache()
        gc.collect()
        
        # Continue training MLP
        print("\n--- Training MLP ---")
        mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        mlp_metrics, mlp_best_mse, mlp_peak_epoch = self._train_and_track_metrics(
            mlp, mlp_optimizer, 'MLP', num_epochs=500
        )
        
        # Save MLP results
        torch.save({
            'model_state': mlp.state_dict(),
            'metrics': mlp_metrics,
            'best_mse': mlp_best_mse,
            'peak_epoch': mlp_peak_epoch
        }, f'./models_house_sales_degradation/mlp_peak{mlp_peak_epoch}.pt')
        
        # Cleanup
        del mlp, mlp_optimizer
        torch.cuda.empty_cache()
        gc.collect()
        
        # Save final results
        self.results_df.to_csv('./results_house_sales_degradation/degradation_metrics.csv', 
                             index=False, float_format='%.6f')

    def test_plot_degradation_results(self):
        """Create visualization of degradation patterns."""
        if not os.path.exists('./results_house_sales_degradation/degradation_metrics.csv'):
            self.skipTest("No results file found. Run degradation analysis first.")
        
        results = pd.read_csv('./results_house_sales_degradation/degradation_metrics.csv')
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot MSE vs Epochs
        for model in ['KAN', 'MLP']:
            model_results = results[results['model_type'] == model]
            
            # Find peak performance
            peak_idx = model_results['val_mse'].idxmin()
            peak_epoch = model_results.loc[peak_idx, 'epoch']
            peak_mse = model_results.loc[peak_idx, 'val_mse']
            final_mse = model_results['val_mse'].iloc[-1]
            degradation = (final_mse - peak_mse) / peak_mse * 100
            
            ax1.plot(model_results['epoch'], model_results['val_mse'],
                    label=f'{model} (deg: {degradation:.1f}%)', linewidth=2)
            ax1.scatter(peak_epoch, peak_mse, marker='*', s=100)
        
        ax1.set_title('Validation MSE vs Epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot degradation from peak
        for model in ['KAN', 'MLP']:
            model_results = results[results['model_type'] == model]
            ax2.plot(model_results['epoch'], model_results['degradation_from_peak'],
                    label=model, linewidth=2)
        
        ax2.set_title('Degradation from Peak Performance')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE Increase from Peak')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'./results_house_sales_degradation/degradation_comparison_{datetime.now()}.png',
                   bbox_inches='tight')
        print("Degradation comparison plot saved.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
