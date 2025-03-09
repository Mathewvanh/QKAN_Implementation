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
            'model_type', 'learning_rate', 'epoch', 'train_mse', 'val_mse',
            'grad_norm', 'weight_change'
        ])

    def _train_and_track_metrics(self, model, optimizer, model_type: str, 
                               learning_rate: float, num_epochs: int = 200,
                               batch_size: int = 128):
        """Train model and track detailed metrics."""
        metrics = defaultdict(list)
        best_val_mse = float('inf')  # For MSE, lower is better
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
                
                # MSE loss
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
            
            # Track peak performance (lower MSE is better)
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
                'learning_rate': [learning_rate],
                'epoch': [epoch],
                'train_mse': [train_mse],
                'val_mse': [val_mse],
                'grad_norm': [epoch_grad_norm / n_batches],
                'weight_change': [epoch_weight_change],
                'degradation_from_peak': [val_mse - best_val_mse if epoch > peak_epoch else 0.0]
            })
            self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
            # Save metrics after each update
            self.results_df.to_csv('./results_house_sales_degradation/degradation_metrics.csv', 
                                 index=False, float_format='%.6f')
            
            if epoch % 10 == 0:
                print(f"[{model_type}] LR={learning_rate:.1e} Epoch {epoch}/{num_epochs}, "
                      f"Train MSE={train_mse:.6f}, Val MSE={val_mse:.6f}")
                
            # Early stopping if performance severely degrades
            # Allow more degradation to show the pattern
            if val_mse > best_val_mse * 3 and epoch > peak_epoch + 50:
                print(f"Severe degradation detected. Stopping early at {val_mse:.6f} (peak was {best_val_mse:.6f})")
                
                # Fill remaining epochs with last value
                remaining_epochs = num_epochs - epoch - 1
                if remaining_epochs > 0:
                    for fill_epoch in range(epoch + 1, num_epochs):
                        metrics['epoch'].append(fill_epoch)
                        metrics['train_mse'].append(train_mse)
                        metrics['val_mse'].append(val_mse)
                        metrics['grad_norm'].append(epoch_grad_norm / n_batches)
                        metrics['weight_change'].append(epoch_weight_change)
                        
                        new_row = pd.DataFrame({
                            'model_type': [model_type],
                            'learning_rate': [learning_rate],
                            'epoch': [fill_epoch],
                            'train_mse': [train_mse],
                            'val_mse': [val_mse],
                            'grad_norm': [epoch_grad_norm / n_batches],
                            'weight_change': [epoch_weight_change],
                            'degradation_from_peak': [val_mse - best_val_mse]
                        })
                        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
                break
        
        return metrics, best_val_mse, peak_epoch

    def test_degradation_analysis(self):
        """Analyze degradation patterns with different learning rates."""
        learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
        num_epochs = 200
        
        # Clear old results at the start of a new run
        self.results_df = pd.DataFrame(columns=[
            'model_type', 'learning_rate', 'epoch', 'train_mse', 'val_mse',
            'grad_norm', 'weight_change'
        ])
        
        # KAN configuration
        qkan_config = FixedKANConfig(
            network_shape=[self.input_dim, 20, 1],  # Single hidden layer
            max_degree=7,                          # Higher degree for house prices
            complexity_weight=0.0,                # No regularization
            trainable_coefficients=True,          # Allow coefficient training
            skip_qubo_for_hidden=False,           # Use QUBO for hidden layer
            default_hidden_degree=5               # Simple hidden polynomials
        )
        
        # MLP configuration
        mlp_hidden_size = 24
        mlp_depth = 3
        
        for lr in learning_rates:
            print(f"\n==== Testing Learning Rate: {lr:.1e} ====")
            
            # Train KAN
            print("\n--- Training KAN ---")
            qkan = FixedKAN(qkan_config)
            qkan.optimize(self.x_train, self.y_train)
            
            params_to_train = []
            for layer in qkan.layers:
                params_to_train.extend([layer.combine_W, layer.combine_b])
                for neuron in layer.neurons:
                    params_to_train.extend([neuron.w, neuron.b])
            
            kan_optimizer = torch.optim.Adam(params_to_train, lr=lr)
            kan_metrics, kan_best_mse, kan_peak_epoch = self._train_and_track_metrics(
                qkan, kan_optimizer, 'KAN', lr, num_epochs
            )
            
            # Save KAN model at peak
            torch.save({
                'model_state': qkan.state_dict(),
                'config': qkan_config,
                'metrics': kan_metrics,
                'best_mse': kan_best_mse,
                'peak_epoch': kan_peak_epoch
            }, f'./models_house_sales_degradation/kan_lr{lr:.1e}_peak{kan_peak_epoch}.pth')
            
            # Cleanup
            del qkan, kan_optimizer
            torch.cuda.empty_cache()
            gc.collect()
            
            # Train MLP
            print("\n--- Training MLP ---")
            mlp = build_mlp(self.input_dim, mlp_hidden_size, mlp_depth)
            mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
            
            mlp_metrics, mlp_best_mse, mlp_peak_epoch = self._train_and_track_metrics(
                mlp, mlp_optimizer, 'MLP', lr, num_epochs
            )
            
            # Save MLP model at peak
            torch.save({
                'model_state': mlp.state_dict(),
                'metrics': mlp_metrics,
                'best_mse': mlp_best_mse,
                'peak_epoch': mlp_peak_epoch
            }, f'./models_house_sales_degradation/mlp_lr{lr:.1e}_peak{mlp_peak_epoch}.pt')
            
            # Cleanup
            del mlp, mlp_optimizer
            torch.cuda.empty_cache()
            gc.collect()
        
        # Save final results with summary
        summary_df = pd.DataFrame(columns=['model_type', 'learning_rate', 'peak_mse', 'peak_epoch', 
                                         'final_mse', 'max_degradation'])
        
        for model in ['KAN', 'MLP']:
            for lr in learning_rates:
                model_data = self.results_df[(self.results_df['model_type'] == model) & 
                                           (self.results_df['learning_rate'] == lr)]
                if len(model_data) > 0:
                    peak_mse = model_data['val_mse'].min()  # Lower MSE is better
                    peak_epoch = model_data.loc[model_data['val_mse'].idxmin(), 'epoch']
                    final_mse = model_data['val_mse'].iloc[-1]
                    max_degradation = model_data['degradation_from_peak'].max()
                    
                    summary_df = pd.concat([summary_df, pd.DataFrame({
                        'model_type': [model],
                        'learning_rate': [lr],
                        'peak_mse': [peak_mse],
                        'peak_epoch': [peak_epoch],
                        'final_mse': [final_mse],
                        'max_degradation': [max_degradation]
                    })], ignore_index=True)
        
        summary_df.to_csv('./results_house_sales_degradation/degradation_summary.csv', 
                         index=False, float_format='%.6f')

    def test_plot_degradation_results(self):
        """Create visualization of degradation patterns."""
        # Run degradation analysis if metrics don't exist
        if not os.path.exists('./results_house_sales_degradation/degradation_metrics.csv'):
            self.test_degradation_analysis()
        
        # Load and plot results
        results = pd.read_csv('./results_house_sales_degradation/degradation_metrics.csv')
        learning_rates = sorted(results['learning_rate'].unique())
        
        # Create figure with two rows of subplots per learning rate
        fig = plt.figure(figsize=(15, 7*len(learning_rates)))
        gs = plt.GridSpec(2*len(learning_rates), 1, height_ratios=[2, 1]*len(learning_rates))
        
        for i, lr in enumerate(learning_rates):
            lr_results = results[results['learning_rate'] == lr]
            
            # Main performance plot
            ax1 = fig.add_subplot(gs[2*i])
            
            # Plot KAN results
            kan_results = lr_results[lr_results['model_type'] == 'KAN']
            if len(kan_results) > 0:
                kan_peak = kan_results['val_mse'].min()  # Lower MSE is better
                kan_peak_epoch = kan_results.loc[kan_results['val_mse'].idxmin(), 'epoch']
                kan_final = kan_results['val_mse'].iloc[-1]
                kan_deg = (kan_final - kan_peak) / kan_peak * 100
                
                ax1.plot(kan_results['epoch'], kan_results['val_mse'],
                        label=f'KAN (deg: {kan_deg:.1f}%)', color='blue', linewidth=2)
                ax1.scatter(kan_peak_epoch, kan_peak, color='blue', s=100,
                          marker='*', label='KAN Peak')
            
            # Plot MLP results
            mlp_results = lr_results[lr_results['model_type'] == 'MLP']
            if len(mlp_results) > 0:
                mlp_peak = mlp_results['val_mse'].min()  # Lower MSE is better
                mlp_peak_epoch = mlp_results.loc[mlp_results['val_mse'].idxmin(), 'epoch']
                mlp_final = mlp_results['val_mse'].iloc[-1]
                mlp_deg = (mlp_final - mlp_peak) / mlp_peak * 100
                
                # Find early stopping point if it occurred
                early_stop = mlp_results['epoch'].max() < 199
                if early_stop:
                    stop_epoch = mlp_results['epoch'].max()
                    ax1.axvline(x=stop_epoch, color='red', linestyle='--', alpha=0.3)
                    ax1.text(stop_epoch+5, ax1.get_ylim()[0], 
                            f'Early stop\n{mlp_deg:.1f}% deg', 
                            color='red', alpha=0.7)
                
                ax1.plot(mlp_results['epoch'], mlp_results['val_mse'],
                        label=f'MLP (deg: {mlp_deg:.1f}%)', color='red', linewidth=2)
                ax1.scatter(mlp_peak_epoch, mlp_peak, color='red', s=100,
                          marker='*', label='MLP Peak')
            
            ax1.set_title(f'Learning Rate: {lr:.1e}')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Validation MSE')
            ax1.set_yscale('log')  # Log scale for MSE
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='center right')
            
            # Degradation subplot
            ax2 = fig.add_subplot(gs[2*i + 1])
            
            if len(kan_results) > 0:
                ax2.plot(kan_results['epoch'], kan_results['degradation_from_peak'],
                        color='blue', linewidth=2, label='KAN')
            
            if len(mlp_results) > 0:
                ax2.plot(mlp_results['epoch'], mlp_results['degradation_from_peak'],
                        color='red', linewidth=2, label='MLP')
            
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MSE Increase from Peak')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='center right')
        
        plt.tight_layout()
        plt.savefig(f'./results_house_sales_degradation/degradation_comparison_{datetime.now()}.png',
                   bbox_inches='tight')
        print("Degradation comparison plot saved.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
