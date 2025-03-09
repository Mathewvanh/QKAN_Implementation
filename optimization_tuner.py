import math
import os
import time
from datetime import datetime
import gc
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
from collections import defaultdict
from tqdm import tqdm, trange

# Import Jane Street data pipeline instead of datasets
import logging
from data_pipeline_js_config import DataConfig
from data_pipeline import DataPipeline

# Local imports
from CP_KAN import FixedKANConfig, FixedKAN

def count_parameters(module: nn.Module) -> int:
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    """Compute weighted R² score using the Jane Street competition formula."""
    # Convert inputs to numpy if they're torch tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(w, torch.Tensor):
        w = w.cpu().numpy()
    
    # Ensure arrays are flattened
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    w = w.flatten()
    
    # Jane Street competition formula
    numerator = np.sum(w * (y_true - y_pred)**2)
    denominator = np.sum(w * (y_true**2))
    if denominator < 1e-12:
        return 0.0
    return float(1.0 - numerator/denominator)

class OptimizationTuner:
    def __init__(self, results_dir: str = "optimization_results"):
        """Initialize tuner with results directory."""
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Results tracking
        self.results_df = pd.DataFrame(columns=[
            'opt_method', 'config', 'opt_time', 'train_r2', 'val_r2',
            'train_mse', 'val_mse', 'epoch', 'param_count'
        ])
        
        # Set up logger
        self.logger = logging.getLogger("OptimizationTuner")
        self.logger.setLevel(logging.INFO)
        
        # Load and preprocess data
        self._load_data()

    def _load_data(self):
        """Load and preprocess Jane Street market data."""
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

    def _train_and_evaluate(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                          opt_method: str, config: Dict[str, Any], num_epochs: int = 50
                          ) -> Tuple[Dict[str, List], float, float]:
        """Train model and track metrics."""
        metrics = defaultdict(list)
        best_val_r2 = float('-inf')
        best_val_mse = float('inf')
        
        # Use tqdm for epoch progress
        pbar = trange(num_epochs, desc=f"Training {opt_method}", leave=False)
        for epoch in pbar:
            # Training step
            model.train()
            optimizer.zero_grad()
            
            y_pred = model(self.x_train).squeeze(-1)
            
            # Use weighted MSE loss for Jane Street
            numerator = torch.sum(self.w_train * (self.y_train.squeeze(-1) - y_pred)**2)
            denominator = torch.sum(self.w_train)
            loss = numerator / (denominator + 1e-12)
            
            loss.backward()
            optimizer.step()
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                # Train metrics
                y_pred_train = model(self.x_train).squeeze(-1).cpu().numpy()
                train_mse = loss.item()  # Use the same weighted MSE
                train_r2 = weighted_r2(
                    self.y_train.squeeze(-1).cpu().numpy(),
                    y_pred_train,
                    self.w_train.cpu().numpy()
                )
                
                # Validation metrics
                y_pred_val = model(self.x_val).squeeze(-1).cpu().numpy()
                # Calculate weighted MSE for validation
                val_numerator = torch.sum(self.w_val * (self.y_val.squeeze(-1) - torch.tensor(y_pred_val, device=self.y_val.device))**2)
                val_denominator = torch.sum(self.w_val)
                val_mse = (val_numerator / (val_denominator + 1e-12)).item()
                
                val_r2 = weighted_r2(
                    self.y_val.squeeze(-1).cpu().numpy(),
                    y_pred_val,
                    self.w_val.cpu().numpy()
                )
            
            # Update progress bar
            pbar.set_postfix({
                'train_r2': f"{train_r2:.4f}",
                'val_r2': f"{val_r2:.4f}"
            })
            
            # Track peak performance
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
            if val_mse < best_val_mse:
                best_val_mse = val_mse
            
            # Store metrics
            metrics['epoch'].append(epoch)
            metrics['train_mse'].append(train_mse)
            metrics['val_mse'].append(val_mse)
            metrics['train_r2'].append(train_r2)
            metrics['val_r2'].append(val_r2)
            
            # Add to results DataFrame
            new_row = pd.DataFrame({
                'opt_method': [opt_method],
                'config': [str(config)],
                'opt_time': [config.get('opt_time', 0)],
                'train_r2': [train_r2],
                'val_r2': [val_r2],
                'train_mse': [train_mse],
                'val_mse': [val_mse],
                'epoch': [epoch],
                'param_count': [config.get('param_count', 0)]
            })
            self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
            
            if epoch % 10 == 0:
                print(f"[{opt_method}] Config={config} Epoch {epoch}/{num_epochs}, "
                      f"Train R²={train_r2:.4f}, Val R²={val_r2:.4f}")
        
        return metrics, best_val_r2, best_val_mse

    def compare_optimization_methods(self, param_grid: Dict[str, List[Any]], num_epochs: int = 50, 
                                  methods_to_run: Optional[List[str]] = None):
        """Compare different optimization methods with grid search."""
        print("\n=== Comparing KAN Optimization Methods ===")
        
        # Define optimization methods
        all_optimization_methods = [
            ('QUBO', lambda kan, x, y: kan.optimize(x, y)),
            ('IntegerProgramming', lambda kan, x, y: kan.optimize_integer_programming(x, y)),
            ('Evolutionary', lambda kan, x, y: kan.optimize_evolutionary(x, y)),
            ('GreedyHeuristic', lambda kan, x, y: kan.optimize_greedy_heuristic(x, y))
        ]
        
        # Filter methods if specified
        if methods_to_run:
            optimization_methods = [m for m in all_optimization_methods if m[0] in methods_to_run]
            if not optimization_methods:
                raise ValueError(f"No valid methods to run. Available methods: {[m[0] for m in all_optimization_methods]}")
        else:
            optimization_methods = all_optimization_methods
            
        print(f"Running optimization methods: {[m[0] for m in optimization_methods]}")
        
        # Track best configuration for each method
        best_configs = {}
        best_models = {}
        for method_name, _ in optimization_methods:
            best_configs[method_name] = {
                'r2': float('-inf'),
                'mse': float('inf'),
                'config': None,
                'time': 0
            }
            best_models[method_name] = None
        
        # Calculate total configurations to test
        total_configs = (
            len(param_grid['max_degree']) * 
            len(param_grid['hidden_size']) * 
            len(param_grid['hidden_degree']) * 
            len(param_grid['learning_rate']) * 
            len(optimization_methods)
        )
        
        # Main progress bar for overall progress
        main_pbar = tqdm(total=total_configs, desc="Overall Progress", position=0)
        config_counter = 0
        
        # Grid search over all methods
        for max_degree in param_grid['max_degree']:
            for hidden_size in param_grid['hidden_size']:
                for hidden_degree in param_grid['hidden_degree']:
                    for lr in param_grid['learning_rate']:
                        # Base configuration for all methods
                        base_config = {
                            'max_degree': max_degree,
                            'hidden_size': hidden_size,
                            'hidden_degree': hidden_degree,
                            'learning_rate': lr,
                        }
                        
                        print(f"\nTrying configuration: {base_config}")
                        
                        for method_name, optimize_fn in optimization_methods:
                            config_counter += 1
                            main_pbar.update(1)
                            main_pbar.set_description(f"Testing {method_name} with {base_config}")
                            
                            print(f"\n--- Testing {method_name} Optimization ---")
                            
                            # Create a new KAN model with this configuration
                            kan_config = FixedKANConfig(
                                network_shape=[self.input_dim, hidden_size, 1],
                                max_degree=max_degree,
                                complexity_weight=0.0,
                                trainable_coefficients=True,
                                skip_qubo_for_hidden=False,
                                default_hidden_degree=hidden_degree
                            )
                            
                            kan = FixedKAN(kan_config)
                            param_count = count_parameters(kan)
                            print(f"KAN parameter count: {param_count}")
                            
                            try:
                                # Measure optimization time
                                print(f"Running {method_name} optimization...")
                                start_time = time.time()
                                optimize_fn(kan, self.x_train, self.y_train)
                                opt_time = time.time() - start_time
                                print(f"{method_name} optimization completed in {opt_time:.2f} seconds")
                                
                                # Update config with measured values
                                config = base_config.copy()
                                config['opt_time'] = opt_time
                                config['param_count'] = param_count
                                
                                # Setup optimizer
                                params_to_train = []
                                for layer in kan.layers:
                                    params_to_train.extend([layer.combine_W, layer.combine_b])
                                    for neuron in layer.neurons:
                                        params_to_train.extend([neuron.w, neuron.b])
                                
                                optimizer = torch.optim.Adam(params_to_train, lr=lr)
                                
                                # Train and evaluate
                                metrics, best_r2, best_mse = self._train_and_evaluate(
                                    kan, optimizer, method_name, config, num_epochs
                                )
                                
                                # Update best config if this is better
                                if best_r2 > best_configs[method_name]['r2']:
                                    best_configs[method_name] = {
                                        'r2': best_r2,
                                        'mse': best_mse,
                                        'config': config,
                                        'time': opt_time
                                    }
                                    
                                    # No need to save each model, just keep track of the best one
                                    print(f"New best {method_name} model: R²={best_r2:.4f}")
                                    best_models[method_name] = {
                                        'model_state': kan.state_dict(),
                                        'kan_config': kan_config,
                                        'metrics': metrics,
                                        'best_r2': best_r2,
                                        'best_mse': best_mse,
                                        'opt_time': opt_time
                                    }
                                
                            except Exception as e:
                                print(f"Error with {method_name} optimization: {str(e)}")
                            
                            # Cleanup
                            del kan
                            if 'optimizer' in locals():
                                del optimizer
                            torch.cuda.empty_cache()
                            gc.collect()
        
        main_pbar.close()
        
        # Print best configs
        print("\n=== Best Configurations ===")
        for method_name, result in best_configs.items():
            print(f"\n{method_name}:")
            print(f"Best R²: {result['r2']:.4f}")
            print(f"Best MSE: {result['mse']:.6f}")
            print(f"Optimization Time: {result['time']:.2f} seconds")
            print(f"Configuration: {result['config']}")
            
            # Save only the best model for each method
            if best_models[method_name] is not None:
                torch.save(
                    best_models[method_name], 
                    f'{self.results_dir}/kan_{method_name.lower()}_best.pth'
                )
                print(f"Best {method_name} model saved to: {self.results_dir}/kan_{method_name.lower()}_best.pth")
        
        # Save results
        self.results_df.to_csv(f'{self.results_dir}/optimization_comparison.csv', index=False)
        print(f"Results saved to {self.results_dir}/optimization_comparison.csv")

    def plot_results(self):
        """Plot comparison of different optimization methods."""
        if len(self.results_df) == 0:
            print("No results to plot. Run comparison first.")
            return
        
        # 1. Plot metrics vs epoch for best configuration of each method
        for metric in ['val_r2', 'val_mse']:
            plt.figure(figsize=(12, 6))
            
            for method in self.results_df['opt_method'].unique():
                method_data = self.results_df[self.results_df['opt_method'] == method]
                # Get best config based on highest final val_r2
                best_configs = method_data.loc[method_data.groupby('config')['val_r2'].idxmax()]
                best_config = best_configs.loc[best_configs['val_r2'].idxmax()]['config']
                
                # Get data for best config
                best_data = method_data[method_data['config'] == best_config]
                
                label = f"{method} (Best)"
                plt.plot(best_data['epoch'], best_data[metric], label=label, linewidth=2)
            
            metric_name = "Validation R²" if metric == 'val_r2' else "Validation MSE"
            plt.title(f"{metric_name} vs Epoch for Different Optimization Methods")
            plt.xlabel("Epoch")
            plt.ylabel(metric_name)
            if metric == 'val_mse':
                plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()  # Ensure tight layout
            
            # Save plot
            plt.savefig(f'{self.results_dir}/{metric}_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
                       bbox_inches='tight')  # Use tight bounding box
            print(f"{metric_name} comparison plot saved.")
        
        # 2. Plot optimization time comparison
        plt.figure(figsize=(10, 6))
        
        # Get best config for each method
        method_times = []
        method_names = []
        
        for method in self.results_df['opt_method'].unique():
            method_data = self.results_df[self.results_df['opt_method'] == method]
            if len(method_data) > 0:
                # Get first occurrence since opt_time is the same for all rows of a run
                opt_time = method_data.iloc[0]['opt_time']
                method_times.append(opt_time)
                method_names.append(method)
        
        # Create bar chart
        bar_colors = ['blue', 'orange', 'green', 'red'][:len(method_names)]
        bars = plt.bar(method_names, method_times, color=bar_colors)
        
        # Add time labels on bars
        for bar, time_val in zip(bars, method_times):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f'{time_val:.2f}s',
                ha='center',
                fontweight='bold'
            )
        
        plt.title('Optimization Time Comparison')
        plt.xlabel('Optimization Method')
        plt.ylabel('Time (seconds)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()  # Ensure tight layout
        
        # Save plot
        plt.savefig(f'{self.results_dir}/optimization_times_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
                   bbox_inches='tight')  # Use tight bounding box
        print("Optimization time comparison plot saved.")
        
        # 3. Plot final performance comparison
        plt.figure(figsize=(10, 6))
        
        # Get best R² for each method
        method_r2s = []
        for method in method_names:
            method_data = self.results_df[self.results_df['opt_method'] == method]
            best_r2 = method_data['val_r2'].max()
            method_r2s.append(best_r2)
        
        # Create bar chart
        bars = plt.bar(method_names, method_r2s, color=bar_colors)
        
        # Add R² labels on bars
        for bar, r2_val in zip(bars, method_r2s):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{r2_val:.4f}',
                ha='center',
                fontweight='bold'
            )
        
        plt.title('Final Validation R² Comparison')
        plt.xlabel('Optimization Method')
        plt.ylabel('Validation R²')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()  # Ensure tight layout
        
        # Save plot
        plt.savefig(f'{self.results_dir}/final_r2_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
                   bbox_inches='tight')  # Use tight bounding box
        print("Final R² comparison plot saved.")

if __name__ == "__main__":
    # Initialize tuner
    tuner = OptimizationTuner()
    
    # Parameter grid
    param_grid = {
        'max_degree': [5, 7],
        'hidden_size': [16, 20],
        'hidden_degree': [3, 5],
        'learning_rate': [1e-3, 5e-4]
    }
    
    # Run comparison
    tuner.compare_optimization_methods(param_grid, num_epochs=30)
    
    # Plot results
    tuner.plot_results() 