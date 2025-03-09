import math
import logging
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
import itertools
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
from tqdm import tqdm

# Local imports
from CP_KAN import FixedKANConfig, FixedKAN
from data_pipeline_js_config import DataConfig
from data_pipeline import DataPipeline

class ModelTuner:
    """Base class for hyperparameter tuning."""
    
    def __init__(self, x_train, y_train, x_val, y_val, logger=None):
        """Initialize tuner with data."""
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.input_dim = x_train.shape[1]
        
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger
            
        # Create directories
        os.makedirs("tuning_results", exist_ok=True)
        os.makedirs("tuning_plots", exist_ok=True)
        
        # Results DataFrame
        self.results_df = pd.DataFrame()
        
        # Initialize timing stats
        self.timing_stats = defaultdict(float)
        
    def _compute_metric(self, y_true, y_pred, weights=None):
        """Compute performance metric (MSE or R² depending on dataset)."""
        if weights is None:
            # MSE for house sales
            return float(torch.mean((y_true - y_pred)**2))
        else:
            # Weighted R² for Jane Street
            numerator = torch.sum(weights * (y_true - y_pred)**2)
            denominator = torch.sum(weights * y_true**2)
            return float(1.0 - numerator/(denominator + 1e-12))
    
    def save_results(self, filename: str):
        """Save results to CSV."""
        path = os.path.join("tuning_results", filename)
        self.results_df.to_csv(path, index=False)
        self.logger.info(f"Results saved to {path}")
        
        # Save timing stats
        timing_path = os.path.join("tuning_results", f"timing_{filename}")
        pd.DataFrame([self.timing_stats]).to_csv(timing_path, index=False)
        self.logger.info(f"Timing stats saved to {timing_path}")
    
    def plot_results(self, x_param: str, y_metric: str, filename: str):
        """Create plot comparing parameter values vs metric."""
        plt.figure(figsize=(12, 7))
        
        # Group by parameter and compute mean/std of metric
        grouped = self.results_df.groupby(x_param)[y_metric]
        means = grouped.mean()
        stds = grouped.std()
        
        # Plot with error bars and individual points
        plt.errorbar(means.index, means.values, yerr=stds.values,
                    marker='o', linestyle='-', capsize=5, label='Mean ± Std', color='blue', linewidth=2)
        
        # Add scatter points for all results
        for param_val in self.results_df[x_param].unique():
            points = self.results_df[self.results_df[x_param] == param_val][y_metric]
            plt.scatter([param_val] * len(points), points, alpha=0.3, color='gray', s=30)
        
        # Formatting
        plt.xlabel(x_param.replace('_', ' ').title())
        plt.ylabel(y_metric.replace('_', ' ').title())
        if 'mse' in y_metric.lower():
            plt.yscale('log')  # Log scale for MSE
            plt.title(f"MSE vs {x_param.replace('_', ' ').title()}")
        else:
            plt.title(f"R² Score vs {x_param.replace('_', ' ').title()}")
        
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save plot with high DPI for paper quality
        path = os.path.join("tuning_plots", filename)
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()
        self.logger.info(f"Plot saved to {path}")
        
    def plot_comparison(self, other_tuner, x_param: str, y_metric: str, filename: str):
        """Create comparison plot between this tuner and another tuner."""
        plt.figure(figsize=(12, 7))
        
        # Plot this tuner's results
        grouped = self.results_df.groupby(x_param)[y_metric]
        means = grouped.mean()
        stds = grouped.std()
        plt.errorbar(means.index, means.values, yerr=stds.values,
                    marker='o', linestyle='-', capsize=5, 
                    label=f'{self.__class__.__name__}', color='blue', linewidth=2)
        
        # Plot other tuner's results
        grouped = other_tuner.results_df.groupby(x_param)[y_metric]
        means = grouped.mean()
        stds = grouped.std()
        plt.errorbar(means.index, means.values, yerr=stds.values,
                    marker='s', linestyle='-', capsize=5,
                    label=f'{other_tuner.__class__.__name__}', color='red', linewidth=2)
        
        # Formatting
        plt.xlabel(x_param.replace('_', ' ').title())
        plt.ylabel(y_metric.replace('_', ' ').title())
        if 'mse' in y_metric.lower():
            plt.yscale('log')
            plt.title(f"MSE Comparison")
        else:
            plt.title(f"R² Score Comparison")
        
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save plot with high DPI for paper quality
        path = os.path.join("tuning_plots", filename)
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()
        self.logger.info(f"Comparison plot saved to {path}")

def plot_timing_comparison(kan_tuner: KANTuner, mlp_tuner: MLPTuner, dataset_name: str):
    """Create timing comparison plots between KAN and MLP."""
    plt.figure(figsize=(15, 6))
    
    # Prepare timing data
    kan_times = {
        'Total Training': kan_tuner.results_df['total_time'].mean(),
        'Avg Epoch': kan_tuner.results_df['avg_epoch_time'].mean(),
        'Avg Fold': kan_tuner.results_df['avg_fold_time'].mean(),
        'QUBO': kan_tuner.results_df['qubo_time'].mean() if 'qubo_time' in kan_tuner.results_df else 0
    }
    
    mlp_times = {
        'Total Training': mlp_tuner.results_df['total_time'].mean(),
        'Avg Epoch': mlp_tuner.results_df['avg_epoch_time'].mean(),
        'Avg Fold': mlp_tuner.results_df['avg_fold_time'].mean(),
        'QUBO': 0  # MLP doesn't use QUBO
    }
    
    # Create bar plot
    metrics = list(kan_times.keys())
    x = np.arange(len(metrics))
    width = 0.35
    
    kan_bars = plt.bar(x - width/2, list(kan_times.values()), width, label='KAN', color='blue', alpha=0.7)
    mlp_bars = plt.bar(x + width/2, list(mlp_times.values()), width, label='MLP', color='red', alpha=0.7)
    
    plt.yscale('log')
    plt.ylabel('Time (seconds)')
    plt.title(f'Timing Comparison on {dataset_name} Dataset')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom')
    
    add_value_labels(kan_bars)
    add_value_labels(mlp_bars)
    
    # Save plot
    path = os.path.join("tuning_plots", f"{dataset_name.lower()}_timing_comparison.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Timing comparison plot saved to: {path}")

class KANTuner(ModelTuner):
    """Hyperparameter tuner for KAN models."""
    
    def __init__(self, x_train, y_train, x_val, y_val, weights_train=None, weights_val=None, logger=None):
        super().__init__(x_train, y_train, x_val, y_val, logger)
        self.weights_train = weights_train
        self.weights_val = weights_val
        # Initialize best metric and model
        self.best_metric = float('-inf') if weights_train is not None else float('inf')
        self.best_model = None
        
    def grid_search(self, param_grid: Dict[str, List[Any]], 
                   n_folds: int = 3, epochs: int = 50, patience: int = 10,
                   save_best: bool = True, save_prefix: str = ""):
        """Perform grid search over KAN hyperparameters."""
        # Initialize timing dictionary
        total_start_time = time.time()
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        # Results storage
        results = []
        
        # Cross validation split
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for params in tqdm(combinations, desc="Parameter combinations"):
            param_dict = dict(zip(param_names, params))
            
            # Track metrics across folds
            fold_metrics = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.x_train)):
                fold_start_time = time.time()
                
                # Split data
                X_train_fold = self.x_train[train_idx]
                y_train_fold = self.y_train[train_idx]
                X_val_fold = self.x_train[val_idx]
                y_val_fold = self.y_train[val_idx]
                
                if self.weights_train is not None:
                    w_train_fold = self.weights_train[train_idx]
                    w_val_fold = self.weights_train[val_idx]
                else:
                    w_train_fold = None
                    w_val_fold = None
                
                # Create and train KAN
                network_shape = param_dict.get('network_shape', [self.input_dim, 20, 1])
                if isinstance(network_shape, int):
                    network_shape = [self.input_dim, network_shape, 1]
                
                kan_config = FixedKANConfig(
                    network_shape=network_shape,
                    max_degree=param_dict['max_degree'],
                    complexity_weight=param_dict['complexity_weight'],
                    trainable_coefficients=True,
                    skip_qubo_for_hidden=False,
                    default_hidden_degree=param_dict['default_hidden_degree']
                )
                
                kan = FixedKAN(kan_config)
                
                # Optimize QUBO with timing
                qubo_start = time.time()
                kan.optimize(X_train_fold, y_train_fold.unsqueeze(-1))
                self.timing_stats['qubo_time'] += time.time() - qubo_start
                
                # Training loop
                optimizer = torch.optim.Adam(kan.parameters(), lr=param_dict['learning_rate'])
                best_val_metric = float('-inf') if self.weights_train is not None else float('inf')
                patience_counter = 0
                
                for epoch in range(epochs):
                    epoch_start = time.time()
                    
                    # Train step
                    optimizer.zero_grad()
                    y_pred = kan(X_train_fold).squeeze(-1)
                    
                    if w_train_fold is not None:
                        # Weighted MSE for Jane Street
                        numerator = torch.sum(w_train_fold * (y_train_fold - y_pred)**2)
                        denominator = torch.sum(w_train_fold)
                        loss = numerator / (denominator + 1e-12)
                    else:
                        # Regular MSE for house sales
                        loss = torch.mean((y_train_fold - y_pred)**2)
                    
                    loss.backward()
                    optimizer.step()
                    
                    # Validation
                    with torch.no_grad():
                        val_pred = kan(X_val_fold).squeeze(-1)
                        val_metric = self._compute_metric(
                            y_val_fold, val_pred, w_val_fold
                        )
                        
                        # Early stopping check
                        if self.weights_train is not None:
                            improved = val_metric > best_val_metric
                        else:
                            improved = val_metric < best_val_metric
                            
                        if improved:
                            best_val_metric = val_metric
                            patience_counter = 0
                        else:
                            patience_counter += 1
                    
                    self.timing_stats['epoch_time'] += time.time() - epoch_start
                            
                    if patience_counter >= patience:
                        break
                
                fold_metrics.append(best_val_metric)
                self.timing_stats['fold_time'] += time.time() - fold_start_time
                
                # Save best model if this is the best performance so far
                if save_best and (
                    (self.weights_train is not None and best_val_metric > self.best_metric) or
                    (self.weights_train is None and best_val_metric < self.best_metric)
                ):
                    self.best_metric = best_val_metric
                    self.best_model = kan
                    if save_prefix:
                        metric_str = f"{best_val_metric:.6f}".replace(".", "_")
                        save_path = f"tuning_results/{save_prefix}_best_kan_{metric_str}.pth"
                        kan.save_model(save_path)
                        print(f"\nNew best KAN model saved: {save_path}")
            
            # Record results with timing information
            result = {
                **param_dict,
                'mean_val_metric': np.mean(fold_metrics),
                'std_val_metric': np.std(fold_metrics),
                'total_time': time.time() - total_start_time,
                'avg_epoch_time': self.timing_stats['epoch_time'] / (len(fold_metrics) * epochs),
                'qubo_time': self.timing_stats['qubo_time'] / len(fold_metrics),
                'avg_fold_time': self.timing_stats['fold_time'] / len(fold_metrics)
            }
            results.append(result)
            
            # Update results DataFrame
            self.results_df = pd.DataFrame(results)
            
        return self.results_df, self.best_model

class MLPTuner(ModelTuner):
    """Hyperparameter tuner for MLP models."""
    
    def __init__(self, x_train, y_train, x_val, y_val, weights_train=None, weights_val=None, logger=None):
        super().__init__(x_train, y_train, x_val, y_val, logger)
        self.weights_train = weights_train
        self.weights_val = weights_val
        # Initialize best metric and model
        self.best_metric = float('-inf') if weights_train is not None else float('inf')
        self.best_model = None
    
    def _build_mlp(self, hidden_sizes: List[int], dropout_rate: float = 0.1) -> nn.Module:
        """Build MLP with specified architecture."""
        layers = []
        curr_dim = self.input_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(curr_dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            curr_dim = hidden_size
            
        layers.append(nn.Linear(curr_dim, 1))
        return nn.Sequential(*layers)
    
    def grid_search(self, param_grid: Dict[str, List[Any]], 
                   n_folds: int = 3, epochs: int = 50, patience: int = 10,
                   save_best: bool = True, save_prefix: str = ""):
        """Perform grid search over MLP hyperparameters."""
        # Initialize timing dictionary
        total_start_time = time.time()
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        # Results storage
        results = []
        
        # Cross validation split
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for params in tqdm(combinations, desc="Parameter combinations"):
            param_dict = dict(zip(param_names, params))
            
            # Track metrics across folds
            fold_metrics = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.x_train)):
                fold_start_time = time.time()
                
                # Split data
                X_train_fold = self.x_train[train_idx]
                y_train_fold = self.y_train[train_idx]
                X_val_fold = self.x_train[val_idx]
                y_val_fold = self.y_train[val_idx]
                
                if self.weights_train is not None:
                    w_train_fold = self.weights_train[train_idx]
                    w_val_fold = self.weights_train[val_idx]
                else:
                    w_train_fold = None
                    w_val_fold = None
                
                # Create MLP
                hidden_sizes = [param_dict['hidden_size']] * param_dict['depth']
                mlp = self._build_mlp(hidden_sizes, param_dict['dropout_rate'])
                
                # Training setup
                optimizer = torch.optim.AdamW(
                    mlp.parameters(), 
                    lr=param_dict['learning_rate'],
                    weight_decay=param_dict['weight_decay']
                )
                
                best_val_metric = float('-inf') if self.weights_train is not None else float('inf')
                patience_counter = 0
                batch_size = param_dict['batch_size']
                
                # Training loop
                for epoch in range(epochs):
                    epoch_start = time.time()
                    mlp.train()
                    
                    # Mini-batch training
                    n_batches = math.ceil(len(X_train_fold) / batch_size)
                    for i in range(n_batches):
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, len(X_train_fold))
                        
                        x_batch = X_train_fold[start_idx:end_idx]
                        y_batch = y_train_fold[start_idx:end_idx]
                        
                        if w_train_fold is not None:
                            w_batch = w_train_fold[start_idx:end_idx]
                        
                        optimizer.zero_grad()
                        y_pred = mlp(x_batch).squeeze(-1)
                        
                        if w_train_fold is not None:
                            # Weighted MSE for Jane Street
                            numerator = torch.sum(w_batch * (y_batch - y_pred)**2)
                            denominator = torch.sum(w_batch)
                            loss = numerator / (denominator + 1e-12)
                        else:
                            # Regular MSE for house sales
                            loss = torch.mean((y_batch - y_pred)**2)
                        
                        loss.backward()
                        optimizer.step()
                    
                    # Validation
                    mlp.eval()
                    with torch.no_grad():
                        val_pred = mlp(X_val_fold).squeeze(-1)
                        val_metric = self._compute_metric(
                            y_val_fold, val_pred, w_val_fold
                        )
                        
                        # Early stopping check
                        if self.weights_train is not None:
                            improved = val_metric > best_val_metric
                        else:
                            improved = val_metric < best_val_metric
                            
                        if improved:
                            best_val_metric = val_metric
                            patience_counter = 0
                        else:
                            patience_counter += 1
                    
                    self.timing_stats['epoch_time'] += time.time() - epoch_start
                            
                    if patience_counter >= patience:
                        break
                
                fold_metrics.append(best_val_metric)
                self.timing_stats['fold_time'] += time.time() - fold_start_time
                
                # Save best model if this is the best performance so far
                if save_best and (
                    (self.weights_train is not None and best_val_metric > self.best_metric) or
                    (self.weights_train is None and best_val_metric < self.best_metric)
                ):
                    self.best_metric = best_val_metric
                    self.best_model = mlp.state_dict()
                    if save_prefix:
                        metric_str = f"{best_val_metric:.6f}".replace(".", "_")
                        save_path = f"tuning_results/{save_prefix}_best_mlp_{metric_str}.pt"
                        torch.save(mlp.state_dict(), save_path)
                        print(f"\nNew best MLP model saved: {save_path}")
            
            # Record results with timing information
            result = {
                **param_dict,
                'mean_val_metric': np.mean(fold_metrics),
                'std_val_metric': np.std(fold_metrics),
                'total_time': time.time() - total_start_time,
                'avg_epoch_time': self.timing_stats['epoch_time'] / (len(fold_metrics) * epochs),
                'avg_fold_time': self.timing_stats['fold_time'] / len(fold_metrics)
            }
            results.append(result)
            
            # Update results DataFrame
            self.results_df = pd.DataFrame(results)
            
        return self.results_df, self.best_model

def run_house_sales_tuning():
    """Run hyperparameter tuning on house sales dataset."""
    print("\nStarting House Sales Dataset Tuning")
    print("===================================")
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
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
    y = np.log1p(y)
    
    # Normalize features
    X = StandardScaler().fit_transform(X)
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to torch
    x_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    
    print("\nRunning KAN hyperparameter tuning...")
    # KAN tuning with expanded parameter grid
    kan_tuner = KANTuner(x_train, y_train, x_val, y_val)
    kan_param_grid = {
        'max_degree': [5, 7, 9, 11],
        'network_shape': [20, 32, 64],
        'default_hidden_degree': [3, 5, 7, 9],
        'complexity_weight': [0.0, 0.001, 0.01],
        'learning_rate': [1e-4, 5e-4, 1e-3]
    }
    kan_results, best_kan = kan_tuner.grid_search(kan_param_grid, save_prefix="house_sales")
    kan_tuner.save_results('house_sales_kan_tuning.csv')
    
    # Generate KAN-specific plots
    kan_tuner.plot_results('max_degree', 'mean_val_metric', 'house_sales_kan_degree.png')
    kan_tuner.plot_results('network_shape', 'mean_val_metric', 'house_sales_kan_width.png')
    kan_tuner.plot_results('default_hidden_degree', 'mean_val_metric', 'house_sales_kan_hidden_degree.png')
    
    print("\nRunning MLP hyperparameter tuning...")
    # MLP tuning with expanded parameter grid
    mlp_tuner = MLPTuner(x_train, y_train, x_val, y_val)
    mlp_param_grid = {
        'depth': [2, 3, 4, 5],
        'hidden_size': [32, 64, 128],
        'dropout_rate': [0.1, 0.2],
        'learning_rate': [1e-4, 1e-3],
        'weight_decay': [0.0, 0.001],
        'batch_size': [64, 128, 256]
    }
    mlp_results, best_mlp = mlp_tuner.grid_search(mlp_param_grid, save_prefix="house_sales")
    mlp_tuner.save_results('house_sales_mlp_tuning.csv')
    
    # Generate MLP-specific plots
    mlp_tuner.plot_results('depth', 'mean_val_metric', 'house_sales_mlp_depth.png')
    mlp_tuner.plot_results('hidden_size', 'mean_val_metric', 'house_sales_mlp_width.png')
    
    # Generate comparison plots
    kan_tuner.plot_comparison(mlp_tuner, 'network_shape', 'mean_val_metric', 'house_sales_width_comparison.png')
    plot_timing_comparison(kan_tuner, mlp_tuner, "House Sales")
    
    # Print best configurations
    print("\nBest KAN configuration:")
    best_kan_idx = kan_results['mean_val_metric'].argmin()
    best_kan = kan_results.iloc[best_kan_idx]
    for param, value in best_kan.items():
        if param not in ['mean_val_metric', 'std_val_metric', 'total_time', 'avg_epoch_time', 'qubo_time', 'avg_fold_time']:
            print(f"{param}: {value}")
    print(f"Best MSE: {best_kan['mean_val_metric']:.6f} ± {best_kan['std_val_metric']:.6f}")
    print(f"Training Time: {best_kan['total_time']:.2f}s")
    print(f"Avg Epoch Time: {best_kan['avg_epoch_time']:.4f}s")
    print(f"Avg QUBO Time: {best_kan['qubo_time']:.4f}s")
    print(f"Avg Fold Time: {best_kan['avg_fold_time']:.2f}s")
    
    print("\nBest MLP configuration:")
    best_mlp_idx = mlp_results['mean_val_metric'].argmin()
    best_mlp = mlp_results.iloc[best_mlp_idx]
    for param, value in best_mlp.items():
        if param not in ['mean_val_metric', 'std_val_metric', 'total_time', 'avg_epoch_time', 'avg_fold_time']:
            print(f"{param}: {value}")
    print(f"Best MSE: {best_mlp['mean_val_metric']:.6f} ± {best_mlp['std_val_metric']:.6f}")
    print(f"Training Time: {best_mlp['total_time']:.2f}s")
    print(f"Avg Epoch Time: {best_mlp['avg_epoch_time']:.4f}s")
    print(f"Avg Fold Time: {best_mlp['avg_fold_time']:.2f}s")

def run_jane_street_tuning():
    """Run hyperparameter tuning on Jane Street dataset."""
    print("\nStarting Jane Street Dataset Tuning")
    print("===================================")
    
    # Load data using DataPipeline
    print("\nLoading and preprocessing data...")
    data_cfg = DataConfig(
        data_path="~/Interning/Kaggle/jane_street_kaggle/jane-street-real-time-market-data-forecasting/train.parquet/",
        n_rows=200000,
        train_ratio=0.7,
        feature_cols=[f'feature_{i:02d}' for i in range(79)],
        target_col="responder_6",
        weight_col="weight",
        date_col="date_id"
    )
    
    pipeline = DataPipeline(data_cfg)
    train_df, train_target, train_weight, val_df, val_target, val_weight = pipeline.load_and_preprocess_data()
    
    # Convert to torch
    x_train = torch.tensor(train_df.to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(train_target.to_numpy(), dtype=torch.float
