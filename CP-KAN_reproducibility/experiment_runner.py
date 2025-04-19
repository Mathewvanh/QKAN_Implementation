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
import torch.nn.functional as F # Added for one-hot encoding
import pandas as pd
from collections import defaultdict
from tqdm import tqdm, trange
import logging

# Dataset specific imports
from data_pipeline import DataPipeline, DataConfig # For Jane Street & its config
from sklearn.datasets import fetch_california_housing # For House Prices
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torchvision # For MNIST/CIFAR
import torchvision.transforms as transforms
from datasets import load_dataset # Added for Hugging Face datasets

# Local imports
from CP_KAN import FixedKANConfig, FixedKAN

def count_parameters(module: nn.Module) -> int:
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

# === Metric Functions ===
def weighted_r2(y_true: torch.Tensor, y_pred: torch.Tensor, w: torch.Tensor) -> float:
    """Compute weighted R² score using the Jane Street competition formula (on device)."""
    w = w.squeeze()
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    numerator = torch.sum(w * (y_true - y_pred)**2)
    denominator = torch.sum(w * (y_true**2))
    if denominator.abs() < 1e-12:
        return 0.0 # Should be undefined or NaN? JS formula implies 0.
    r2 = 1.0 - (numerator / denominator)
    # Handle potential edge cases where prediction is much worse than mean -> large negative R2
    return float(r2.item())

def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Compute standard R² score (on device)."""
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    ss_res = torch.sum((y_true - y_pred)**2)
    ss_tot = torch.sum((y_true - y_true.mean())**2)
    if ss_tot.abs() < 1e-12:
        # If total variance is zero, R2 is undefined or 1 if residual variance is also zero.
        return 1.0 if ss_res.abs() < 1e-12 else 0.0
    r2 = 1.0 - (ss_res / ss_tot)
    return float(r2.item())

def accuracy(y_true: torch.Tensor, y_pred_logits: torch.Tensor) -> float:
    """Compute accuracy for classification (on device)."""
    y_true = y_true.squeeze()
    preds = torch.argmax(y_pred_logits, dim=1)
    correct = (preds == y_true).sum().item()
    return float(correct / len(y_true))

# === Experiment Runner Class ===
class ExperimentRunner:
    def __init__(self, config: Dict):
        """Initialize tuner with general experiment configuration."""
        self.config = config
        self.results_dir = config['results_dir']
        self.dataset_config = config['dataset']
        self.dataset_name = self.dataset_config['name']
        self.task_type = self._determine_task_type()
        self.primary_metric = self._determine_primary_metric()
        self.higher_is_better = self.primary_metric in ['r2', 'accuracy']

        os.makedirs(self.results_dir, exist_ok=True)
        
        self.results_df = pd.DataFrame()
        
        self.logger = logging.getLogger("ExperimentRunner")
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            self.logger.addHandler(logging.StreamHandler())

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Primary metric: {self.primary_metric} (Higher is better: {self.higher_is_better})")

        self._load_data()
        self._prepare_optimize_data() # Ensure this call exists

    def _determine_task_type(self) -> str:
        if 'task_type' in self.dataset_config:
            tt = self.dataset_config['task_type'].lower()
            if tt in ['regression', 'classification']: return tt
            else: self.logger.warning(f"Invalid task_type '{tt}'. Inferring...")
        name = self.dataset_name.lower()
        if any(n in name for n in ['jane_street', 'house', 'california']):
             self.logger.info("Inferred task type: regression"); return 'regression'
        elif any(n in name for n in ['mnist', 'cifar', 'forest', 'covertype']):
             self.logger.info("Inferred task type: classification"); return 'classification'
        else: raise ValueError(f"Could not determine task type for '{self.dataset_name}'. Specify in config.")

    def _determine_primary_metric(self) -> str:
        if 'primary_metric' in self.dataset_config:
            metric = self.dataset_config['primary_metric'].lower()
            valid_regression = self.task_type == 'regression' and metric in ['r2', 'mse']
            valid_classification = self.task_type == 'classification' and metric == 'accuracy'
            if valid_regression or valid_classification: return metric
            else: self.logger.warning(f"Metric '{metric}' invalid for task '{self.task_type}'. Using default.")
        return 'accuracy' if self.task_type == 'classification' else 'r2'

    # --- Data Loading --- 
    def _load_data(self):
        name = self.dataset_name.lower()
        self.logger.info(f"Loading dataset: {name}")
        if 'house' in name:
            self._load_house_sales()
        elif 'mnist' in name:
            self._load_mnist()
        elif 'cifar' in name:
            self._load_cifar10()
        elif 'forest' in name or 'covertype' in name:
            self._load_forest_cover()
        elif 'jane' in name:
            self._load_jane_street()
        else:
            raise ValueError(f"Unsupported dataset: {name}")
        self.logger.info(f"Loaded dataset shapes: Train X: {self.x_train.shape}, Train y: {self.y_train.shape}, Val X: {self.x_val.shape}, Val y: {self.y_val.shape}")
        self.input_dim = self.x_train.shape[1]
        self.output_dim = len(torch.unique(self.y_train)) if self.task_type == 'classification' else (self.y_train.shape[1] if self.y_train.dim() > 1 else 1)
        self.logger.info(f"Input Dim: {self.input_dim}, Output Dim: {self.output_dim}")

    def _load_jane_street(self):
        """Load Jane Street data using its specific pipeline."""
        # Filter the dataset_config to only include keys expected by DataConfig
        expected_keys = DataConfig.__annotations__.keys()
        filtered_config = {k: v for k, v in self.dataset_config.items() if k in expected_keys}
        
        if 'data_path' not in filtered_config:
             self.logger.error("Missing 'data_path' in dataset config for Jane Street.")
             raise ValueError("Missing 'data_path' for Jane Street")
             
        # Expand 'auto' features *after* filtering and *before* creating DataConfig
        if filtered_config.get('feature_cols') == 'auto':
            self.logger.info("Auto-generating Jane Street feature columns inside runner.")
            num_features = 79 # Assuming 79 features based on original config
            filtered_config['feature_cols'] = [f'feature_{i:02d}' for i in range(num_features)]
            self.logger.debug(f"DEBUG: feature_cols in filtered_config AFTER expansion: {filtered_config['feature_cols'][:5]}... (Type: {type(filtered_config['feature_cols'])})")
        else:
             self.logger.debug(f"DEBUG: feature_cols in filtered_config (not 'auto'): {filtered_config.get('feature_cols')}")
             
        js_data_cfg = DataConfig.from_dict(filtered_config) 
        self.logger.debug(f"DEBUG: feature_cols in js_data_cfg AFTER DataConfig init: {js_data_cfg.feature_cols[:5]}... (Type: {type(js_data_cfg.feature_cols)})")
        
        pipeline = DataPipeline(js_data_cfg, self.logger)
        train_df, train_target, train_weight, val_df, val_target, val_weight = pipeline.load_and_preprocess_data()

        self.x_train = torch.tensor(train_df.to_numpy(), dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(train_target.to_numpy(), dtype=torch.float32).squeeze(-1).unsqueeze(-1).to(self.device)
        self.w_train = torch.tensor(train_weight.to_numpy(), dtype=torch.float32).squeeze(-1).to(self.device)

        self.x_val = torch.tensor(val_df.to_numpy(), dtype=torch.float32).to(self.device)
        self.y_val = torch.tensor(val_target.to_numpy(), dtype=torch.float32).squeeze(-1).unsqueeze(-1).to(self.device)
        self.w_val = torch.tensor(val_weight.to_numpy(), dtype=torch.float32).squeeze(-1).to(self.device)
        self.use_weights = True

    def _load_house_sales(self):
        """Load INRIA House Sales dataset using Hugging Face datasets library."""
        try:
            from datasets import load_dataset
        except ImportError:
            self.logger.error("Hugging Face `datasets` library not installed. Please install it: pip install datasets")
            raise
            
        dataset_name = "inria-soda/tabular-benchmark"
        data_file = "reg_num/house_sales.csv"
        self.logger.info(f"Loading dataset '{dataset_name}' with data file '{data_file}'")
        
        try:
            dataset = load_dataset(dataset_name, data_files=data_file, split="train")
            df = pd.DataFrame(dataset)
        except Exception as e:
            self.logger.error(f"Failed to load dataset {dataset_name}/{data_file}: {e}")
            raise
            
        # Determine label column (assuming last column if 'target' not present)
        label_col = "target" if "target" in df.columns else df.columns[-1]
        self.logger.info(f"Using label column: '{label_col}'")
        
        # Get features and target
        y = df[label_col].values.astype(np.float32)
        X = df.drop(columns=[label_col]).values.astype(np.float32)
        
        # Log transform target (same as test_house_sales_degradation_v2.py)
        self.logger.info("Applying log1p transformation to the target variable.")
        y = np.log1p(y) 
        y = y.reshape(-1, 1) # Ensure y is [n_samples, 1]

        # Split data (using params from config if available)
        test_size = self.dataset_config.get('test_split', 0.2)
        val_size_ratio = self.dataset_config.get('val_from_train_split', 0.2)
        random_state = self.config.get('random_seed', 42)
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        # Split train_val into actual train and validation sets
        # Calculate val_size relative to the original dataset size for correct proportion
        val_size_abs = int(val_size_ratio * len(X))
        train_size_abs = len(X_train_val) - val_size_abs
        if train_size_abs <= 0 or val_size_abs <= 0:
             self.logger.warning(f"Train/Val split resulted in non-positive size (Train: {train_size_abs}, Val: {val_size_abs}). Adjusting split.")
             # Fallback to simple relative split of X_train_val
             X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=random_state) # e.g., 25% of train_val -> val
        else:
             X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, train_size=train_size_abs, test_size=val_size_abs, random_state=random_state)

        self.logger.info(f"Data split: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")

        # Scale features (StandardScaler)
        self.logger.info("Applying StandardScaler to features.")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        # X_test = scaler.transform(X_test) # Test set not used in runner currently

        # Target scaling (optional, often helpful for regression stability)
        # Keep consistent with test_house_sales_degradation_v2.py which didn't scale target
        # target_scaler = StandardScaler()
        # y_train = target_scaler.fit_transform(y_train)
        # y_val = target_scaler.transform(y_val)
        # self.target_scaler = target_scaler
        self.logger.info("Target variable not scaled (only log1p transformed).")

        # Convert to tensors
        self.x_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        self.x_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        self.y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        self.use_weights = False # No weights for this dataset
        self.scaler = scaler # Store scaler
        self.target_scaler = None # No target scaling
        
    def _load_mnist(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.Lambda(lambda x: torch.flatten(x))])
        dp = self.dataset_config.get('data_path', './data'); rs = self.config.get('random_seed', 42)
        train_ds = torchvision.datasets.MNIST(root=dp, train=True, download=True, transform=transform)
        val_ds = torchvision.datasets.MNIST(root=dp, train=False, download=True, transform=transform) # Use test set as validation
        # Create full tensors (adjust if using DataLoaders)
        self.x_train = torch.stack([data for data, target in train_ds], dim=0).to(self.device)
        self.y_train = torch.tensor([target for data, target in train_ds], dtype=torch.long).to(self.device)
        self.x_val = torch.stack([data for data, target in val_ds], dim=0).to(self.device)
        self.y_val = torch.tensor([target for data, target in val_ds], dtype=torch.long).to(self.device)
        self.use_weights = False
        max_optimize_samples = 5000
        if len(self.x_train) > max_optimize_samples:
             self.logger.warning(f"MNIST train set ({len(self.x_train)}) exceeds max samples ({max_optimize_samples}) for optimize step. Subsampling.")
             indices = torch.randperm(len(self.x_train))[:max_optimize_samples]
             self.x_optimize = self.x_train[indices]
             # Recompute output dim based on subset if necessary, though usually same for MNIST
             self.output_dim = len(torch.unique(self.y_train[indices]))
             # Corrected: Use torch.nn.functional.one_hot
             self.y_optimize_onehot = torch.nn.functional.one_hot(self.y_train[indices], num_classes=self.output_dim).float()
        else:
             self.x_optimize = self.x_train
             # Corrected: Use torch.nn.functional.one_hot
             self.y_optimize_onehot = torch.nn.functional.one_hot(self.y_train, num_classes=self.output_dim).float()

    def _load_cifar10(self):
        """Load CIFAR-10 dataset."""
        self.logger.info("Loading CIFAR-10 dataset...")
        # CIFAR-10 normalization values (commonly used)
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            transforms.Lambda(lambda x: torch.flatten(x))
        ])

        data_path = self.dataset_config.get('data_path', './data')
        try:
            train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
            val_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
        except Exception as e:
            self.logger.error(f"Failed to download/load CIFAR-10: {e}. Check network connection or data_path: {data_path}")
            raise

        # Use the standard train/test split provided by torchvision for train/val
        self.x_train = torch.stack([data for data, target in train_dataset], dim=0).to(self.device)
        self.y_train = torch.tensor([target for data, target in train_dataset], dtype=torch.long).to(self.device)
        self.x_val = torch.stack([data for data, target in val_dataset], dim=0).to(self.device)
        self.y_val = torch.tensor([target for data, target in val_dataset], dtype=torch.long).to(self.device)
        self.use_weights = False
        
        # Prepare data for KAN optimization step (subsampling if needed)
        max_optimize_samples = 5000 # Limit to avoid memory issues with one-hot vectors
        if len(self.x_train) > max_optimize_samples:
             self.logger.warning(f"CIFAR-10 training set ({len(self.x_train)}) exceeds max samples ({max_optimize_samples}) for optimize step. Subsampling.")
             indices = torch.randperm(len(self.x_train))[:max_optimize_samples]
             self.x_optimize = self.x_train[indices]
             self.output_dim = len(torch.unique(self.y_train[indices])) # Recalculate output dim based on subset
             self.y_optimize_onehot = torch.nn.functional.one_hot(self.y_train[indices], num_classes=self.output_dim).float()
        else:
             self.x_optimize = self.x_train
             self.output_dim = len(torch.unique(self.y_train)) # Use full train set
             self.y_optimize_onehot = torch.nn.functional.one_hot(self.y_train, num_classes=self.output_dim).float()

    def _load_forest_cover(self):
        """Load Forest Cover Type dataset (NUMERIC version) from INRIA SODA benchmark."""
        self.logger.info("Loading Forest Cover Type dataset (numeric version)...")
        try:
            from datasets import load_dataset
        except ImportError:
            self.logger.error("Hugging Face `datasets` library not installed."); raise

        dataset_name = "inria-soda/tabular-benchmark"
        data_file = "clf_num/covertype.csv" # Changed to NUMERIC version
        self.logger.info(f"Loading dataset '{dataset_name}' with data file '{data_file}'")
        
        try:
            dataset = load_dataset(dataset_name, data_files=data_file, split="train")
            df = pd.DataFrame(dataset)
        except Exception as e:
            self.logger.error(f"Failed to load dataset {dataset_name}/{data_file}: {e}")
            raise

        label_col = "target" if "target" in df.columns else df.columns[-1]
        self.logger.info(f"Using label column: '{label_col}'")
        
        y = df[label_col].values.astype(np.int64) # Use int64 for labels
        X = df.drop(columns=[label_col]).values.astype(np.float32)
        
        # Check label range (removed explicit shift y = y - 1)
        min_label, max_label = y.min(), y.max()
        self.logger.info(f"Found label range: min={min_label}, max={max_label}")
        if min_label != 0:
             self.logger.warning(f"Minimum label is {min_label}. CrossEntropyLoss expects 0-based labels.")
        # Assuming num_classes = max_label + 1 if min is 0

        # Split data (using validation_split from config)
        val_size = self.dataset_config.get('validation_split', 0.2) # Use validation_split key
        random_state = self.config.get('random_seed', 42)
        
        # Stratified split is important for classification
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=random_state, stratify=y
        )
        self.logger.info(f"Data split: Train={len(X_train)}, Validation={len(X_val)}")

        # Scale features
        self.logger.info("Applying StandardScaler to features.")
        scaler = StandardScaler(); X_train = scaler.fit_transform(X_train); X_val = scaler.transform(X_val)
        self.scaler = scaler # Store scaler
        
        # Convert to tensors
        self.x_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.long).to(self.device) # Long for CrossEntropy
        self.x_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        self.y_val = torch.tensor(y_val, dtype=torch.long).to(self.device)
        self.use_weights = False
        self.target_scaler = None # No target scaling for classification

    def _prepare_optimize_data(self):
        """Prepares data specifically needed for KAN's optimize method.

        This includes subsampling the training data if specified in the config
        and creating one-hot encoded targets for classification tasks.
        """
        self.logger.info("Preparing data for KAN optimize step.")
        
        # Determine sample size for optimization step
        optimize_sample_size_config = self.dataset_config.get('optimize_sample_size')
        if optimize_sample_size_config:
             optimize_sample_size = min(optimize_sample_size_config, len(self.x_train))
        else:
             optimize_sample_size = len(self.x_train)

        # Subsample if necessary
        if optimize_sample_size < len(self.x_train):
            self.logger.info(f"Subsampling train data to {optimize_sample_size} for optimize step.")
            indices = torch.randperm(len(self.x_train))[:optimize_sample_size]
            self.x_optimize = self.x_train[indices].clone()
            y_optimize_orig = self.y_train[indices].clone()
        else:
            self.logger.info("Using full training data for optimize step.")
            self.x_optimize = self.x_train.clone()
            y_optimize_orig = self.y_train.clone()
        
        # Prepare target for optimize step (one-hot for classification)
        if self.task_type == 'classification':
            # Get number of classes - handle potential KeyError
            try:
                num_classes = self.dataset_config['num_classes']
            except KeyError:
                self.logger.warning("'num_classes' not found in dataset config. Inferring from unique training labels.")
                num_classes = len(torch.unique(self.y_train)) # Infer from all train labels
            
            if y_optimize_orig.max() >= num_classes:
                self.logger.error(f"Label index {y_optimize_orig.max()} is out of bounds for num_classes={num_classes}. Check label mapping (should be 0-based).")
                # Option: Raise error, or try to infer num_classes again from the subsampled data
                num_classes = int(y_optimize_orig.max().item()) + 1 # Infer from max label in subset
                self.logger.warning(f"Adjusted num_classes to {num_classes} based on max label in optimize subset.")

            self.y_optimize_onehot = F.one_hot(y_optimize_orig, num_classes=num_classes).float().to(self.device)
            self.logger.info(f"Created one-hot encoded targets for optimize step with shape: {self.y_optimize_onehot.shape} (Num Classes: {num_classes})")
        else: # Regression
            self.y_optimize_onehot = y_optimize_orig.clone()
            self.logger.info(f"Using original regression targets for optimize step with shape: {self.y_optimize_onehot.shape}")

    # --- Training & Evaluation --- 
    def _train_and_evaluate(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                          opt_method: str, config: Dict[str, Any], num_epochs: int
                          ) -> Tuple[Dict[str, List], float]:
        model.to(self.device); metrics = defaultdict(list)
        criterion = nn.CrossEntropyLoss() if self.task_type == 'classification' else nn.MSELoss()
        if self.dataset_name.lower() == 'jane_street' and self.use_weights:
             criterion = lambda y_pred, y_true, w: torch.sum(w * (y_true.squeeze() - y_pred.squeeze())**2) / (torch.sum(w) + 1e-12)
        best_primary_metric_val = float('-inf') if self.higher_is_better else float('inf')
        
        pbar = trange(num_epochs, desc=f"Training {opt_method}", leave=False)
        for epoch in pbar:
            model.train(); optimizer.zero_grad(); output = model(self.x_train)
            loss_args = [output, self.y_train, self.w_train] if self.dataset_name.lower() == 'jane_street' and self.use_weights else [output, self.y_train]
            loss = criterion(*loss_args); loss.backward(); optimizer.step()
            
            model.eval(); epoch_metrics = {}
            with torch.no_grad():
                train_output = model(self.x_train); epoch_metrics['train_loss'] = loss.item()
                val_output = model(self.x_val)
                val_loss_args = [val_output, self.y_val, self.w_val] if self.dataset_name.lower() == 'jane_street' and self.use_weights else [val_output, self.y_val]
                epoch_metrics['val_loss'] = criterion(*val_loss_args).item()
                
                if self.task_type == 'regression':
                    epoch_metrics['train_mse'] = epoch_metrics['train_loss']
                    epoch_metrics['val_mse'] = epoch_metrics['val_loss']
                    epoch_metrics['train_r2'] = weighted_r2(self.y_train, train_output, self.w_train) if 'jane_street' in self.dataset_name.lower() and self.use_weights else r2_score(self.y_train, train_output)
                    epoch_metrics['val_r2'] = weighted_r2(self.y_val, val_output, self.w_val) if 'jane_street' in self.dataset_name.lower() and self.use_weights else r2_score(self.y_val, val_output)
                else:
                    epoch_metrics['train_accuracy'] = accuracy(self.y_train, train_output)
                    epoch_metrics['val_accuracy'] = accuracy(self.y_val, val_output)
                
                current_primary_val_metric_key = f'val_{self.primary_metric}'
                current_primary_metric = epoch_metrics.get(current_primary_val_metric_key)
                if current_primary_metric is None:
                     self.logger.error(f"Primary metric '{current_primary_val_metric_key}' not found in calculated metrics!")
                     current_primary_metric = epoch_metrics['val_loss']; temp_higher_is_better = False
                else: temp_higher_is_better = self.higher_is_better

            pbar.set_postfix({f"val_{self.primary_metric}": f"{current_primary_metric:.4f}"})
            is_better = (current_primary_metric > best_primary_metric_val) if temp_higher_is_better else (current_primary_metric < best_primary_metric_val)
            if is_better: best_primary_metric_val = current_primary_metric
            
            metrics['epoch'].append(epoch)
            for k, v in epoch_metrics.items(): metrics[k].append(v)
            
            row_data = {'opt_method': [opt_method], 'config': [str(config)], 'opt_time': [config.get('opt_time', np.nan)],
                        'epoch': [epoch], 'param_count': [config.get('param_count', np.nan)],
                        **{k: [v] for k, v in epoch_metrics.items()}}
            new_row = pd.DataFrame(row_data)
            if self.results_df.empty: self.results_df = new_row
            else: self.results_df = pd.concat([self.results_df.reindex(columns=self.results_df.columns.union(new_row.columns)),
                                               new_row.reindex(columns=self.results_df.columns.union(new_row.columns))], ignore_index=True)

            if epoch % 20 == 0:
                 log_msg = f"[{opt_method}] Cfg={config} Ep {epoch}/{num_epochs}, Loss(tr/v)={epoch_metrics['train_loss']:.4f}/{epoch_metrics['val_loss']:.4f}, Val {self.primary_metric.upper()}={current_primary_metric:.4f}"
                 self.logger.info(log_msg)
        pbar.close()
        return metrics, best_primary_metric_val

    # --- Grid Search --- 
    def run_grid_search(self, param_grid: Dict[str, List[Any]], num_epochs: int = 50, methods_to_run: Optional[List[str]] = None):
        self.logger.info(f"\n=== Running Grid Search for {self.dataset_name} ({self.task_type}) ===")
        all_opt = { 'QUBO': lambda k,x,y: k.optimize(x,y), 'IntegerProgramming': lambda k,x,y: k.optimize_integer_programming(x,y),
                      'Evolutionary': lambda k,x,y: k.optimize_evolutionary(x,y), 'GreedyHeuristic': lambda k,x,y: k.optimize_greedy_heuristic(x,y) }
        opt_methods_to_run = {k:v for k,v in all_opt.items() if k in methods_to_run} if methods_to_run else all_opt
        if not opt_methods_to_run: raise ValueError("No valid optimization methods selected.")
        self.logger.info(f"Running KAN optimization methods: {list(opt_methods_to_run.keys())}")
        
        best_configs_perf = { m: {'metric_val': float('-inf') if self.higher_is_better else float('inf'), 'config': None, 'time': 0} 
                              for m in opt_methods_to_run }
        best_models = {m: None for m in opt_methods_to_run}

        grid_keys = list(param_grid.keys())
        if not all(k in param_grid for k in grid_keys): raise ValueError("Grid missing required keys.")
        total_configs = np.prod([len(v) for v in param_grid.values()]) * len(opt_methods_to_run)
        self.logger.info(f"Grid keys: {grid_keys}, Total configs: {total_configs}")
        main_pbar = tqdm(total=int(total_configs), desc="Overall Grid Search", position=0, ncols=100)
        
        import itertools
        for param_combination in itertools.product(*param_grid.values()):
            base_config = dict(zip(grid_keys, param_combination))
            max_deg = base_config.get('max_degree', 5); hid_size = base_config.get('hidden_size', 64); lr = base_config.get('learning_rate', 1e-3)

            for method_name, optimize_fn in opt_methods_to_run.items():
                main_pbar.update(1)
                main_pbar.set_description(f"{method_name} Cfg:{base_config}")
                self.logger.info(f"\n--- Testing {method_name} --- Config: {base_config}")
                
                kan_config = FixedKANConfig(network_shape=[self.input_dim, hid_size, self.output_dim], max_degree=max_deg,
                                          trainable_coefficients=True, skip_qubo_for_hidden=False, default_hidden_degree=4)
                kan = FixedKAN(kan_config).to(self.device); p_count = count_parameters(kan)
                self.logger.info(f"KAN params: {p_count}")
                
                try:
                    self.logger.info(f"Running {method_name} optimization..."); start_time = time.time()
                    opt_data_x = self.x_optimize if self.task_type == 'classification' else self.x_train
                    opt_data_y = self.y_optimize_onehot if self.task_type == 'classification' else self.y_train
                    optimize_fn(kan, opt_data_x, opt_data_y)
                    opt_time = time.time() - start_time; self.logger.info(f"Opt done: {opt_time:.2f}s")
                    
                    current_run_config = {**base_config, 'opt_time': opt_time, 'param_count': p_count}
                    params = [p for layer in kan.layers for p in [layer.combine_W, layer.combine_b] + [n.w for n in layer.neurons] + [n.b for n in layer.neurons]]
                    optimizer = torch.optim.Adam(params, lr=lr)
                    
                    _, best_metric = self._train_and_evaluate(kan, optimizer, method_name, current_run_config, num_epochs)
                    
                    is_better = (best_metric > best_configs_perf[method_name]['metric_val']) if self.higher_is_better else (best_metric < best_configs_perf[method_name]['metric_val'])
                    if is_better:
                         best_configs_perf[method_name] = {'metric_val': best_metric, 'config': current_run_config, 'time': opt_time}
                         self.logger.info(f"** New best {method_name}: Val {self.primary_metric}={best_metric:.4f} **")
                         best_models[method_name] = {'model_state': kan.state_dict(), 'kan_config': kan_config, 'best_metric_val': best_metric, 'opt_time': opt_time}
                            
                except ImportError as ie: self.logger.warning(f"Skipping {method_name}: {ie}"); # Add placeholder...
                except Exception as e: self.logger.error(f"Error {method_name} (Cfg: {base_config}): {e}", exc_info=False) # Shorter error log
                finally: del kan; torch.cuda.empty_cache(); gc.collect()
        main_pbar.close()
        
        self.logger.info(f"\n=== Best Configurations for {self.dataset_name} ===")
        for method, result in best_configs_perf.items():
            if result['config']: self.logger.info(f"{method}: Val {self.primary_metric}={result['metric_val']:.4f}, Time={result['time']:.2f}s, Cfg={result['config']}")
            if best_models.get(method): # Save best model
                save_path = os.path.join(self.results_dir, f'kan_{method.lower()}_best.pth')
                try: torch.save(best_models[method], save_path); self.logger.info(f"Saved best {method} model: {save_path}")
                except Exception as e: self.logger.error(f"Failed to save model {method}: {e}")
            elif result['config']: self.logger.warning(f"No best model state found for {method}, though config exists.")
            else: self.logger.warning(f"No successful runs for {method}.")
        
        results_path = os.path.join(self.results_dir, f'{self.dataset_name}_optimization_comparison.csv')
        try: self.results_df.round(6).to_csv(results_path, index=False); self.logger.info(f"Results saved: {results_path}")
        except Exception as e: self.logger.error(f"Failed to save results CSV: {e}")

    # --- Plotting --- 
    def plot_results(self):
        """Plot comparison results based on task type and primary metric."""
        if self.results_df.empty: self.logger.warning("No results to plot."); return
        
        plot_dir = self.results_dir; ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        metric_col = f'val_{self.primary_metric}'
        metric_name = f"Validation {self.primary_metric.upper() if self.primary_metric == 'mse' else self.primary_metric.capitalize()}"
        loss_col = 'val_loss'
        
        # 1. Plot primary metric vs epoch
        plt.figure(figsize=(10, 6))
        has_data = False
        valid_methods = self.results_df.dropna(subset=[metric_col])['opt_method'].unique()
        for method in valid_methods:
            method_data = self.results_df[self.results_df['opt_method'] == method].copy()
            method_data.dropna(subset=[metric_col], inplace=True)
            if method_data.empty:
                 continue # Skip method if no valid data after dropna
            
            # Find best config using self.higher_is_better
            if self.higher_is_better:
                 best_configs_idx = method_data.loc[method_data.groupby('config')[metric_col].idxmax()]
                 best_config_idx = best_configs_idx[metric_col].idxmax()
            else: # Lower is better (MSE)
                 best_configs_idx = method_data.loc[method_data.groupby('config')[metric_col].idxmin()]
                 best_config_idx = best_configs_idx[metric_col].idxmin()
            
            best_config_str = method_data.loc[best_config_idx, 'config']
            best_data = method_data[method_data['config'] == best_config_str]
            if best_data.empty:
                 continue # Skip if best config data is somehow empty (shouldn't happen)
            plt.plot(best_data['epoch'], best_data[metric_col], label=f"{method} (Best Cfg)", lw=2); has_data = True
        
        if has_data:
            plt.title(f"{metric_name} vs Epoch (Best Config per Method) - {self.dataset_name}"); plt.xlabel("Epoch"); plt.ylabel(metric_name)
            if self.primary_metric == 'mse': plt.yscale('log')
            plt.grid(True, alpha=0.4); plt.legend(); plt.tight_layout()
            path = os.path.join(plot_dir, f'{self.dataset_name}_{self.primary_metric}_epoch_{ts}.png')
            try: plt.savefig(path, bbox_inches='tight'); self.logger.info(f"Saved plot: {path}")
            except Exception as e: self.logger.error(f"Save plot error: {e}")
            plt.close()
        else:
             self.logger.warning(f"No valid data to plot for {metric_name}."); plt.close()
             
        # --- Summary Plots ---
        summary_df = self.results_df.dropna(subset=['opt_time', metric_col]).copy()
        if summary_df.empty: self.logger.warning(f"No summary data (opt_time, {metric_col}) to plot."); return
        # Get best run per method using self.higher_is_better
        if self.higher_is_better:
            best_runs = summary_df.loc[summary_df.groupby('opt_method')[metric_col].idxmax()]
        else:
            best_runs = summary_df.loc[summary_df.groupby('opt_method')[metric_col].idxmin()]
        if best_runs.empty: self.logger.warning("No best runs for summary plots."); return

        methods = best_runs['opt_method'].tolist(); times = best_runs['opt_time'].tolist(); metrics = best_runs[metric_col].tolist()
        colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
        
        # 2. Opt Time Plot
        plt.figure(figsize=(8, 5)); bars = plt.bar(methods, times, color=colors)
        for bar, t in zip(bars, times): plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01, f'{t:.2f}s', ha='center', va='bottom', fontsize=9)
        plt.title(f'KAN Opt Time (Best Config) - {self.dataset_name}'); plt.ylabel('Time (s)'); plt.grid(axis='y', ls='--', alpha=0.6); plt.tight_layout()
        path = os.path.join(plot_dir, f'{self.dataset_name}_opttimes_{ts}.png')
        try: plt.savefig(path, bbox_inches='tight'); self.logger.info(f"Saved plot: {path}")
        except Exception as e: self.logger.error(f"Save plot error: {e}")
        plt.close()

        # 3. Final Performance Plot
        plt.figure(figsize=(8, 5)); bars = plt.bar(methods, metrics, color=colors)
        for bar, m in zip(bars, metrics): plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01, f'{m:.4f}', ha='center', va='bottom', fontsize=9)
        plt.title(f'Final {metric_name} (Best Config) - {self.dataset_name}'); plt.ylabel(metric_name)
        min_m = min(metrics) if metrics else 0
        bot_lim = min(0, min_m - abs(min_m*0.1)) if self.primary_metric == 'r2' else 0
        plt.ylim(bottom=bot_lim)
        if self.primary_metric == 'accuracy': plt.ylim(top=1.05)
        plt.grid(axis='y', ls='--', alpha=0.6); plt.tight_layout()
        path = os.path.join(plot_dir, f'{self.dataset_name}_final_{self.primary_metric}_{ts}.png')
        try: plt.savefig(path, bbox_inches='tight'); self.logger.info(f"Saved plot: {path}")
        except Exception as e: self.logger.error(f"Save plot error: {e}")
        plt.close()

# Main block for direct testing
if __name__ == "__main__":
    print("Testing ExperimentRunner directly...")
    test_cfg = {
        'experiment_name': 'DirectRunnerTest_HP_MSE', 'random_seed': 123, 'num_epochs': 3,
        'results_dir': 'runner_test_results_mse',
        'dataset': { 'name': 'california_housing', 'primary_metric': 'mse' },
        'parameter_grid': { 'default': {'max_degree': [3], 'hidden_size': [8], 'learning_rate': [1e-3]} }
    }
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        runner = ExperimentRunner(config=test_cfg)
        runner.run_grid_search(param_grid=test_cfg['parameter_grid']['default'], num_epochs=test_cfg['num_epochs'], methods_to_run=['GreedyHeuristic'])
        runner.plot_results()
        print(f"Test complete. Check {test_cfg['results_dir']}")
    except Exception as e: import traceback; print(f"Test failed: {e}"); traceback.print_exc() 