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
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score as sk_r2_score # Added MSE, aliased r2_score
import torchvision # For MNIST/CIFAR
import torchvision.transforms as transforms
from datasets import load_dataset # Added for Hugging Face datasets
# GBT Imports
import xgboost as xgb
import lightgbm as lgb
from lightgbm import early_stopping # Import the callback

# Local imports
from CP_KAN import FixedKANConfig, FixedKAN
# EasyTSF KAN Layer Imports (now from local copy)
from kanlayer_easytsf import (
    WaveKANLayer,
    NaiveFourierKANLayer,
    JacobiKANLayer,
    ChebyKANLayer,
    TaylorKANLayer,
    RBFKANLayer
)

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
        
        # --- Initialize Logger Early --- #
        self.logger = logging.getLogger("ExperimentRunner")
        self.logger.setLevel(logging.INFO) # Set default level
        # Configure logging (e.g., level from config) if needed here
        # Basic handler setup if not already configured by main script
        if not self.logger.hasHandlers():
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)
            # Optionally add FileHandler here too if main script doesn't handle it

        # --- Determine Task Type and Metric (Now logger is available) --- #
        self.task_type = self._determine_task_type()
        self.primary_metric = self._determine_primary_metric()
        self.higher_is_better = self.primary_metric in ['r2', 'accuracy', 'weighted_r2'] # Added weighted_r2

        os.makedirs(self.results_dir, exist_ok=True)
        self.results_df = pd.DataFrame()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Primary metric: {self.primary_metric} (Higher is better: {self.higher_is_better}) - Task: {self.task_type}")

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
                          opt_method: str, config: Dict[str, Any], num_epochs: int,
                          # --- New parameters for degradation tracking ---
                          track_degradation: bool = False,
                          degradation_patience: int = 50,
                          degradation_threshold_ratio: float = 0.3
                          ) -> Tuple[Dict[str, List], float, Optional[int]]: # Return best metric, peak epoch
        """Trains and evaluates a model, optionally tracking degradation metrics.

        Args:
            model: The model to train.
            optimizer: The optimizer.
            opt_method: String identifier for the optimization method/run type.
            config: Dictionary containing hyperparameters for this run.
            num_epochs: Total epochs to train.
            track_degradation: If True, track grad norm, weight change, degradation.
            degradation_patience: Epochs to wait after peak before stopping due to degradation.
            degradation_threshold_ratio: Stop if val metric drops below peak * threshold_ratio.

        Returns:
            Tuple: (metrics_dict, best_primary_metric_val, peak_epoch)
        """
        model.to(self.device); metrics = defaultdict(list)
        criterion = nn.CrossEntropyLoss() if self.task_type == 'classification' else nn.MSELoss()
        if self.dataset_name.lower() == 'jane_street' and self.use_weights:
             # --- Use weighted MSE Loss for Jane Street ---
             criterion = lambda y_pred, y_true, w: torch.sum(w * (y_true.squeeze() - y_pred.squeeze())**2) / (torch.sum(w) + 1e-12)
             
        best_primary_metric_val = float('-inf') if self.higher_is_better else float('inf')
        peak_epoch: Optional[int] = None
        epochs_since_peak = 0
        prev_weights = {name: param.clone().detach().cpu() for name, param in model.named_parameters() if param.requires_grad} if track_degradation else {}

        # Extract identifiers from the passed config
        model_type = config.get('model_type', 'Unknown')
        kan_opt_method = config.get('kan_opt_method') # Will be None for MLP
        param_count = config.get('param_count', np.nan)
        kan_opt_time = config.get('kan_opt_time', np.nan)
        core_config = {k: v for k, v in config.items() if k not in ['model_type', 'kan_opt_method', 'param_count', 'kan_opt_time']}

        pbar = trange(num_epochs, desc=f"Training {opt_method}", leave=False)
        stopped_early = False
        for epoch in pbar:
            model.train()
            epoch_loss = 0.0
            epoch_grad_norm = 0.0
            epoch_weight_change = 0.0
            batch_size = config.get('batch_size', self.config.get('training',{}).get('batch_size', 64)) # Get batch size
            n_batches = math.ceil(len(self.x_train) / batch_size)
            permutation = torch.randperm(self.x_train.size(0))

            for i in range(n_batches):
                indices = permutation[i * batch_size:(i + 1) * batch_size]
                x_batch, y_batch = self.x_train[indices], self.y_train[indices]
                w_batch = self.w_train[indices] if self.use_weights else None

                optimizer.zero_grad()
                output = model(x_batch)
                
                loss_args = [output, y_batch, w_batch] if self.dataset_name.lower() == 'jane_street' and self.use_weights else [output, y_batch]
                loss = criterion(*loss_args)

                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.error(f"NaN/Inf loss detected at epoch {epoch}, batch {i}. Stopping training.")
                    stopped_early = True; break # Stop batch loop and epoch loop
                    
                loss.backward()

                # --- Degradation Tracking: Grad Norm (before step) ---
                batch_grad_norm = 0.0
                if track_degradation:
                    for p in model.parameters():
                         if p.grad is not None:
                             param_norm = p.grad.detach().data.norm(2)
                             batch_grad_norm += param_norm.item() ** 2
                    epoch_grad_norm += (batch_grad_norm ** 0.5) # Sum of norms of batches

                optimizer.step()
                epoch_loss += loss.item() * len(x_batch) # Accumulate loss weighted by batch size
                
            if stopped_early: break # Exit epoch loop if loss was NaN/Inf
            
            avg_epoch_loss = epoch_loss / len(self.x_train)
            avg_epoch_grad_norm = epoch_grad_norm / n_batches if track_degradation else np.nan
            
            # --- Degradation Tracking: Weight Change (after step) ---
            current_weight_change = 0.0
            if track_degradation:
                 with torch.no_grad():
                      for name, param in model.named_parameters():
                          if param.requires_grad and name in prev_weights:
                               weight_diff = param.data - prev_weights[name].to(self.device)
                               current_weight_change += torch.norm(weight_diff).item()
                               prev_weights[name] = param.clone().detach().cpu()
            epoch_weight_change = current_weight_change if track_degradation else np.nan
            
            model.eval(); epoch_metrics = {}
            with torch.no_grad():
                # Use full dataset for Eval (can be slow, consider batching if needed)
                train_output = model(self.x_train)
                epoch_metrics['train_loss'] = avg_epoch_loss # Use calculated average epoch loss
                val_output = model(self.x_val)
                val_loss_args = [val_output, self.y_val, self.w_val] if self.dataset_name.lower() == 'jane_street' and self.use_weights else [val_output, self.y_val]
                epoch_metrics['val_loss'] = criterion(*val_loss_args).item()

                if self.task_type == 'regression':
                    # Use calculated train loss if MSE, otherwise recalc might be needed
                    epoch_metrics['train_mse'] = epoch_metrics['train_loss'] 
                    epoch_metrics['val_mse'] = epoch_metrics['val_loss']
                    epoch_metrics['train_r2'] = weighted_r2(self.y_train, train_output, self.w_train) if 'jane_street' in self.dataset_name.lower() and self.use_weights else r2_score(self.y_train, train_output)
                    epoch_metrics['val_r2'] = weighted_r2(self.y_val, val_output, self.w_val) if 'jane_street' in self.dataset_name.lower() and self.use_weights else r2_score(self.y_val, val_output)
                else: # Classification
                    epoch_metrics['train_accuracy'] = accuracy(self.y_train, train_output)
                    epoch_metrics['val_accuracy'] = accuracy(self.y_val, val_output)

                # --- Determine Primary Metric for Logging/Comparison --- #
                current_primary_val_metric_key = f'val_{self.primary_metric}'
                current_primary_metric = epoch_metrics.get(current_primary_val_metric_key)
                temp_higher_is_better = self.higher_is_better
                if current_primary_metric is None:
                     self.logger.error(f"Primary metric '{current_primary_val_metric_key}' not found! Using val_loss.")
                     current_primary_metric = epoch_metrics['val_loss']; temp_higher_is_better = False

            pbar.set_postfix({f"val_{self.primary_metric}": f"{current_primary_metric:.4f}"}) # Update progress bar
            
            # --- Degradation Tracking: Check vs Peak & Early Stopping --- #
            current_degradation_val = np.nan
            if track_degradation and pd.notna(current_primary_metric):
                is_better = (current_primary_metric > best_primary_metric_val) if temp_higher_is_better else (current_primary_metric < best_primary_metric_val)
                if is_better:
                    best_primary_metric_val = current_primary_metric
                    peak_epoch = epoch
                    epochs_since_peak = 0
                    # Optionally save best model checkpoint here if tracking degradation
                else:
                    epochs_since_peak += 1
                
                if peak_epoch is not None and epoch > peak_epoch: # Calculate degradation after peak
                    current_degradation_val = abs(best_primary_metric_val - current_primary_metric) # Absolute difference
                
                # Degradation-based Early Stopping Check
                if peak_epoch is not None and epochs_since_peak >= degradation_patience:
                    threshold = abs(best_primary_metric_val * degradation_threshold_ratio)
                    # Stop if metric is WORSE than peak by more than threshold * peak
                    stop = False
                    if temp_higher_is_better:
                        stop = current_primary_metric < (best_primary_metric_val - threshold) # Degraded too much
                    else: # lower is better
                        stop = current_primary_metric > (best_primary_metric_val + threshold) # Increased too much
                        
                    if stop:
                         self.logger.warning(f"DEGRADATION EARLY STOPPING for {opt_method} at epoch {epoch}. "
                                             f"Metric ({current_primary_metric:.4f}) degraded beyond threshold ({threshold:.4f}) of peak ({best_primary_metric_val:.4f}) "
                                             f"after {epochs_since_peak} epochs.")
                         stopped_early = True
                         # Don't break yet, finish logging this epoch
            elif pd.notna(current_primary_metric): # Standard best metric tracking if not tracking degradation
                 is_better = (current_primary_metric > best_primary_metric_val) if temp_higher_is_better else (current_primary_metric < best_primary_metric_val)
                 if is_better: best_primary_metric_val = current_primary_metric; peak_epoch = epoch
                 
            # --- Append results for this epoch to the main DataFrame --- #
            row_data = {
                 'model_type': model_type, 'kan_opt_method': kan_opt_method,
                 'config': str(core_config), 'param_count': param_count, 'kan_opt_time': kan_opt_time,
                 'epoch': epoch, **epoch_metrics,
                 # Add degradation metrics if tracked
                 'grad_norm': avg_epoch_grad_norm, 'weight_change': epoch_weight_change,
                 'degradation_from_peak': current_degradation_val,
                 'kan_optimized_model_path': config.get('kan_optimized_model_path') # Get model path from config
            }
            # Ensure all expected columns exist before appending
            for col in ['grad_norm', 'weight_change', 'degradation_from_peak', 'kan_optimized_model_path']:
                 if col not in self.results_df.columns: self.results_df[col] = pd.NA
                 
            new_row = pd.DataFrame([row_data])
            # Align columns before concat
            shared_cols = self.results_df.columns.intersection(new_row.columns)
            missing_in_df = new_row.columns.difference(self.results_df.columns)
            missing_in_row = self.results_df.columns.difference(new_row.columns)
            for col in missing_in_df: self.results_df[col] = pd.NA
            for col in missing_in_row: new_row[col] = pd.NA
            self.results_df = pd.concat([self.results_df.reindex(columns=new_row.columns, fill_value=pd.NA),
                                            new_row], ignore_index=True)

            if epoch % 20 == 0:
                 log_msg = f"[{opt_method}] Cfg={core_config} Ep {epoch}/{num_epochs}, Loss(tr/v)={epoch_metrics['train_loss']:.4f}/{epoch_metrics['val_loss']:.4f}, Val {self.primary_metric.upper()}={current_primary_metric:.4f}"
                 self.logger.info(log_msg)
                 
            if stopped_early: break # Exit epoch loop now if early stopping triggered
        # End of epoch loop
        pbar.close()
        
        # Handle filling remaining epochs if stopped early
        if stopped_early and epoch < num_epochs - 1:
             last_row = self.results_df.iloc[-1].copy()
             fill_row = last_row # Initialize fill_row with the last valid data
             for fill_epoch in range(epoch + 1, num_epochs):
                  fill_row['epoch'] = fill_epoch
                  # Add other metrics from last valid epoch if needed, or keep as NaN/last value
                  fill_row['train_loss'] = np.nan # Mark as not computed
                  fill_row['val_loss'] = np.nan
                  fill_row[f'train_{self.primary_metric}'] = np.nan
                  fill_row[f'val_{self.primary_metric}'] = np.nan
                  fill_row['grad_norm'] = np.nan
                  fill_row['weight_change'] = np.nan
                  fill_row['degradation_from_peak'] = np.nan
                  fill_row['kan_optimized_model_path'] = np.nan
                  new_row = pd.DataFrame([fill_row])
                  # Align columns again
                  shared_cols = self.results_df.columns.intersection(new_row.columns)
                  missing_in_df = new_row.columns.difference(self.results_df.columns)
                  missing_in_row = self.results_df.columns.difference(new_row.columns)
                  for col in missing_in_df: self.results_df[col] = pd.NA
                  for col in missing_in_row: new_row[col] = pd.NA
                  self.results_df = pd.concat([self.results_df.reindex(columns=new_row.columns, fill_value=pd.NA), new_row], ignore_index=True)
                  
        return metrics, best_primary_metric_val, peak_epoch # Return epoch metrics history, best val score, and peak epoch

    # --- Grid Search ---
    def run_grid_search(self, num_epochs: Optional[int] = None):
        """Runs grid search based on the structured config (parameter_grids)."""
        
        # Get settings from the main config
        methods_to_run = self.config.get('methods_to_run', ['FixedKAN']) # Default to KAN if not specified
        num_epochs = num_epochs if num_epochs is not None else self.config.get('num_epochs', 50)
        parameter_grids = self.config.get('parameter_grids')

        if not parameter_grids:
            raise ValueError("Configuration file must contain 'parameter_grids' section for grid search.")

        self.logger.info(f"\n=== Starting Grid Search for {self.dataset_name} ({self.task_type}) ===")
        self.logger.info(f"Methods to run: {methods_to_run}")
        self.logger.info(f"Number of epochs per run: {num_epochs}")

        # --- KAN Optimization Methods (if KAN is run) ---
        kan_optimizers = { 
            'QUBO': lambda k,x,y,cfg: k.optimize(x,y), # Placeholder
            'IntegerProgramming': lambda k,x,y,cfg: k.optimize_integer_programming(x,y), # Placeholder
            'Evolutionary': lambda k,x,y,cfg: k.optimize_evolutionary(x,y), # Placeholder
            'GreedyHeuristic': lambda k,x,y,cfg: k.optimize_greedy_heuristic(x,y) # Placeholder
        }
        kan_methods_to_run = {}
        if 'FixedKAN' in methods_to_run:
            kan_opt_method_names = self.config.get('kan_optimize_methods', list(kan_optimizers.keys()))
            kan_methods_to_run = {k:v for k,v in kan_optimizers.items() if k in kan_opt_method_names}
            if not kan_methods_to_run: self.logger.warning("No KAN optimization methods selected/valid for FixedKAN run.")
            else: self.logger.info(f"KAN optimization methods to attempt: {list(kan_methods_to_run.keys())}")

        import itertools

        # --- Iterate through Model Methods (e.g., FixedKAN, MLP) ---
        for method_name in methods_to_run:
            self.logger.info(f"\n--- Processing Model Type: {method_name} ---")
            
            specific_grid_config = parameter_grids.get(method_name)
            if not specific_grid_config:
                self.logger.warning(f"No parameter grid found for method '{method_name}' in 'parameter_grids'. Skipping.")
                continue
                
            # Use the 'default' grid for now
            param_grid = specific_grid_config.get('default')
            if not param_grid:
                self.logger.warning(f"No 'default' grid found within parameter_grids.{method_name}. Skipping.")
                continue
            
            grid_keys = list(param_grid.keys())
            param_values = list(param_grid.values())
            total_combinations = np.prod([len(v) for v in param_values])
            self.logger.info(f"Parameter grid keys for {method_name}: {grid_keys}")
            self.logger.info(f"Total parameter combinations for {method_name}: {total_combinations}")

            main_pbar = tqdm(itertools.product(*param_values), total=total_combinations, desc=f"Grid Search ({method_name})", position=0, ncols=100)
            
            # --- Iterate through Hyperparameter Combinations for this method ---
            for param_combination_tuple in main_pbar:
                current_params = dict(zip(grid_keys, param_combination_tuple))
                main_pbar.set_postfix(current_params, refresh=False)
                self.logger.debug(f"Testing {method_name} with params: {current_params}")

                model: Optional[nn.Module] = None
                model_config_obj = None 

                # --- Initialize Correct Model Type ---
                try: 
                    if method_name == 'FixedKAN':
                        hidden_size_cfg = current_params.get('hidden_size', 64) 
                        if isinstance(hidden_size_cfg, int): hidden_layers = [hidden_size_cfg]
                        elif isinstance(hidden_size_cfg, list): hidden_layers = hidden_size_cfg
                        else: self.logger.error(f"Invalid hidden_size type '{type(hidden_size_cfg)}' for KAN."); continue
                        max_deg = current_params.get('max_degree', 3)
                        network_shape = [self.input_dim] + hidden_layers + [self.output_dim]
                        kan_specific_params = { 'network_shape': network_shape, 'max_degree': max_deg, 
                                              'trainable_coefficients': current_params.get('trainable_coefficients', True),
                                              'skip_qubo_for_hidden': current_params.get('skip_qubo_for_hidden', False),
                                              'default_hidden_degree': current_params.get('default_hidden_degree', 4) }
                        model_config_obj = FixedKANConfig(**kan_specific_params)
                        model = FixedKAN(model_config_obj).to(self.device)
                        self.logger.debug(f"Initialized FixedKAN with shape: {network_shape}")

                    elif method_name == 'MLP':
                        hidden_layers = current_params.get('mlp_hidden_layers', [64, 64]) 
                        activation = current_params.get('mlp_activation', 'ReLU') 
                        mlp_full_layers = [self.input_dim] + hidden_layers + [self.output_dim]
                        model_config_obj = {'layers': mlp_full_layers, 'activation': activation}
                        model = self._create_mlp(mlp_full_layers, activation).to(self.device)
                        self.logger.debug(f"Initialized MLP with layers: {mlp_full_layers}, activation: {activation}")
                    
                    # --- Add EasyTSF KAN Variants --- #
                    elif method_name == 'WaveKAN':
                        wavelet_type = current_params.get('wavelet_type', 'mexican_hat')
                        model_config_obj = {'wavelet_type': wavelet_type} 
                        model = WaveKANLayer(self.input_dim, self.output_dim, wavelet_type=wavelet_type, with_bn=False).to(self.device)
                        self.logger.debug(f"Initialized WaveKANLayer with type: {wavelet_type}")
                        
                    elif method_name == 'FourierKAN':
                        gridsize = current_params.get('fourier_gridsize', 10) # Example default
                        model_config_obj = {'gridsize': gridsize}
                        model = NaiveFourierKANLayer(self.input_dim, self.output_dim, gridsize=gridsize).to(self.device)
                        self.logger.debug(f"Initialized NaiveFourierKANLayer with gridsize: {gridsize}")
                        
                    elif method_name == 'JacobiKAN':
                        degree = current_params.get('jacobi_degree', 5) # Example default
                        a = current_params.get('jacobi_a', 1.0)
                        b = current_params.get('jacobi_b', 1.0)
                        model_config_obj = {'degree': degree, 'a': a, 'b': b}
                        model = JacobiKANLayer(self.input_dim, self.output_dim, degree=degree, a=a, b=b).to(self.device)
                        self.logger.debug(f"Initialized JacobiKANLayer with degree: {degree}, a={a}, b={b}")

                    elif method_name == 'ChebyKAN':
                        degree = current_params.get('cheby_degree', 5) # Example default
                        model_config_obj = {'degree': degree}
                        model = ChebyKANLayer(self.input_dim, self.output_dim, degree=degree).to(self.device)
                        self.logger.debug(f"Initialized ChebyKANLayer with degree: {degree}")
                        
                    elif method_name == 'TaylorKAN':
                        order = current_params.get('taylor_order', 3) # Example default
                        addbias = current_params.get('taylor_addbias', True)
                        model_config_obj = {'order': order, 'addbias': addbias}
                        model = TaylorKANLayer(self.input_dim, self.output_dim, order=order, addbias=addbias).to(self.device)
                        self.logger.debug(f"Initialized TaylorKANLayer with order: {order}, addbias={addbias}")
                        
                    elif method_name == 'RBFKAN':
                        num_centers = current_params.get('rbf_num_centers', int(np.sqrt(self.input_dim * self.output_dim)) or 10) # Example default heuristic
                        alpha = current_params.get('rbf_alpha', 1.0)
                        model_config_obj = {'num_centers': num_centers, 'alpha': alpha}
                        model = RBFKANLayer(self.input_dim, self.output_dim, num_centers=num_centers, alpha=alpha).to(self.device)
                        self.logger.debug(f"Initialized RBFKANLayer with num_centers: {num_centers}, alpha={alpha}")
                        
                    # --- End EasyTSF KAN Variants --- #
                    
                    elif method_name == 'LightGBM':
                        self.logger.debug(f"Initializing LightGBM...")
                        model_config_obj = current_params # Store params for GBT
                        gbt_params = {k: v for k, v in current_params.items() if k not in ['learning_rate']} 
                        gbt_params['random_state'] = self.config.get('random_seed', 42)
                        gbt_params['verbose'] = -1
                        if self.task_type == 'classification':
                            model = lgb.LGBMClassifier(**gbt_params)
                        else: 
                            model = lgb.LGBMRegressor(**gbt_params)
                        model_instance_for_eval = model 
                        
                    else:
                        self.logger.warning(f"Unsupported model type '{method_name}' during initialization. Skipping.")
                        continue
                except Exception as e:
                    self.logger.error(f"Error initializing {method_name} with params {current_params}: {e}", exc_info=True)
                    continue 
                
                # Check if model initialization succeeded before proceeding
                model_initialized = (model is not None) or (method_name == 'LightGBM' and model_instance_for_eval is not None)
                if not model_initialized:
                     self.logger.error(f"Model object is None/not initialized for {method_name}, skipping combination.")
                     continue 
                
                # Get param count for NNs, use placeholder for GBTs
                nn_model_types = ['FixedKAN', 'MLP', 'WaveKAN', 'FourierKAN', 'JacobiKAN', 'ChebyKAN', 'TaylorKAN', 'RBFKAN']
                if method_name in nn_model_types:
                    p_count = count_parameters(model)
                    self.logger.debug(f"{method_name} params: {p_count}")
                else: # GBTs (LightGBM)
                    p_count = -1 
                    self.logger.debug(f"{method_name} (Param count not applicable)")

                # --- Handle Training/Evaluation based on Model Type --- #
                if method_name in nn_model_types: # Check if it's an NN model
                    # --- Optimizer for NN models ---
                    lr = current_params.get('learning_rate', self.config.get('training',{}).get('learning_rate', 1e-3))
                    optimizer_name = current_params.get('optimizer', self.config.get('training',{}).get('optimizer', 'Adam'))
                    try:
                        optimizer_cls = getattr(torch.optim, optimizer_name)
                        optimizer = optimizer_cls(model.parameters(), lr=lr)
                    except AttributeError:
                        self.logger.error(f"Optimizer '{optimizer_name}' not found. Skipping.")
                        if model: del model 
                        torch.cuda.empty_cache(); gc.collect()
                        continue
                    if method_name == 'FixedKAN':
                        for kan_opt_method_name, kan_optimize_fn in kan_methods_to_run.items():
                            self.logger.info(f"--- Running KAN Optimize: {kan_opt_method_name} ---")
                            model_for_opt = model 
                            optimizer_for_train = optimizer 
                            kan_opt_time = 0.0
                            kan_opt_start_time = time.time()
                            optimized_state_path = None # Initialize path variable
                            try:
                                # We assume optimize methods modify the model in-place
                                kan_optimize_fn(model_for_opt, self.x_optimize, self.y_optimize_onehot, model_config_obj)
                                kan_opt_time = time.time() - kan_opt_start_time
                                self.logger.info(f"KAN Optimize ({kan_opt_method_name}) done: {kan_opt_time:.2f}s")
                                
                                # --- Save Optimized KAN Model Object ---
                                state_save_dir = os.path.join(self.results_dir, "optimized_kan_models") # Changed folder name
                                os.makedirs(state_save_dir, exist_ok=True)
                                # Create a unique filename (e.g., based on hash of config + opt method)
                                import hashlib
                                config_hash = hashlib.md5(str(current_params).encode()).hexdigest()[:8]
                                model_filename = f"kan_{config_hash}_{kan_opt_method_name}_model.pth" # Changed extension/name
                                optimized_model_path = os.path.join(state_save_dir, model_filename)
                                try:
                                     # Save the entire model object, not just state_dict
                                     torch.save(model_for_opt, optimized_model_path) 
                                     self.logger.debug(f"Saved optimized KAN model object to: {optimized_model_path}")
                                except Exception as save_e:
                                     self.logger.error(f"Failed to save optimized KAN model object: {save_e}")
                                     optimized_model_path = None # Ensure path is None if saving failed
                                     
                                # --- Prepare config for training run ---
                                current_run_config = {
                                    **current_params, 
                                    'model_type': method_name,
                                    'kan_opt_method': kan_opt_method_name, 
                                    'kan_opt_time': kan_opt_time, 
                                    'param_count': p_count, 
                                    # Use a distinct key for the model path
                                    'kan_optimized_model_path': optimized_model_path 
                                }
                                # Unpack 3 values now
                                _, _, _ = self._train_and_evaluate(
                                    model=model_for_opt, 
                                    optimizer=optimizer_for_train, 
                                    opt_method=f"KAN-{kan_opt_method_name}", 
                                    config=current_run_config, 
                                    num_epochs=num_epochs
                                )
                            except ImportError as ie:
                                self.logger.warning(f"Skipping KAN Optimize {kan_opt_method_name}: {ie}")
                            except Exception as e:
                                self.logger.error(f"Error KAN Optimize {kan_opt_method_name}: {e}", exc_info=False)
                    else: # Other NNs (MLP, WaveKAN, etc.)
                        current_run_config = {**current_params, 'param_count': p_count, 'model_type': method_name, 'kan_opt_method': 'N/A', 'kan_opt_time': 0.0}
                        # Unpack 3 values now
                        _, _, _ = self._train_and_evaluate(model, optimizer, method_name, current_run_config, num_epochs)
                    del model, optimizer 

                elif method_name == 'LightGBM': # Only LightGBM remains here
                    self._train_evaluate_gbt(model_instance_for_eval, method_name, current_params)
                    del model_instance_for_eval 
                    
                else:
                    self.logger.error(f"Logic error: Reached unexpected training path for {method_name}.")
            # End of hyperparameter combinations loop
            main_pbar.close()
            # End of model type loop

        # --- Save Final Results (using the member self.results_df) --- 
        results_path = os.path.join(self.results_dir, f'{self.dataset_name}_grid_search_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        try:
            if not self.results_df.empty:
                 self.results_df = self.results_df.round(6)
                 # Define desired column order
                 core_cols = ['model_type', 'kan_opt_method', f'val_{self.primary_metric}', 'train_{self.primary_metric}', 'val_loss', 'train_loss', 'param_count', 'kan_opt_time', 'epoch']
                 # Ensure core columns exist, handle potential missing ones gracefully
                 existing_core_cols = [c for c in core_cols if c in self.results_df.columns]
                 # Get remaining columns (parameters)
                 param_cols = sorted([k for k in self.results_df.columns if k not in existing_core_cols])
                 final_cols = existing_core_cols + param_cols
                 self.results_df = self.results_df[final_cols] # Reorder
                 self.results_df.to_csv(results_path, index=False)
                 self.logger.info(f"Grid search results saved: {results_path}")
            else:
                 self.logger.warning("No results were generated to save.")
        except Exception as e:
            self.logger.error(f"Failed to save results CSV: {e}")

        # --- Generate Degradation Config (if requested) ---
        if self.config.get('generate_degradation_config', False) and not self.results_df.empty:
            self._generate_degradation_config()

    def _generate_degradation_config(self):
        """Finds best KAN/MLP configs and generates a degradation study config file."""
        self.logger.info("Attempting to generate degradation study config file...")
        
        results_df = self.results_df.copy()
        results_df.dropna(subset=[f'val_{self.primary_metric}'], inplace=True) # Only consider runs with valid final metric
        if results_df.empty:
            self.logger.warning("No valid results found in DataFrame to determine best models for degradation config.")
            return

        best_kan_run = None
        best_mlp_run = None

        # Find the best overall run for each model type (KAN variations and MLP)
        idx_best = results_df.loc[results_df.groupby('model_type')[f'val_{self.primary_metric}'].idxmax()] if self.higher_is_better else results_df.loc[results_df.groupby('model_type')[f'val_{self.primary_metric}'].idxmin()]
        
        # Specifically find KAN (any opt method) and MLP
        kan_types = ['FixedKAN'] # Add other KAN types if needed
        best_kan_row = idx_best[idx_best['model_type'].isin(kan_types)]
        best_mlp_row = idx_best[idx_best['model_type'] == 'MLP']

        if best_kan_row.empty:
            self.logger.warning("Could not find a best KAN run from the results.")
        else:
            # If multiple KAN opt methods ran, pick the overall best KAN
            best_kan_idx = best_kan_row[f'val_{self.primary_metric}'].idxmax() if self.higher_is_better else best_kan_row[f'val_{self.primary_metric}'].idxmin()
            best_kan_run = best_kan_row.loc[best_kan_idx]
            self.logger.info(f"Best KAN configuration found: {best_kan_run['config']} (Val {self.primary_metric}: {best_kan_run[f'val_{self.primary_metric}']:.4f})")

        # Check the DataFrame slice returned by the lookup FIRST
        if best_mlp_row.empty:
            self.logger.warning("Could not find a best MLP run from the results.")
            # Explicitly set best_mlp_run to None if not found
            best_mlp_run = None 
        else:
            # Assign only if the row was found
            best_mlp_run = best_mlp_row.iloc[0] 
            self.logger.info(f"Best MLP configuration found: {best_mlp_run['config']} (Val {self.primary_metric}: {best_mlp_run[f'val_{self.primary_metric}']:.4f})")

        # Check if at least one model was found before proceeding
        if best_kan_run is None and best_mlp_run is None:
            self.logger.error("Failed to find best configurations for either KAN or MLP. Cannot generate degradation config.")
            return

        # --- Construct the new config dictionary --- #
        degradation_config = {
            'experiment_name': f"{self.config.get('experiment_name', 'Experiment')}_DegradationStudy",
            'experiment_type': 'degradation_study',
            'random_seed': self.config.get('random_seed', 42),
            'results_dir': os.path.join(self.results_dir, 'degradation_study_results'), # Subdirectory
            'dataset': self.dataset_config, # Reuse original dataset config
            'degradation_models': [],
            # Get degradation-specific training params from original config
            'training': self.config.get('degradation_study_params', 
                                        self.config.get('training', {})), # Fallback to original training params
        }
        # Add degradation tracking flags to training config
        degradation_config['training']['track_degradation_metrics'] = True

        # Add best models found
        if best_kan_run is not None:
            try:
                 # Convert the string representation back to a dict
                 kan_core_config = eval(best_kan_run['config'])
                 kan_config_for_study = {
                     'model_type': best_kan_run['model_type'],
                     'kan_opt_method': best_kan_run.get('kan_opt_method', 'N/A'),
                     # Use the correct key for model path
                     'kan_optimized_model_path': best_kan_run.get('kan_optimized_model_path'), 
                     **kan_core_config
                 }
                 # Ensure model path is valid before adding
                 model_path = kan_config_for_study.get('kan_optimized_model_path')
                 if model_path and os.path.exists(model_path):
                     degradation_config['degradation_models'].append(kan_config_for_study)
                 else:
                     self.logger.warning(f"Optimized KAN model path was invalid or missing ({model_path}). KAN will not be included in degradation study config.")
            except Exception as e:
                 self.logger.error(f"Error processing best KAN run config: {best_kan_run['config']}. Error: {e}")

        if best_mlp_run is not None:
            try:
                 mlp_core_config = eval(best_mlp_run['config'])
                 mlp_config_for_study = {
                     'model_type': best_mlp_run['model_type'],
                     **mlp_core_config
                 }
                 degradation_config['degradation_models'].append(mlp_config_for_study)
            except Exception as e:
                 self.logger.error(f"Error parsing MLP config string: {best_mlp_run['config']}. Error: {e}")

        # --- Save the new config as YAML --- #
        output_path = os.path.join(self.results_dir, 'config_degradation_study_generated.yaml')
        try:
            import yaml
            with open(output_path, 'w') as f:
                yaml.dump(degradation_config, f, default_flow_style=False, sort_keys=False)
            self.logger.info(f"Degradation study configuration saved to: {output_path}")
        except ImportError:
            self.logger.error("PyYAML library not found. Cannot save degradation config YAML.")
        except Exception as e:
            self.logger.error(f"Error saving degradation config YAML: {e}")

    def run_degradation_study(self):
        """Runs the degradation analysis using specific model configs from the config file."""
        self.logger.info("\n=== Starting Degradation Study ===")
        
        # Get model configurations to run for the study
        models_to_run = self.config.get('degradation_models')
        if not models_to_run:
            self.logger.error("No models specified under 'degradation_models' in the config. Cannot run study.")
            return
            
        # Get training parameters for the degradation study
        training_params = self.config.get('training', {})
        num_epochs = training_params.get('num_epochs', 200) # Use possibly extended epochs
        batch_size = training_params.get('batch_size', 64) # Use batch size from config
        track_degradation = training_params.get('track_degradation_metrics', True) # Should be True
        degradation_patience = training_params.get('degradation_patience', 50)
        degradation_threshold = training_params.get('degradation_threshold_ratio', 0.3)

        # Ensure results_df is clean for this specific study run
        self.results_df = pd.DataFrame()

        for model_config in models_to_run:
            model_type = model_config.get('model_type')
            core_params = {k: v for k, v in model_config.items() if k not in ['model_type', 'kan_opt_method']} # Exclude identifiers
            self.logger.info(f"\n--- Running Degradation Study for: {model_type} --- Configuruation {core_params}")
            
            model: Optional[nn.Module] = None
            model_config_obj = None
            p_count = np.nan
            kan_opt_time = np.nan # No KAN optimize step during degradation run
            kan_opt_method = model_config.get('kan_opt_method', 'N/A') # Get original opt method if KAN
            
            # --- Initialize Model --- 
            try:
                if model_type == 'FixedKAN':
                    # --- Load the entire optimized model object --- 
                    model_path = model_config.get('kan_optimized_model_path')
                    if model_path and os.path.exists(model_path):
                        try:
                            model = torch.load(model_path, map_location=self.device)
                            model.to(self.device) # Ensure model is on the correct device after loading
                            self.logger.info(f"Successfully loaded optimized KAN model object from {model_path}")
                            # Re-count parameters after loading the specific model
                            p_count = count_parameters(model)
                        except Exception as load_e:
                            self.logger.error(f"Failed to load optimized KAN model object from {model_path}: {load_e}. Skipping KAN.")
                            continue # Skip this KAN model if loading failed
                    else:
                         self.logger.error(f"Optimized KAN model path not found or invalid ({model_path}). Skipping KAN.")
                         continue # Skip this KAN model if path is invalid

                    # No need to initialize from config if loading full model
                    # model_config_obj = FixedKANConfig(**kan_specific_params)
                    # model = FixedKAN(model_config_obj).to(self.device)
                    
                elif model_type == 'MLP':
                    hidden_layers = core_params.get('mlp_hidden_layers', [64, 64]) 
                    activation = core_params.get('mlp_activation', 'ReLU') 
                    mlp_full_layers = [self.input_dim] + hidden_layers + [self.output_dim]
                    model = self._create_mlp(mlp_full_layers, activation).to(self.device)
                    if model: p_count = count_parameters(model)
                    self.logger.debug(f"Initialized MLP (Degradation) layers: {mlp_full_layers}")
                # Add other model types if needed, copying initialization logic
                else:
                     self.logger.warning(f"Unsupported model type '{model_type}' for degradation study run. Skipping.")
                     continue
                     
            except Exception as e:
                 self.logger.error(f"Error initializing {model_type} for degradation study: {e}", exc_info=True)
                 continue
                 
            if model is None: continue # Skip if initialization failed
            
            # --- Optimizer --- 
            lr = core_params.get('learning_rate', training_params.get('learning_rate', 1e-3))
            optimizer_name = core_params.get('optimizer', training_params.get('optimizer', 'Adam'))
            try:
                optimizer_cls = getattr(torch.optim, optimizer_name)
                optimizer = optimizer_cls(model.parameters(), lr=lr)
            except AttributeError:
                self.logger.error(f"Optimizer '{optimizer_name}' not found. Skipping {model_type}.")
                del model; torch.cuda.empty_cache(); gc.collect()
                continue
                
            # --- Train & Evaluate with Degradation Tracking --- 
            current_run_config = { 
                 **core_params, # Include original core hyperparams
                 'model_type': model_type,
                 'kan_opt_method': kan_opt_method,
                 'param_count': p_count,
                 'kan_opt_time': kan_opt_time, # Should be NaN or 0 for degradation run
                 'batch_size': batch_size # Pass batch_size for training loop
            }
            
            try:
                self._train_and_evaluate(
                    model=model, 
                    optimizer=optimizer, 
                    opt_method=f"{model_type}-Degradation", 
                    config=current_run_config, 
                    num_epochs=num_epochs,
                    track_degradation=track_degradation,
                    degradation_patience=degradation_patience,
                    degradation_threshold_ratio=degradation_threshold
                )
            except Exception as e:
                 self.logger.exception(f"Error during degradation training for {model_type}: {e}")
            finally:
                 del model, optimizer; torch.cuda.empty_cache(); gc.collect()

        # --- Save Final Results for Degradation Study --- 
        results_path = os.path.join(self.config['results_dir'], f'degradation_study_{self.dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        try:
            if not self.results_df.empty:
                 # Use same reordering logic as grid search saving
                 core_cols = ['model_type', 'kan_opt_method', f'val_{self.primary_metric}', 'train_{self.primary_metric}', 
                              'val_loss', 'train_loss', 'param_count', 'kan_opt_time', 'epoch',
                              'grad_norm', 'weight_change', 'degradation_from_peak', 'kan_optimized_model_path'] # Include degradation cols
                 existing_core_cols = [c for c in core_cols if c in self.results_df.columns]
                 param_cols = sorted([k for k in self.results_df.columns if k not in existing_core_cols])
                 final_cols = existing_core_cols + param_cols
                 self.results_df = self.results_df[final_cols].round(6)
                 self.results_df.to_csv(results_path, index=False)
                 self.logger.info(f"Degradation study results saved: {results_path}")
                 # Optionally call plotting function here
                 # self.plot_degradation_results() # A new plotting function?
            else:
                 self.logger.warning("No results generated during degradation study to save.")
        except Exception as e:
            self.logger.error(f"Failed to save degradation study results CSV: {e}")

        # --- Plot Degradation Study Results --- 
        self.plot_degradation_study_results(results_path) 

    def plot_degradation_study_results(self, results_csv_path: str):
        """Creates plots summarizing the degradation study results."""
        self.logger.info(f"Generating plots for degradation study from: {results_csv_path}")
        plot_dir = os.path.dirname(results_csv_path) # Save plots in the same dir as the CSV
        
        if not os.path.exists(results_csv_path):
            self.logger.error(f"Results file not found: {results_csv_path}. Cannot generate plots.")
            return
            
        try:
            results_df = pd.read_csv(results_csv_path)
        except Exception as e:
            self.logger.error(f"Failed to read results CSV {results_csv_path}: {e}. Cannot plot.")
            return
            
        if results_df.empty:
            self.logger.warning("Degradation results DataFrame is empty. Skipping plotting.")
            return
            
        # Check for required columns
        required_cols = ['model_type', 'epoch', f'val_{self.primary_metric}', 'degradation_from_peak', 'grad_norm', 'weight_change']
        missing_cols = [col for col in required_cols if col not in results_df.columns]
        if missing_cols:
            self.logger.error(f"Results CSV is missing required columns: {missing_cols}. Cannot plot.")
            return
            
        model_types = results_df['model_type'].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_types)))
        color_map = {mtype: colors[i] for i, mtype in enumerate(model_types)}
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
        fig.suptitle(f"Degradation Study Analysis - {self.dataset_name}", fontsize=16, y=0.99)

        # Plot 1: Validation Metric
        ax = axes[0]
        for model_type in model_types:
            model_data = results_df[results_df['model_type'] == model_type]
            if not model_data.empty:
                ax.plot(model_data['epoch'], model_data[f'val_{self.primary_metric}'], 
                        label=model_type, color=color_map[model_type], lw=1.5, alpha=0.9)
                # Mark peak 
                peak_idx = model_data[f'val_{self.primary_metric}'].idxmax() if self.higher_is_better else model_data[f'val_{self.primary_metric}'].idxmin()
                if pd.notna(peak_idx) and peak_idx in model_data.index:
                     peak_epoch = model_data.loc[peak_idx, 'epoch']
                     peak_val = model_data.loc[peak_idx, f'val_{self.primary_metric}']
                     ax.scatter(peak_epoch, peak_val, color=color_map[model_type], s=100, marker='*', 
                                label=f'{model_type} Peak Ep {int(peak_epoch)}' if model_type not in ax.get_legend_handles_labels()[1] else None, 
                                zorder=5)
        ax.set_ylabel(f"Validation {self.primary_metric.upper()}")
        ax.set_title("Validation Performance vs Epoch")
        ax.grid(True, alpha=0.5)
        ax.legend()

        # Plot 2: Degradation from Peak
        ax = axes[1]
        for model_type in model_types:
            model_data = results_df[results_df['model_type'] == model_type]
            if not model_data.empty:
                # Fill NaN degradation with 0, especially before the peak
                ax.plot(model_data['epoch'], model_data['degradation_from_peak'].fillna(0), 
                        label=model_type, color=color_map[model_type], lw=1.5, alpha=0.9)
        ax.set_ylabel("Degradation from Peak")
        ax.set_title("Performance Degradation vs Epoch")
        ax.grid(True, alpha=0.5)
        ax.legend()
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0])) # Ensure y starts at or below 0
        
        # Plot 3: Gradient Norm
        ax = axes[2]
        for model_type in model_types:
            model_data = results_df[results_df['model_type'] == model_type]
            if not model_data.empty and model_data['grad_norm'].notna().any(): # Check if any data exists
                ax.plot(model_data['epoch'], model_data['grad_norm'], 
                        label=model_type, color=color_map[model_type], lw=1.5, alpha=0.9)
        ax.set_ylabel("Average Gradient Norm (L2)")
        ax.set_title("Gradient Norm vs Epoch")
        ax.grid(True, alpha=0.5)
        ax.legend()
        # Optional: Use log scale if values vary greatly
        # if results_df['grad_norm'].max() / results_df['grad_norm'].min() > 100: ax.set_yscale('log')
        
        # Plot 4: Weight Change
        ax = axes[3]
        for model_type in model_types:
            model_data = results_df[results_df['model_type'] == model_type]
            if not model_data.empty and model_data['weight_change'].notna().any():
                ax.plot(model_data['epoch'], model_data['weight_change'], 
                        label=model_type, color=color_map[model_type], lw=1.5, alpha=0.9)
        ax.set_ylabel("Weight Change Norm (L2)")
        ax.set_title("Weight Change vs Epoch")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.5)
        ax.legend()
        # Optional: Use log scale
        # if results_df['weight_change'].max() / results_df['weight_change'].min() > 100: ax.set_yscale('log')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout
        
        plot_filename = f'degradation_study_plots_{self.dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plot_path = os.path.join(plot_dir, plot_filename)
        try:
            plt.savefig(plot_path, bbox_inches='tight', dpi=150)
            self.logger.info(f"Degradation study plots saved to {plot_path}")
        except Exception as e:
            self.logger.error(f"Failed to save degradation plots: {e}")
        finally:
            plt.close(fig)
            gc.collect()

    def _create_mlp(self, layers: List[int], activation: str) -> nn.Sequential:
        """Helper to create a simple MLP.
        
        Args:
            layers: List of integers defining layer sizes (including input and output).
            activation: String name of the activation function (e.g., 'ReLU', 'GELU').
            
        Returns:
            An nn.Sequential MLP model.
        """
        net = nn.Sequential()
        try:
            activation_fn = getattr(nn, activation)
        except AttributeError:
            self.logger.error(f"Activation function '{activation}' not found in torch.nn. Defaulting to ReLU.")
            activation_fn = nn.ReLU
            
        self.logger.info(f"Creating MLP with layers: {layers} and activation: {activation_fn.__name__}")
        
        if len(layers) < 2:
            self.logger.error("MLP needs at least an input and output layer.")
            return None # Return None on error

        for i in range(len(layers) - 1):
            try:
                net.add_module(f"linear_{i}", nn.Linear(layers[i], layers[i+1]))
                if i < len(layers) - 2: # No activation after the output layer
                    net.add_module(f"activation_{i}", activation_fn())
            except Exception as e:
                 self.logger.error(f"Error adding layer {i} (Linear {layers[i]}->{layers[i+1]} or Activation {activation}) to MLP: {e}")
                 return None # Return None on error
                    
        return net # Return the constructed network

    # --- Plotting --- 
    def plot_results(self):
        """Plot comparison results based on task type and primary metric."""
        if self.results_df.empty: self.logger.warning("No results to plot."); return
        
        plot_dir = self.config.get('plotting', {}).get('plot_dir', self.results_dir)
        os.makedirs(plot_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        metric_col = f'val_{self.primary_metric}'
        metric_name = f"Validation {self.primary_metric.upper() if self.primary_metric == 'mse' else self.primary_metric.capitalize()}"
        loss_col = 'val_loss'
        
        # --- Plot 1: Primary Metric vs Epoch (Best Config per Method/Run Type) --- #
        plt.figure(figsize=(10, 6))
        has_epoch_data = False
        
        # Filter results to only include runs with epoch data (epoch != -1 or not NaN)
        epoch_results_df = self.results_df[self.results_df['epoch'].notna() & (self.results_df['epoch'] != -1)].copy()
        
        if not epoch_results_df.empty:
            group_cols = ['model_type', 'kan_opt_method']
            if epoch_results_df['kan_opt_method'].isnull().all() or (epoch_results_df['kan_opt_method'] == 'N/A').all():
                 group_cols.remove('kan_opt_method')

            # Find the best final metric achieved for each group combination *within epoch data*
            if self.higher_is_better:
                best_epoch_final_metrics = epoch_results_df.loc[epoch_results_df.groupby(group_cols)[metric_col].idxmax()]
            else:
                best_epoch_final_metrics = epoch_results_df.loc[epoch_results_df.groupby(group_cols)[metric_col].idxmin()]
                
            plotted_labels = set()
            for idx, best_run_summary in best_epoch_final_metrics.iterrows():
                config_str = best_run_summary['config']
                model_type = best_run_summary['model_type']
                kan_opt_method = best_run_summary.get('kan_opt_method', 'N/A')
                
                plot_label = f"{model_type}"
                if pd.notna(kan_opt_method) and kan_opt_method != 'N/A':
                    plot_label += f" ({kan_opt_method})"
                
                if plot_label in plotted_labels: continue
                plotted_labels.add(plot_label)
                
                # Filter the original epoch DataFrame for all epochs of this specific best run configuration
                run_data = epoch_results_df[
                    (epoch_results_df['config'] == config_str) &
                    (epoch_results_df['model_type'] == model_type) &
                    (epoch_results_df['kan_opt_method'].fillna('N/A') == pd.Series([kan_opt_method]).fillna('N/A')[0])
                ].copy()
                run_data.dropna(subset=[metric_col, 'epoch'], inplace=True)

                if not run_data.empty:
                    plt.plot(run_data['epoch'], run_data[metric_col], label=f"{plot_label} (Best Cfg)", lw=2, alpha=0.8)
                    has_epoch_data = True # Mark that we plotted something
            
            if has_epoch_data:
                plt.title(f"{metric_name} vs Epoch (Best Config per NN Method/Run) - {self.dataset_name}")
                plt.xlabel("Epoch"); plt.ylabel(metric_name)
                if self.primary_metric == 'mse': plt.yscale('log')
                plt.grid(True, alpha=0.4); plt.legend(); plt.tight_layout()
                path = os.path.join(plot_dir, f'{self.dataset_name}_{self.primary_metric}_epoch_{ts}.png')
                try: plt.savefig(path, bbox_inches='tight'); self.logger.info(f"Saved plot: {path}")
                except Exception as e: self.logger.error(f"Save plot error: {e}")
                plt.close()
            else:
                 self.logger.info("No epoch-based results found to plot metric vs epoch."); plt.close()
        else:
            self.logger.info("No epoch-based results found in DataFrame to plot metric vs epoch."); plt.close()
             
        # --- Summary Plots (Using ALL results, including GBTs) --- #
        # Find the best overall result row for each unique run type (model_type + kan_opt_method)
        summary_group_cols = ['model_type', 'kan_opt_method']
        if self.results_df['kan_opt_method'].isnull().all() or (self.results_df['kan_opt_method'] == 'N/A').all():
             summary_group_cols.remove('kan_opt_method')
             
        if self.higher_is_better:
            best_summary_metrics = self.results_df.loc[self.results_df.groupby(summary_group_cols)[metric_col].idxmax()]
        else:
            best_summary_metrics = self.results_df.loc[self.results_df.groupby(summary_group_cols)[metric_col].idxmin()]
        
        summary_df = best_summary_metrics.copy()
        summary_df.dropna(subset=['param_count', metric_col], how='any', inplace=True) # Drop if no params OR no metric
        
        if summary_df.empty:
             self.logger.warning(f"No summary data found after dropping NaNs."); return
             
        summary_df['plot_label'] = summary_df.apply(lambda row: f"{row['model_type']}" + (f" ({row['kan_opt_method']})" if pd.notna(row['kan_opt_method']) and row['kan_opt_method'] != 'N/A' else ""), axis=1)
        methods = summary_df['plot_label'].tolist()
        metrics = summary_df[metric_col].tolist()
        params = summary_df['param_count'].tolist()
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
        
        # 2. Opt Time Plot (Only KAN)
        kan_summary_df = summary_df[(summary_df['model_type'] == 'FixedKAN') & summary_df['kan_opt_time'].notna() & (summary_df['kan_opt_time'] > 0)]
        if not kan_summary_df.empty:
            kan_methods = kan_summary_df['plot_label'].tolist()
            kan_times = kan_summary_df['kan_opt_time'].tolist()
            kan_colors = plt.cm.viridis(np.linspace(0, 1, len(kan_methods)))
            plt.figure(figsize=(max(8, len(kan_methods)*1.5), 5))
            bars = plt.bar(kan_methods, kan_times, color=kan_colors)
            for bar, t in zip(bars, kan_times): plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01, f'{t:.2f}s', ha='center', va='bottom', fontsize=9)
            plt.title(f'KAN Optimize Time (Best Config per Method) - {self.dataset_name}')
            plt.ylabel('Time (s)'); plt.xticks(rotation=30, ha='right'); plt.grid(axis='y', ls='--', alpha=0.6); plt.tight_layout()
            path = os.path.join(plot_dir, f'{self.dataset_name}_kan_opttimes_{ts}.png')
            try: plt.savefig(path, bbox_inches='tight'); self.logger.info(f"Saved plot: {path}")
            except Exception as e: self.logger.error(f"Save plot error: {e}")
            plt.close()
        else:
             self.logger.info("No KAN optimize times found to plot.")

        # 3. Final Performance Plot (All Methods)
        if not methods: self.logger.warning("No methods found for final performance plot."); return
        plt.figure(figsize=(max(8, len(methods)*1.5), 5));
        bars = plt.bar(methods, metrics, color=colors)
        for bar, m in zip(bars, metrics): plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01, f'{m:.4f}', ha='center', va='bottom', fontsize=9)
        plt.title(f'Final Best {metric_name} - {self.dataset_name}'); plt.ylabel(metric_name)
        plt.xticks(rotation=30, ha='right')
        min_m = min(metrics) if metrics else 0
        bot_lim = min(0, min_m - abs(min_m*0.1)) if self.primary_metric == 'r2' else 0
        plt.ylim(bottom=bot_lim)
        if self.primary_metric == 'accuracy': plt.ylim(top=max(1.0, max(metrics)*1.05) if metrics else 1.05)
        plt.grid(axis='y', ls='--', alpha=0.6); plt.tight_layout()
        path = os.path.join(plot_dir, f'{self.dataset_name}_final_{self.primary_metric}_{ts}.png')
        try: plt.savefig(path, bbox_inches='tight'); self.logger.info(f"Saved plot: {path}")
        except Exception as e: self.logger.error(f"Save plot error: {e}")
        plt.close()
        
        # 4. Performance vs Parameter Count (Filter out GBTs with placeholder param count)
        nn_summary_df = summary_df[summary_df['param_count'] != -1].copy()
        if not nn_summary_df.empty:
            plt.figure(figsize=(10, 6))
            unique_method_types = nn_summary_df['model_type'].unique()
            scatter_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_method_types)))
            color_map = {mtype: scatter_colors[i] for i, mtype in enumerate(unique_method_types)}
            
            plotted_labels_scatter = set() # Avoid duplicate labels in scatter
            for i, row in nn_summary_df.iterrows():
                label = row['plot_label']
                if label not in plotted_labels_scatter:
                     plt.scatter(row['param_count'], row[metric_col], 
                                 label=label, 
                                 color=color_map[row['model_type']],
                                 s=80, alpha=0.7)
                     plotted_labels_scatter.add(label)
                else: # Plot subsequent points without label
                    plt.scatter(row['param_count'], row[metric_col], 
                                 color=color_map[row['model_type']],
                                 s=80, alpha=0.7)
            
            # Create legend based on unique labels plotted
            handles, labels = plt.gca().get_legend_handles_labels()
            # Filter unique labels/handles if needed, but direct use might be fine
            plt.legend(title="Model/Run Types", handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.xlabel("Number of Parameters (Log Scale)")
            plt.ylabel(f"Best Validation {self.primary_metric.upper()}")
            plt.xscale('log')
            if self.primary_metric == 'mse': plt.yscale('log')
            plt.title(f"Performance vs Parameter Count (NNs) - {self.dataset_name}")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for external legend
            path = os.path.join(plot_dir, f'{self.dataset_name}_perf_vs_params_{ts}.png')
            try: plt.savefig(path, bbox_inches='tight'); self.logger.info(f"Saved plot: {path}")
            except Exception as e: self.logger.error(f"Save plot error: {e}")
            plt.close()
        else:
            self.logger.info("No NN results with valid parameter counts found to plot performance vs params.")

    def _train_evaluate_gbt(self, model_instance, model_name: str, params: Dict):
        """Trains and evaluates a Gradient Boosting Tree model (XGBoost, LightGBM).
        
        Appends a single row with final validation performance to self.results_df.
        Assumes model_instance is a scikit-learn compatible classifier/regressor.
        """
        self.logger.info(f"Training and evaluating {model_name} with params: {params}")
        start_time = time.time()

        # Convert data to NumPy (if not already)
        # GBTs typically work best with original data before scaling for NNs, 
        # but for fair comparison let's use the scaled data prepared for NNs.
        # Note: This might disadvantage GBTs slightly.
        if isinstance(self.x_train, torch.Tensor):
            X_train_np = self.x_train.cpu().numpy()
            y_train_np = self.y_train.cpu().numpy()
            X_val_np = self.x_val.cpu().numpy()
            y_val_np = self.y_val.cpu().numpy()
            # Also convert weights if they exist and are tensors
            w_train_np = self.w_train.cpu().numpy() if hasattr(self, 'w_train') and isinstance(self.w_train, torch.Tensor) else None
            w_val_np = self.w_val.cpu().numpy() if hasattr(self, 'w_val') and isinstance(self.w_val, torch.Tensor) else None
        else: # Assuming already NumPy
            X_train_np = self.x_train
            y_train_np = self.y_train
            X_val_np = self.x_val
            y_val_np = self.y_val
            w_train_np = self.w_train if hasattr(self, 'w_train') else None # Assume numpy if exists
            w_val_np = self.w_val if hasattr(self, 'w_val') else None
            
        # Squeeze target if necessary (e.g., if it's [N, 1])
        if y_train_np.ndim > 1 and y_train_np.shape[1] == 1:
             y_train_np = y_train_np.squeeze()
        if y_val_np.ndim > 1 and y_val_np.shape[1] == 1:
             y_val_np = y_val_np.squeeze()
        # Squeeze weights if necessary
        if w_train_np is not None and w_train_np.ndim > 1:
            w_train_np = w_train_np.squeeze()
        if w_val_np is not None and w_val_np.ndim > 1:
            w_val_np = w_val_np.squeeze()

        try:
            # Set model parameters before fitting
            model_instance.set_params(**params)
            
            # Fit the model
            fit_params = {}
            callbacks = [] 
            eval_set = [(X_val_np, y_val_np)]
            
            # Prepare eval_set with weights if applicable and model supports it
            # Note: Check specific model docs if weight support in eval_set varies
            # For LGBM/XGBM, sample_weight in fit is usually separate from eval_set weights
            
            if isinstance(model_instance, (xgb.XGBClassifier, xgb.XGBRegressor)):
                 fit_params['eval_set'] = eval_set
                 fit_params['early_stopping_rounds'] = self.config.get('training',{}).get('gbt_early_stopping_rounds', 10)
            elif isinstance(model_instance, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
                 fit_params['eval_set'] = eval_set
                 stopping_rounds = self.config.get('training',{}).get('gbt_early_stopping_rounds', 10)
                 callbacks.append(early_stopping(stopping_rounds=stopping_rounds, verbose=False))
                 fit_params['callbacks'] = callbacks

            # Add sample weights to fit method if available and model accepts it
            if self.use_weights and w_train_np is not None and hasattr(model_instance, 'fit'):
                # Inspect fit signature to check for sample_weight argument
                import inspect
                fit_sig = inspect.signature(model_instance.fit)
                if 'sample_weight' in fit_sig.parameters:
                    fit_params['sample_weight'] = w_train_np
                    self.logger.debug(f"Passing sample_weight to {model_name}.fit()")
                else:
                    self.logger.warning(f"Model {model_name} does not accept 'sample_weight' in fit method. Weights not used for training.")
            
            model_instance.fit(X_train_np, y_train_np, **fit_params)
            train_time = time.time() - start_time
            self.logger.info(f"{model_name} fitting completed in {train_time:.2f}s")

            # Evaluate
            val_metric = None
            train_metric = None
            val_loss = np.nan 
            train_loss = np.nan

            if self.task_type == 'classification':
                y_pred_val = model_instance.predict(X_val_np)
                y_pred_train = model_instance.predict(X_train_np)
                val_metric = accuracy_score(y_val_np, y_pred_val)
                train_metric = accuracy_score(y_train_np, y_pred_train)
                try:
                    y_prob_val = model_instance.predict_proba(X_val_np)
                    y_prob_train = model_instance.predict_proba(X_train_np)
                    from sklearn.metrics import log_loss
                    val_loss = log_loss(y_val_np, y_prob_val)
                    train_loss = log_loss(y_train_np, y_prob_train)
                except Exception:
                    self.logger.debug(f"Could not calculate log_loss for {model_name}.")
                    
            else: # Regression
                y_pred_val = model_instance.predict(X_val_np)
                y_pred_train = model_instance.predict(X_train_np)
                
                # Calculate metrics, using weights if primary metric requires it
                if self.primary_metric == 'weighted_r2':
                     if w_val_np is not None and w_train_np is not None:
                         val_metric = sk_r2_score(y_val_np, y_pred_val, sample_weight=w_val_np)
                         train_metric = sk_r2_score(y_train_np, y_pred_train, sample_weight=w_train_np)
                         self.logger.info(f"Calculated weighted_r2 for {model_name}.")
                     else:
                         self.logger.warning(f"Weights not available for weighted_r2 calculation for {model_name}. Calculating standard r2 instead.")
                         val_metric = sk_r2_score(y_val_np, y_pred_val)
                         train_metric = sk_r2_score(y_train_np, y_pred_train)
                     # MSE is unweighted
                     val_loss = mean_squared_error(y_val_np, y_pred_val)
                     train_loss = mean_squared_error(y_train_np, y_pred_train)
                elif self.primary_metric == 'r2':
                      val_metric = sk_r2_score(y_val_np, y_pred_val)
                      train_metric = sk_r2_score(y_train_np, y_pred_train)
                      val_loss = mean_squared_error(y_val_np, y_pred_val)
                      train_loss = mean_squared_error(y_train_np, y_pred_train)
                else: # Assume primary metric is mse
                     val_metric = mean_squared_error(y_val_np, y_pred_val)
                     train_metric = mean_squared_error(y_train_np, y_pred_train)
                     val_loss = val_metric
                     train_loss = train_metric

            self.logger.info(f"{model_name} Eval: Val {self.primary_metric}={val_metric:.4f}, Train {self.primary_metric}={train_metric:.4f}")

            # Append result to DataFrame
            row_data = {
                'model_type': model_name,
                'kan_opt_method': 'N/A',
                'kan_opt_time': 0.0,
                'param_count': -1, # Parameter count is not straightforward for GBTs
                f'val_{self.primary_metric}': val_metric,
                f'train_{self.primary_metric}': train_metric, # Include train metric
                'val_loss': val_loss, # Include loss
                'train_loss': train_loss,
                'epoch': -1, # Mark as non-epoch-based
                'fit_time': train_time, # Store fit time
                'config': str(params), # Store hyperparams used
                 **params # Also store params flattened
            }
            new_row = pd.DataFrame([row_data])
            if self.results_df.empty:
                 self.results_df = new_row
            else:
                 # Ensure all columns exist before concat
                 shared_cols = self.results_df.columns.intersection(new_row.columns)
                 missing_in_df = new_row.columns.difference(self.results_df.columns)
                 missing_in_row = self.results_df.columns.difference(new_row.columns)
                 for col in missing_in_df: self.results_df[col] = pd.NA
                 for col in missing_in_row: new_row[col] = pd.NA
                 # Align columns before concat
                 self.results_df = pd.concat([self.results_df.reindex(columns=new_row.columns, fill_value=pd.NA),
                                            new_row], ignore_index=True)

        except Exception as e:
            self.logger.error(f"Error during training/evaluation for {model_name} with params {params}: {e}", exc_info=True)
            # Optionally append a row indicating failure?

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