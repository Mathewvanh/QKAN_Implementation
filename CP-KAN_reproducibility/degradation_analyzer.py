import math
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
from typing import Dict

# Assuming these local imports are accessible from the main execution context
from data_pipeline_js_config import DataConfig # May need adjustment based on final structure
from data_pipeline import DataPipeline        # May need adjustment based on final structure
from CP_KAN import FixedKANConfig, FixedKAN     # May need adjustment based on final structure

# --- Helper Functions ---

def count_parameters(module: nn.Module) -> int:
    """Count trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    """Compute weighted R² score."""
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(w, torch.Tensor):
        w = w.cpu().numpy()
        
    # Squeeze if necessary
    if y_true.ndim > 1 and y_true.shape[1] == 1: y_true = y_true.squeeze()
    if y_pred.ndim > 1 and y_pred.shape[1] == 1: y_pred = y_pred.squeeze()
    if w.ndim > 1 and w.shape[1] == 1: w = w.squeeze()
        
    numerator = np.sum(w * (y_true - y_pred)**2)
    denominator = np.sum(w * (y_true**2)) # Assuming y_true is already centered or R2 definition doesn't require mean subtraction in denom
    
    # Epsilon handling for stability, common in R2 calculations
    epsilon = 1e-9 
    if denominator < epsilon:
        # If variance of true values (weighted) is near zero, R2 is ill-defined. 
        # Return 0 or handle as appropriate. If numerator is also ~0, could be 1.0.
        # Let's return 0 for now, common practice.
        return 0.0 
        
    r2 = 1.0 - (numerator / (denominator + epsilon)) # Add epsilon to denominator too
    
    # Clip R2 score (optional, but prevents extreme negative values in edge cases)
    # return max(0.0, r2) # If you only want R2 >= 0
    return float(r2) # Return as float

def build_mlp(input_dim: int, hidden_size: int, depth: int, output_dim: int = 1, dropout_rate: float = 0.1) -> nn.Module:
    """Build MLP with specified depth, dropout, and batch normalization."""
    layers = []
    curr_dim = input_dim
    for _ in range(depth):
        layers.extend([
            nn.Linear(curr_dim, hidden_size),
            nn.BatchNorm1d(hidden_size), # BatchNorm before activation is common
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        curr_dim = hidden_size
    layers.append(nn.Linear(curr_dim, output_dim)) # Final layer to output_dim
    return nn.Sequential(*layers)


# --- Main Analyzer Class ---

class DegradationAnalyzer:
    def __init__(self, config: Dict):
        """
        Initialize the analyzer with configuration.

        Args:
            config (dict): Dictionary containing experiment parameters.
                           Expected keys: data_config, kan_config, mlp_config,
                           training_config, output_dir, logging_level etc.
        """
        self.config = config
        # Assign output_dir and create directories BEFORE setting up logger
        self.output_dir = config.get('output_dir', './results_degradation_analysis')
        self.model_save_dir = os.path.join(self.output_dir, 'models')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # Now setup logger, which might need the output_dir
        self.logger = self._setup_logger()
        
        # Initialize other attributes
        self.x_train, self.y_train, self.w_train = None, None, None
        self.x_val, self.y_val, self.w_val = None, None, None
        self.input_dim = None
        self.results_df = pd.DataFrame(columns=[
            'model_type', 'learning_rate', 'epoch', 'train_r2', 'val_r2',
            'grad_norm', 'weight_change', 'degradation_from_peak'
        ])

    def _setup_logger(self):
        """Sets up the logger for the analyzer."""
        logger = logging.getLogger("DegradationAnalyzer")
        level_str = self.config.get('logging_level', 'INFO').upper()
        level = getattr(logging, level_str, logging.INFO)
        logger.setLevel(level)
        
        # Avoid adding multiple handlers if logger already configured
        if not logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            
        # Optionally add file handler
        log_file = self.config.get('log_file')
        if log_file:
            fh = logging.FileHandler(os.path.join(self.output_dir, log_file))
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
        return logger

    def _load_data(self):
        """Loads and preprocesses data needed for the analysis."""
        self.logger.info("Loading and preprocessing data...")
        # Access dataset config keys directly from self.config
        data_config_dict = self.config.copy() # Use a copy to modify

        # <<< Expansion logic >>>
        if data_config_dict.get('feature_cols') == 'auto':
            self.logger.info("Auto-generating Jane Street feature columns for degradation analysis.")
            num_features = data_config_dict.get('num_features', 79)
            data_config_dict['feature_cols'] = [f'feature_{i:02d}' for i in range(num_features)]
            self.logger.debug(f"Expanded feature_cols: {data_config_dict['feature_cols'][:5]}...")

        # Filter the dictionary to only contain keys expected by DataConfig
        expected_keys = DataConfig.__annotations__.keys()
        filtered_data_config_dict = {k: v for k, v in data_config_dict.items() if k in expected_keys}

        # Now create DataConfig with the filtered dictionary
        try:
            js_data_cfg = DataConfig.from_dict(filtered_data_config_dict)
        except ValueError as e:
            self.logger.error(f"Error creating DataConfig: {e}")
            raise
        except KeyError as e:
             self.logger.error(f"Missing key in dataset config for DataConfig: {e}")
             raise

        pipeline = DataPipeline(js_data_cfg, self.logger)
        train_df, train_target, train_weight, val_df, val_target, val_weight = pipeline.load_and_preprocess_data()

        # Convert to torch tensors
        self.x_train = torch.tensor(train_df.to_numpy(dtype=np.float32), dtype=torch.float32)
        self.y_train = torch.tensor(train_target.to_numpy(dtype=np.float32), dtype=torch.float32).squeeze() # Squeeze here
        self.w_train = torch.tensor(train_weight.to_numpy(dtype=np.float32), dtype=torch.float32).squeeze() # Squeeze here

        self.x_val = torch.tensor(val_df.to_numpy(dtype=np.float32), dtype=torch.float32)
        self.y_val = torch.tensor(val_target.to_numpy(dtype=np.float32), dtype=torch.float32).squeeze() # Squeeze here
        self.w_val = torch.tensor(val_weight.to_numpy(dtype=np.float32), dtype=torch.float32).squeeze() # Squeeze here

        self.input_dim = self.x_train.shape[1]
        self.logger.info(f"Data loaded. Input dim: {self.input_dim}, Train samples: {len(self.x_train)}, Val samples: {len(self.x_val)}")
        
        # Basic check for NaN/Inf after loading (optional but good practice)
        for name, tensor in [('x_train', self.x_train), ('y_train', self.y_train), ('w_train', self.w_train),
                             ('x_val', self.x_val), ('y_val', self.y_val), ('w_val', self.w_val)]:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                 self.logger.warning(f"NaN or Inf detected in {name} tensor after loading/conversion.")


    def _train_and_track_metrics(self, model, optimizer, model_type: str, 
                               learning_rate: float, num_epochs: int, 
                               batch_size: int, early_stopping_patience: int = 50, 
                               early_stopping_threshold_ratio: float = 0.3):
        """
        Trains a model, tracks metrics epoch by epoch, and handles early stopping.

        Args:
            model (nn.Module): The model to train.
            optimizer: The optimizer.
            model_type (str): 'KAN' or 'MLP'.
            learning_rate (float): Learning rate used (for logging/saving).
            num_epochs (int): Total epochs to train.
            batch_size (int): Mini-batch size.
            early_stopping_patience (int): How many epochs to wait after peak before stopping due to degradation.
            early_stopping_threshold_ratio (float): Stop if val_r2 drops below peak * threshold_ratio.

        Returns:
            tuple: (metrics_dict, best_val_r2, peak_epoch)
        """
        # Ensure model is on the correct device (if using GPU in the future)
        # device = next(model.parameters()).device 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        self.x_train, self.y_train, self.w_train = self.x_train.to(device), self.y_train.to(device), self.w_train.to(device)
        self.x_val, self.y_val, self.w_val = self.x_val.to(device), self.y_val.to(device), self.w_val.to(device)
        
        metrics = defaultdict(list)
        best_val_r2 = float('-inf')
        peak_epoch = 0
        epochs_since_peak = 0
        
        # Store initial weights (on CPU to save GPU memory if needed)
        prev_weights = {name: param.clone().detach().cpu() 
                       for name, param in model.named_parameters()}

        self.logger.info(f"Starting training for {model_type} with LR={learning_rate:.1e}")
        training_start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            model.train()
            
            epoch_loss = 0.0
            epoch_grad_norm = 0.0
            epoch_weight_change = 0.0
            
            # Mini-batch training
            n_batches = math.ceil(len(self.x_train) / batch_size)
            permutation = torch.randperm(self.x_train.size(0))

            for i in range(n_batches):
                indices = permutation[i * batch_size:(i + 1) * batch_size]
                x_batch, y_batch, w_batch = self.x_train[indices], self.y_train[indices], self.w_train[indices]
                
                optimizer.zero_grad()
                # Ensure output is squeezed correctly for loss calculation
                y_pred = model(x_batch).squeeze() 
                
                # Weighted MSE loss
                numerator = torch.sum(w_batch * (y_batch - y_pred)**2)
                denominator = torch.sum(w_batch)
                loss = numerator / (denominator + 1e-12) # Add epsilon for stability
                
                # Check for NaN loss
                if torch.isnan(loss):
                    self.logger.error(f"NaN loss detected at epoch {epoch}, batch {i}. Stopping training for this model.")
                    # Fill remaining epochs with NaN or last valid value? Let's use NaN.
                    for fill_epoch in range(epoch, num_epochs):
                         metrics['epoch'].append(fill_epoch)
                         metrics['train_r2'].append(np.nan)
                         metrics['val_r2'].append(np.nan)
                         metrics['grad_norm'].append(np.nan)
                         metrics['weight_change'].append(np.nan)
                         new_row = pd.DataFrame({'model_type': [model_type], 'learning_rate': [learning_rate], 'epoch': [fill_epoch], 
                                                 'train_r2': [np.nan], 'val_r2': [np.nan], 'grad_norm': [np.nan], 
                                                 'weight_change': [np.nan], 'degradation_from_peak': [np.nan]})
                         self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
                    return metrics, best_val_r2, peak_epoch # Return immediately
                
                loss.backward()
                
                # Optional: Gradient Clipping (useful for stability)
                # clip_value = self.config.get('training_config', {}).get('grad_clip_value', None)
                # if clip_value:
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

                # Calculate gradient norm before optimizer step clears gradients
                batch_grad_norm = torch.tensor(0.0, device=device)
                for p in model.parameters():
                     if p.grad is not None:
                         param_norm = p.grad.detach().data.norm(2)
                         batch_grad_norm += param_norm.item() ** 2
                batch_grad_norm = batch_grad_norm ** 0.5
                epoch_grad_norm += batch_grad_norm.item()
                
                optimizer.step()
                epoch_loss += loss.item() * len(x_batch) # Accumulate weighted loss

            avg_epoch_loss = epoch_loss / len(self.x_train)
            avg_epoch_grad_norm = epoch_grad_norm / n_batches
            
            # Calculate weight changes (compare current weights on device to previous on CPU)
            current_weight_change = 0.0
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in prev_weights:
                         # Move prev_weight to current device for comparison
                         weight_diff = param.data - prev_weights[name].to(device)
                         current_weight_change += torch.norm(weight_diff).item()
                         # Update prev_weights (move back to CPU)
                         prev_weights[name] = param.clone().detach().cpu() 
                    else:
                        # Handle newly added parameters if any (shouldn't happen here)
                         prev_weights[name] = param.clone().detach().cpu()
            epoch_weight_change = current_weight_change

            # --- Evaluation Phase ---
            model.eval()
            train_r2, val_r2 = np.nan, np.nan # Default to NaN
            with torch.no_grad():
                try:
                    # Use full dataset for R2 calculation
                    # Process in batches if dataset is very large to avoid OOM
                    eval_batch_size = self.config.get('training_config',{}).get('eval_batch_size', 2048)
                    
                    # Train R²
                    y_pred_train_list = []
                    for i in range(0, len(self.x_train), eval_batch_size):
                         x_eval_batch = self.x_train[i:i+eval_batch_size]
                         y_pred_batch = model(x_eval_batch).squeeze()
                         y_pred_train_list.append(y_pred_batch)
                    y_pred_train = torch.cat(y_pred_train_list).cpu() # Move to CPU for numpy conversion
                    
                    train_r2 = weighted_r2(
                        self.y_train.cpu(), # Ensure y_train is on CPU
                        y_pred_train,
                        self.w_train.cpu()  # Ensure w_train is on CPU
                    )
                    
                    # Val R²
                    y_pred_val_list = []
                    for i in range(0, len(self.x_val), eval_batch_size):
                        x_eval_batch = self.x_val[i:i+eval_batch_size]
                        y_pred_batch = model(x_eval_batch).squeeze()
                        y_pred_val_list.append(y_pred_batch)
                    y_pred_val = torch.cat(y_pred_val_list).cpu() # Move to CPU for numpy conversion
                    
                    val_r2 = weighted_r2(
                        self.y_val.cpu(),  # Ensure y_val is on CPU
                        y_pred_val,
                        self.w_val.cpu()   # Ensure w_val is on CPU
                    )
                except Exception as e:
                    self.logger.error(f"Error during R2 calculation at epoch {epoch}: {e}")
                    # Keep train_r2 and val_r2 as np.nan

            # Track peak performance
            current_degradation = np.nan
            if not np.isnan(val_r2): # Only update peak if val_r2 is valid
                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    peak_epoch = epoch
                    epochs_since_peak = 0
                    # Save the best model state
                    try:
                        torch.save({
                            'model_state': model.state_dict(),
                            'epoch': epoch,
                            'val_r2': val_r2,
                            'model_type': model_type,
                            'learning_rate': learning_rate
                        }, os.path.join(self.model_save_dir, f'{model_type}_lr{learning_rate:.1e}_best.pth'))
                        self.logger.debug(f"Saved best model for {model_type} at epoch {epoch} with Val R2: {val_r2:.4f}")
                    except Exception as e:
                        self.logger.error(f"Failed to save best model state: {e}")
                else:
                    epochs_since_peak += 1
                
                # Calculate degradation only after the peak has been established
                if epoch > peak_epoch:
                     current_degradation = best_val_r2 - val_r2
            
            # Store metrics
            metrics['epoch'].append(epoch)
            metrics['train_r2'].append(train_r2)
            metrics['val_r2'].append(val_r2)
            metrics['grad_norm'].append(avg_epoch_grad_norm)
            metrics['weight_change'].append(epoch_weight_change)
            metrics['degradation_from_peak'].append(current_degradation)
            
            # Add row to DataFrame immediately for real-time saving
            new_row = pd.DataFrame({
                'model_type': [model_type], 'learning_rate': [learning_rate], 'epoch': [epoch], 
                'train_r2': [train_r2], 'val_r2': [val_r2], 'grad_norm': [avg_epoch_grad_norm], 
                'weight_change': [epoch_weight_change], 'degradation_from_peak': [current_degradation]
            })
            self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
            
            # Save results frequently
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                try:
                    results_path = os.path.join(self.output_dir, 'degradation_metrics.csv')
                    self.results_df.to_csv(results_path, index=False, float_format='%.6f')
                except Exception as e:
                    self.logger.error(f"Failed to save results CSV at epoch {epoch}: {e}")

            epoch_duration = time.time() - epoch_start_time
            if epoch % 10 == 0: # Log every 10 epochs
                self.logger.info(f"[{model_type}] LR={learning_rate:.1e} | Epoch {epoch}/{num_epochs} | "
                                 f"Loss={avg_epoch_loss:.4f} | Train R²={train_r2:.4f} | Val R²={val_r2:.4f} | "
                                 f"GradNorm={avg_epoch_grad_norm:.2f} | WgtChg={epoch_weight_change:.2f} | "
                                 f"Time={epoch_duration:.2f}s")
                
            # Early stopping check based on severe degradation
            stop_training = False
            if not np.isnan(val_r2) and not np.isnan(best_val_r2) and best_val_r2 != float('-inf'): # Ensure valid values
                 if val_r2 < best_val_r2 * early_stopping_threshold_ratio and epochs_since_peak >= early_stopping_patience:
                     self.logger.warning(f"EARLY STOPPING: Severe degradation detected for {model_type} (LR={learning_rate:.1e}) at epoch {epoch}. "
                                         f"Val R² ({val_r2:.4f}) dropped below {early_stopping_threshold_ratio*100:.0f}% of peak ({best_val_r2:.4f}) "
                                         f"after {epochs_since_peak} epochs.")
                     stop_training = True
                     
            if stop_training:
                # Fill remaining epochs in results_df to keep consistent plot lengths
                last_metrics = metrics # Use the metrics dict from the last successful epoch
                if len(last_metrics['epoch']) > 0:
                    last_epoch_idx = -1
                    for fill_epoch in range(epoch + 1, num_epochs):
                        fill_row = pd.DataFrame({
                            'model_type': [model_type], 'learning_rate': [learning_rate], 'epoch': [fill_epoch], 
                            'train_r2': [last_metrics['train_r2'][last_epoch_idx]], 
                            'val_r2': [last_metrics['val_r2'][last_epoch_idx]], 
                            'grad_norm': [last_metrics['grad_norm'][last_epoch_idx]], 
                            'weight_change': [last_metrics['weight_change'][last_epoch_idx]], 
                            'degradation_from_peak': [last_metrics['degradation_from_peak'][last_epoch_idx]] 
                        })
                        self.results_df = pd.concat([self.results_df, fill_row], ignore_index=True)
                        # Also add to metrics dict if needed later, though df is primary
                        metrics['epoch'].append(fill_epoch)
                        metrics['train_r2'].append(last_metrics['train_r2'][last_epoch_idx])
                        metrics['val_r2'].append(last_metrics['val_r2'][last_epoch_idx])
                        metrics['grad_norm'].append(last_metrics['grad_norm'][last_epoch_idx])
                        metrics['weight_change'].append(last_metrics['weight_change'][last_epoch_idx])
                        metrics['degradation_from_peak'].append(last_metrics['degradation_from_peak'][last_epoch_idx])
                break # Exit the training loop

        total_training_time = time.time() - training_start_time
        self.logger.info(f"Finished training for {model_type} with LR={learning_rate:.1e}. "
                         f"Peak Val R²={best_val_r2:.4f} at epoch {peak_epoch}. Total time={total_training_time:.2f}s")
        
        # Final save of results
        try:
             results_path = os.path.join(self.output_dir, 'degradation_metrics.csv')
             self.results_df.to_csv(results_path, index=False, float_format='%.6f')
        except Exception as e:
             self.logger.error(f"Failed to save final results CSV: {e}")
             
        return metrics, best_val_r2, peak_epoch

    def run_analysis(self):
        """
        Runs the full degradation analysis experiment:
        1. Loads data.
        2. Iterates through configured learning rates.
        3. Trains KAN and MLP for each LR, tracking metrics.
        4. Saves results and model checkpoints.
        5. Generates plots.
        """
        self._load_data() # Load data first

        training_config = self.config.get('training_config', {})
        learning_rates = training_config.get('learning_rates', [1e-3, 1e-4]) # Default LRs
        num_epochs = training_config.get('num_epochs', 200)
        batch_size = training_config.get('batch_size', 128)
        early_stop_patience = training_config.get('early_stopping_patience', 50)
        early_stop_threshold = training_config.get('early_stopping_threshold_ratio', 0.3)

        # Clear results from previous runs if analyzer object is reused
        self.results_df = self.results_df.iloc[0:0] 
        
        # KAN configuration from main config
        kan_params = self.config.get('kan_config', {})
        kan_config = FixedKANConfig(
            network_shape=[self.input_dim] + kan_params.get('hidden_layers', [20]) + [1], # Ensure input_dim is used
            max_degree=kan_params.get('max_degree', 5),
            complexity_weight=kan_params.get('complexity_weight', 0.0),
            trainable_coefficients=kan_params.get('trainable_coefficients', True),
            skip_qubo_for_hidden=kan_params.get('skip_qubo_for_hidden', False),
            default_hidden_degree=kan_params.get('default_hidden_degree', 5)
        )
        
        # MLP configuration from main config
        mlp_params = self.config.get('mlp_config', {})
        mlp_hidden_size = mlp_params.get('hidden_size', 24)
        mlp_depth = mlp_params.get('depth', 3)
        mlp_dropout = mlp_params.get('dropout_rate', 0.1)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {device}")
        
        for lr in learning_rates:
            self.logger.info(f"==== Starting Analysis for Learning Rate: {lr:.1e} ====")
            
            # --- Train KAN ---
            self.logger.info("--- Training KAN ---")
            try:
                qkan = FixedKAN(kan_config)
                # Run initial QUBO optimization step if needed (part of FixedKAN setup)
                qkan.optimize(self.x_train.to(device), self.y_train.to(device).unsqueeze(-1)) 
                qkan.to(device) # Ensure model is on the correct device
                
                # Define parameters to train (coefficients, biases)
                params_to_train = []
                for layer in qkan.layers:
                    params_to_train.extend([layer.combine_W, layer.combine_b])
                    for neuron in layer.neurons:
                        # Ensure neuron.w and neuron.b are Parameter tensors
                        if isinstance(neuron.w, nn.Parameter): params_to_train.append(neuron.w)
                        if isinstance(neuron.b, nn.Parameter): params_to_train.append(neuron.b)
                        # Or if they are simple tensors that require grad:
                        # elif isinstance(neuron.w, torch.Tensor) and neuron.w.requires_grad: params_to_train.append(neuron.w) # etc.

                if not params_to_train:
                    self.logger.warning(f"No trainable parameters found for KAN (LR={lr:.1e}). Skipping training.")
                else:
                    kan_optimizer = torch.optim.Adam(params_to_train, lr=lr)
                    kan_metrics, kan_best_r2, kan_peak_epoch = self._train_and_track_metrics(
                        qkan, kan_optimizer, 'KAN', lr, num_epochs, batch_size,
                        early_stop_patience, early_stop_threshold
                    )
                    self.logger.info(f"KAN (LR={lr:.1e}) finished. Peak Val R²: {kan_best_r2:.4f} at epoch {kan_peak_epoch}")
            except Exception as e:
                self.logger.exception(f"Error during KAN training for LR={lr:.1e}: {e}")
            finally:
                 # Cleanup KAN model and optimizer
                 if 'qkan' in locals(): del qkan
                 if 'kan_optimizer' in locals(): del kan_optimizer
                 if torch.cuda.is_available(): torch.cuda.empty_cache()
                 gc.collect()
            
            # --- Train MLP ---
            self.logger.info("--- Training MLP ---")
            try:
                mlp = build_mlp(self.input_dim, mlp_hidden_size, mlp_depth, dropout_rate=mlp_dropout)
                mlp.to(device) # Ensure model is on device
                mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
                
                mlp_metrics, mlp_best_r2, mlp_peak_epoch = self._train_and_track_metrics(
                    mlp, mlp_optimizer, 'MLP', lr, num_epochs, batch_size,
                    early_stop_patience, early_stop_threshold
                )
                self.logger.info(f"MLP (LR={lr:.1e}) finished. Peak Val R²: {mlp_best_r2:.4f} at epoch {mlp_peak_epoch}")

            except Exception as e:
                self.logger.exception(f"Error during MLP training for LR={lr:.1e}: {e}")
            finally:
                # Cleanup MLP model and optimizer
                if 'mlp' in locals(): del mlp
                if 'mlp_optimizer' in locals(): del mlp_optimizer
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                gc.collect()
        
        # --- Analysis Complete ---
        self.logger.info("Degradation analysis complete for all learning rates.")
        
        # Final save of comprehensive results
        final_results_path = os.path.join(self.output_dir, 'degradation_metrics_final.csv')
        try:
            self.results_df.to_csv(final_results_path, index=False, float_format='%.6f')
            self.logger.info(f"Final detailed results saved to {final_results_path}")
        except Exception as e:
            self.logger.error(f"Failed to save final detailed results: {e}")
            
        # Generate and save summary statistics
        self._generate_summary()

        # Generate plots
        self.plot_results()
        
    def _generate_summary(self):
        """Calculates and saves summary statistics from the results_df."""
        self.logger.info("Generating summary statistics...")
        summary_df = pd.DataFrame(columns=['model_type', 'learning_rate', 'peak_val_r2', 'peak_epoch', 
                                         'final_val_r2', 'max_degradation_val', 'avg_grad_norm', 'avg_weight_change'])
        
        if self.results_df.empty:
            self.logger.warning("Results DataFrame is empty. Cannot generate summary.")
            return
            
        learning_rates = sorted(self.results_df['learning_rate'].unique())
        model_types = sorted(self.results_df['model_type'].unique())

        for model_type in model_types:
            for lr in learning_rates:
                model_lr_data = self.results_df[(self.results_df['model_type'] == model_type) & 
                                                (self.results_df['learning_rate'] == lr)].copy() # Use copy to avoid SettingWithCopyWarning
                
                if not model_lr_data.empty:
                    # Find peak validation R2 and corresponding epoch
                    peak_idx = model_lr_data['val_r2'].idxmax()
                    peak_val_r2 = model_lr_data.loc[peak_idx, 'val_r2']
                    peak_epoch = model_lr_data.loc[peak_idx, 'epoch']
                    
                    # Final validation R2 (last recorded epoch)
                    final_val_r2 = model_lr_data['val_r2'].iloc[-1]
                    
                    # Max degradation (ensure it's calculated relative to the actual peak for this run)
                    model_lr_data['degradation_recalc'] = peak_val_r2 - model_lr_data['val_r2']
                    max_degradation_val = model_lr_data[model_lr_data['epoch'] > peak_epoch]['degradation_recalc'].max()
                    if pd.isna(max_degradation_val): max_degradation_val = 0.0 # Handle case where no degradation occurred after peak
                        
                    # Average gradient norm and weight change (consider filtering NaNs if they occur)
                    avg_grad_norm = model_lr_data['grad_norm'].mean(skipna=True)
                    avg_weight_change = model_lr_data['weight_change'].mean(skipna=True)

                    summary_row = pd.DataFrame([{
                        'model_type': model_type,
                        'learning_rate': lr,
                        'peak_val_r2': peak_val_r2,
                        'peak_epoch': peak_epoch,
                        'final_val_r2': final_val_r2,
                        'max_degradation_val': max_degradation_val,
                        'avg_grad_norm': avg_grad_norm,
                        'avg_weight_change': avg_weight_change
                    }])
                    summary_df = pd.concat([summary_df, summary_row], ignore_index=True)

        summary_path = os.path.join(self.output_dir, 'degradation_summary.csv')
        try:
            summary_df.to_csv(summary_path, index=False, float_format='%.6f')
            self.logger.info(f"Summary statistics saved to {summary_path}")
        except Exception as e:
            self.logger.error(f"Failed to save summary statistics: {e}")


    def plot_results(self):
        """Creates visualization of degradation patterns based on the results_df."""
        self.logger.info("Generating degradation plots...")
        results_path = os.path.join(self.output_dir, 'degradation_metrics_final.csv') # Use final results
        if not os.path.exists(results_path):
            self.logger.warning("Results file not found. Skipping plotting.")
            return
        
        try:
            results = pd.read_csv(results_path)
        except Exception as e:
             self.logger.error(f"Failed to read results file {results_path}: {e}. Skipping plotting.")
             return

        if results.empty:
            self.logger.warning("Results file is empty. Skipping plotting.")
            return
            
        learning_rates = sorted(results['learning_rate'].unique())
        num_lrs = len(learning_rates)
        if num_lrs == 0:
             self.logger.warning("No learning rates found in results. Skipping plotting.")
             return
             
        # Split plots for better readability if many LRs
        plots_per_file = self.config.get('plotting_config', {}).get('plots_per_file', 3) # e.g., 3 LR comparisons per plot file
        num_files = math.ceil(num_lrs / plots_per_file)
        
        # Get timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for file_idx in range(num_files):
            start_lr_idx = file_idx * plots_per_file
            end_lr_idx = min((file_idx + 1) * plots_per_file, num_lrs)
            current_lrs = learning_rates[start_lr_idx:end_lr_idx]
            num_plots_in_file = len(current_lrs)
            
            if num_plots_in_file == 0: continue

            fig = plt.figure(figsize=(15, 6 * num_plots_in_file)) # Adjusted height per subplot pair
            # Create a GridSpec with 2 rows per LR (performance + degradation)
            gs = plt.GridSpec(2 * num_plots_in_file, 1, height_ratios=[2, 1] * num_plots_in_file, hspace=0.4) 
            self.logger.info(f"Generating plot file {file_idx + 1}/{num_files} (LRs: {current_lrs})...")

            for i, lr in enumerate(current_lrs):
                lr_results = results[results['learning_rate'] == lr]
                if lr_results.empty: continue

                # --- Performance Plot (Validation R2 vs Epoch) ---
                ax1 = fig.add_subplot(gs[2*i]) # Top plot for this LR
                
                for model_type, color in [('KAN', 'blue'), ('MLP', 'red')]:
                    model_results = lr_results[lr_results['model_type'] == model_type]
                    if not model_results.empty:
                        peak_idx = model_results['val_r2'].idxmax()
                        peak_r2 = model_results.loc[peak_idx, 'val_r2']
                        peak_epoch = model_results.loc[peak_idx, 'epoch']
                        final_r2 = model_results['val_r2'].iloc[-1]
                        
                        # Calculate percentage degradation (handle near-zero peak R2)
                        if pd.notna(peak_r2) and abs(peak_r2) > 1e-9:
                            percent_deg = (peak_r2 - final_r2) / abs(peak_r2) * 100 if pd.notna(final_r2) else np.nan
                            label_deg = f"{percent_deg:.1f}%" if pd.notna(percent_deg) else "N/A"
                        else:
                            label_deg = "N/A" # R2 near zero, degradation % is meaningless

                        label = f'{model_type} (Peak: {peak_r2:.3f}, Deg: {label_deg})'
                        
                        # Plot R2 line
                        ax1.plot(model_results['epoch'], model_results['val_r2'], 
                                 label=label, color=color, linewidth=1.5, alpha=0.8)
                                 
                        # Mark Peak R2
                        if pd.notna(peak_r2) and pd.notna(peak_epoch):
                             ax1.scatter(peak_epoch, peak_r2, color=color, s=80, marker='*', 
                                         label=f'{model_type} Peak Epoch {int(peak_epoch)}' if i == 0 else None, # Avoid duplicate legend items
                                         zorder=5) # Ensure star is on top

                        # Indicate Early Stopping if it happened (check if max epoch < expected)
                        max_epoch_expected = results['epoch'].max() # Overall max epoch
                        if model_results['epoch'].max() < max_epoch_expected:
                             stop_epoch = model_results['epoch'].max()
                             ax1.axvline(x=stop_epoch, color=color, linestyle=':', alpha=0.5, linewidth=1)
                             # Add text label for early stop - position carefully
                             y_range = ax1.get_ylim()
                             y_pos = y_range[0] + 0.1 * (y_range[1] - y_range[0]) # Adjust y position
                             ax1.text(stop_epoch + 2, y_pos, f'{model_type} Stop', color=color, alpha=0.7, fontsize=9,
                                      rotation=90, verticalalignment='bottom')

                ax1.set_title(f'Validation R² vs Epoch (Learning Rate: {lr:.1e})')
                ax1.set_ylabel('Validation R²')
                ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
                ax1.legend(fontsize=9, loc='best')
                # ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # Remove x-axis labels/ticks for top plot

                # --- Degradation Plot (Degradation vs Epoch) ---
                ax2 = fig.add_subplot(gs[2*i + 1], sharex=ax1) # Bottom plot, share x-axis
                
                for model_type, color in [('KAN', 'blue'), ('MLP', 'red')]:
                     model_results = lr_results[lr_results['model_type'] == model_type]
                     if not model_results.empty:
                          # Plot degradation (use the column directly, handle NaNs)
                          ax2.plot(model_results['epoch'], model_results['degradation_from_peak'].fillna(0), # Fill NaN with 0 for plotting
                                   label=f'{model_type}', color=color, linewidth=1.5, alpha=0.8)

                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Degradation from Peak R²')
                ax2.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
                ax2.legend(fontsize=9, loc='best')
                # Ensure y-axis starts at or below 0 for degradation plot
                ax2.set_ylim(bottom=min(0, ax2.get_ylim()[0])) 


            plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
            # Add a main title for the figure
            fig.suptitle(f'Model Degradation Analysis - Part {file_idx + 1}/{num_files}', fontsize=16, y=0.99) 
            
            plot_filename = f'degradation_comparison_{timestamp}_part{file_idx + 1}.png'
            plot_path = os.path.join(self.output_dir, plot_filename)
            
            try:
                 plt.savefig(plot_path, bbox_inches='tight', dpi=150) # Use bbox_inches and decent dpi
                 self.logger.info(f"Degradation comparison plot saved to {plot_path}")
            except Exception as e:
                 self.logger.error(f"Failed to save plot {plot_path}: {e}")
            finally:
                 plt.close(fig) # Close the figure explicitly to free memory
                 gc.collect()

        self.logger.info("Finished generating degradation plots.")

# Example usage (if you want to run this file directly for testing)
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     
#     # --- Define Configuration ---
#     test_config = {
#         'output_dir': './results_degradation_analysis_test',
#         'logging_level': 'INFO',
#         'log_file': 'analyzer_test.log',
#         'data_config': {
#              # Use absolute or relative paths carefully
#             'data_path': "~/Interning/Kaggle/jane_street_kaggle/jane-street-real-time-market-data-forecasting/train.parquet/", 
#             'n_rows': 50000, # Use fewer rows for quick testing
#             'train_ratio': 0.7,
#             'feature_cols': [f'feature_{i:02d}' for i in range(79)],
#             'target_col': "responder_6",
#             'weight_col': "weight",
#             'date_col': "date_id"
#         },
#         'kan_config': {
#             'hidden_layers': [10], # Smaller KAN for testing
#             'max_degree': 3,
#              # Add other FixedKANConfig params as needed
#         },
#         'mlp_config': {
#             'hidden_size': 16, # Smaller MLP
#             'depth': 2,
#             'dropout_rate': 0.1
#         },
#         'training_config': {
#             'learning_rates': [1e-3, 1e-4], # Test a couple of LRs
#             'num_epochs': 30, # Fewer epochs for testing
#             'batch_size': 64,
#             'eval_batch_size': 512,
#             'early_stopping_patience': 10,
#             'early_stopping_threshold_ratio': 0.5
#         },
#         'plotting_config': {
#              'plots_per_file': 2 
#         }
#     }
#
#     # --- Run Analysis ---
#     analyzer = DegradationAnalyzer(test_config)
#     analyzer.run_analysis()
#     print("Degradation analysis test run complete.") 