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
    # Class variable for results path
    results_path = 'results_js/kan_vs_mlp_metrics.csv'

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

        # Load existing results or create new DataFrame
        if os.path.exists(self.results_path):
            self.results_df = pd.read_csv(self.results_path)
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
                'model_type', 'depth', 'epoch', 'train_r2', 'val_r2', 'param_count'
            ])

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
        num_epochs = 50  # Reduced from 500
        lr = 1e-4
        
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
            
            # Compute metrics every epoch
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
                
                if epoch % 10 == 0:  # Only print every 10 epochs
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
        num_epochs = 50   # Reduced from 500
        lr = 1e-4          # Back to original learning rate
        weight_decay = 0.001 # Reduced L2 regularization
        batch_size = 128
        patience = 500      # Increased patience
        
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
                
                # Compute metrics every epoch
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
                    
                # Save metrics every epoch
                self._save_metrics(f'MLP', depth, epoch, train_r2, val_r2, param_count)
                
                # Print progress every 10 epochs
                if epoch % 10 == 0:
                    print(f"[MLP-{depth}] Epoch {epoch}/{num_epochs}, Train R²={train_r2:.4f}, Val R²={val_r2:.4f}")
                
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
        if not os.path.exists(self.results_path):
            self.skipTest("No results_js file found. Run KAN and MLP tests first.")
        
        results = pd.read_csv(self.results_path)
        
        # Skip plotting if no data
        if len(results) == 0:
            self.skipTest("Results file is empty. Run KAN and MLP tests first.")
            
        # Plot R² vs Epochs
        plt.figure(figsize=(12, 6))
        
        # Plot validation R² only
        # KAN
        kan_results = results[results['model_type'] == 'KAN']
        if len(kan_results) > 0:  # Only plot if we have KAN results
            plt.plot(kan_results['epoch'], kan_results['val_r2'], 
                    label=f'KAN [{kan_results.iloc[0]["param_count"]} params]', 
                    color='blue', linewidth=2)
        
        # MLPs
        colors = ['orange', 'green', 'red']
        for depth, color in zip([2, 3, 4], colors):
            mlp_d = results[(results['model_type'] == 'MLP') & (results['depth'] == depth)]
            if len(mlp_d) > 0:  # Only plot if we have results for this depth
                param_count = mlp_d.iloc[0]['param_count']
                plt.plot(mlp_d['epoch'], mlp_d['val_r2'], 
                        label=f'MLP-{depth} [{param_count} params]', 
                        color=color, linewidth=2)
        
        plt.title("KAN vs MLP Depths: Validation R² vs Epoch\nJane Street Market Prediction")
        plt.xlabel("Epoch")
        plt.xlim(0, 50)  # Limit x-axis to 50 epochs
        plt.ylabel("Weighted R²")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'./results_js/kan_vs_mlp_comparison_{datetime.now()}.png', bbox_inches='tight')
        print("Comparison plot saved to: ./results_js/kan_vs_mlp_comparison.png")

    def test_4_alternative_optimization(self):
        """Compare different optimization methods for KAN."""
        print("\n==== Testing Alternative Optimization Methods ====")
        
        # Base KAN config - same as in test_1_kan_training
        base_config = FixedKANConfig(
            network_shape=[self.input_dim, 20, 1],
            max_degree=5,
            complexity_weight=0.0,
            trainable_coefficients=True,
            skip_qubo_for_hidden=False,
            default_hidden_degree=5
        )
        
        # Common training parameters
        num_epochs = 50
        lr = 1e-4
        
        # Define optimization methods to test
        optimization_methods = [
            ('QUBO', lambda kan, x, y: kan.optimize(x, y)),
            ('IntegerProgramming', lambda kan, x, y: kan.optimize_integer_programming(x, y)),
            ('Evolutionary', lambda kan, x, y: kan.optimize_evolutionary(x, y)),
            ('GreedyHeuristic', lambda kan, x, y: kan.optimize_greedy_heuristic(x, y))
        ]
        
        # Measure optimization time and performance for each method
        import time
        import signal
        
        # Define a timeout handler
        class TimeoutException(Exception):
            pass
        
        def timeout_handler(signum, frame):
            raise TimeoutException("Optimization timed out")
        
        # Store performance metrics
        opt_results = []
        
        for method_name, optimize_fn in optimization_methods:
            print(f"\n--- Testing {method_name} Optimization ---")
            
            # Create a new KAN model
            qkan = FixedKAN(base_config)
            param_count = count_parameters(qkan)
            print(f"KAN parameter count: {param_count}")
            
            # Set a timeout (10 minutes)
            optimization_timeout = 600  # seconds
            
            # Measure optimization time
            start_time = time.time()
            
            # Initialize optimization success flag
            optimization_successful = False
            opt_time = 0
            
            try:
                # Set the timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(optimization_timeout)
                
                # Run the optimization
                optimize_fn(qkan, self.x_train, self.y_train.unsqueeze(-1))
                
                # Optimization completed successfully
                opt_time = time.time() - start_time
                optimization_successful = True
                
                # Clear the alarm
                signal.alarm(0)
                
                print(f"{method_name} optimization completed in {opt_time:.2f} seconds")
            except TimeoutException:
                print(f"{method_name} optimization timed out after {optimization_timeout} seconds!")
                # Use default optimization to recover
                if method_name != "QUBO":
                    print("Falling back to QUBO optimization to continue the test...")
                    qkan.optimize(self.x_train, self.y_train.unsqueeze(-1))
                opt_time = optimization_timeout
            except Exception as e:
                print(f"{method_name} optimization failed with error: {str(e)}")
                # Use default optimization to recover
                if method_name != "QUBO":
                    print("Falling back to QUBO optimization to continue the test...")
                    qkan.optimize(self.x_train, self.y_train.unsqueeze(-1))
                opt_time = time.time() - start_time
            
            # Only continue with training if optimization was successful or we recovered
            if optimization_successful or method_name != "QUBO":
                # Training loop
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
                    
                    # Compute metrics every epoch
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
                        
                        if epoch % 10 == 0:  # Only print every 10 epochs
                            print(f"[KAN-{method_name}] Epoch {epoch}/{num_epochs}, Train R²={train_r2:.4f}, Val R²={val_r2:.4f}")
                        self._save_metrics(f'KAN-{method_name}', 0, epoch, train_r2, val_r2, param_count)
                
                # Save final model
                save_path = f"./models_janestreet/kan_{method_name.lower()}_valr2_{val_r2:.4f}.pth"
                qkan.save_model(save_path)
                print(f"KAN model with {method_name} optimization saved to: {save_path}")
                
                # Record final metrics
                opt_results.append({
                    'method': method_name,
                    'opt_time': opt_time,
                    'final_train_r2': train_r2,
                    'final_val_r2': val_r2,
                    'param_count': param_count,
                    'optimization_successful': optimization_successful
                })
            else:
                print(f"Skipping training for {method_name} due to optimization failure")
                # Record failure in metrics
                opt_results.append({
                    'method': method_name,
                    'opt_time': opt_time,
                    'final_train_r2': float('nan'),
                    'final_val_r2': float('nan'),
                    'param_count': param_count,
                    'optimization_successful': False
                })
            
            # Cleanup
            del qkan
            torch.cuda.empty_cache()
            gc.collect()
        
        # Save optimization comparison results
        opt_df = pd.DataFrame(opt_results)
        opt_df.to_csv('./results_js/kan_optimization_comparison.csv', index=False)
        print("Optimization comparison saved to: ./results_js/kan_optimization_comparison.csv")
        
        # Load the results for plotting
        if 'results' not in locals() or 'results' not in globals():
            # If results variable is not defined, load from file
            if os.path.exists(self.results_path):
                results = pd.read_csv(self.results_path)
            else:
                # Just use the current session's data
                results = self.results_df
        
        # Create comparison plot for the optimization methods
        plt.figure(figsize=(12, 6))
        
        # Plot validation R² for each optimization method
        colors = ['blue', 'orange', 'green', 'red']
        for i, method in enumerate(['QUBO', 'IntegerProgramming', 'Evolutionary', 'GreedyHeuristic']):
            method_results = results[results['model_type'] == f'KAN-{method}']
            if len(method_results) > 0:
                plt.plot(method_results['epoch'], method_results['val_r2'], 
                        label=f'KAN-{method}', 
                        color=colors[i], linewidth=2)
        
        plt.title("KAN Optimization Methods: Validation R² vs Epoch\nJane Street Market Prediction")
        plt.xlabel("Epoch")
        plt.xlim(0, 50)
        plt.ylabel("Weighted R²")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'./results_js/kan_optimization_comparison_{datetime.now()}.png', bbox_inches='tight')
        print("Optimization comparison plot saved to: ./results_js/kan_optimization_comparison.png")
        
        # Create a bar chart to compare optimization times
        plt.figure(figsize=(10, 6))
        methods = [result['method'] for result in opt_results]
        times = [result['opt_time'] for result in opt_results]
        
        # Bar colors to match the line plot
        bar_colors = ['blue', 'orange', 'green', 'red'][:len(methods)]
        
        bars = plt.bar(methods, times, color=bar_colors)
        
        # Add time values on top of each bar
        for bar, time_val in zip(bars, times):
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
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'./results_js/kan_optimization_times_{datetime.now()}.png', bbox_inches='tight')
        print("Optimization time comparison saved to: ./results_js/kan_optimization_times.png")
        
        # Create a bar chart to compare final validation R²
        plt.figure(figsize=(10, 6))
        val_r2s = [result['final_val_r2'] for result in opt_results]
        
        bars = plt.bar(methods, val_r2s, color=bar_colors)
        
        # Add R² values on top of each bar
        for bar, r2_val in zip(bars, val_r2s):
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
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'./results_js/kan_validation_r2_comparison_{datetime.now()}.png', bbox_inches='tight')
        print("Validation R² comparison saved to: ./results_js/kan_validation_r2_comparison.png")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
