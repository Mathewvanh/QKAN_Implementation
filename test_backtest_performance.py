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

# Local imports
from data_pipeline_backtest import BacktestConfig, DataPipelineBacktest
from CP_KAN import FixedKANConfig, FixedKAN

def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    """Compute weighted R² score."""
    numerator = np.sum(w * (y_true - y_pred)**2)
    denominator = np.sum(w * (y_true**2))
    if denominator < 1e-12:
        return 0.0
    return float(1.0 - numerator/denominator)

class TestBacktestPerformance(unittest.TestCase):
    def setUp(self):
        """Initialize data and configurations."""
        self.logger = logging.getLogger("TestBacktestPerformance")
        self.logger.setLevel(logging.INFO)

        # Ensure directories exist
        os.makedirs("./models_backtest", exist_ok=True)
        os.makedirs("results_backtest", exist_ok=True)

        # Load data from earlier period
        self.data_cfg = BacktestConfig(
            data_path="~/Interning/Kaggle/jane_street_kaggle/jane-street-real-time-market-data-forecasting/train.parquet/",
            start_row=13000000,  # Start well before our previous training data
            n_rows=200000,
            train_ratio=0.7,
            feature_cols=[f'feature_{i:02d}' for i in range(79)],
            target_col="responder_6",
            weight_col="weight",
            date_col="date_id"
        )

        # Load and preprocess data
        pipeline = DataPipelineBacktest(self.data_cfg, self.logger)
        train_df, train_target, train_weight, val_df, val_target, val_weight = pipeline.load_and_preprocess_data()
        train_dates, val_dates = pipeline.get_date_info()

        # Convert to numpy then torch
        self.x_train = torch.tensor(train_df.to_numpy(), dtype=torch.float32)
        self.y_train = torch.tensor(train_target.to_numpy(), dtype=torch.float32).squeeze(-1)
        self.w_train = torch.tensor(train_weight.to_numpy(), dtype=torch.float32).squeeze(-1)

        self.x_val = torch.tensor(val_df.to_numpy(), dtype=torch.float32)
        self.y_val = torch.tensor(val_target.to_numpy(), dtype=torch.float32).squeeze(-1)
        self.w_val = torch.tensor(val_weight.to_numpy(), dtype=torch.float32).squeeze(-1)

        self.train_dates = train_dates
        self.val_dates = val_dates
        self.input_dim = self.x_train.shape[1]

        # Results DataFrame
        self.results_df = pd.DataFrame(columns=[
            'model_type', 'date_id', 'r2_score', 'inference_time',
            'prediction_mean', 'prediction_std'
        ])

    def _load_best_models(self):
        """Load best models from degradation test."""
        # Load best KAN (lr=1e-4)
        kan_checkpoint = torch.load('./models_degradation/kan_lr1.0e-04_peak199.pth')
        kan_config = kan_checkpoint['config']
        
        kan = FixedKAN(kan_config)
        kan.load_state_dict(kan_checkpoint['model_state'])
        
        # Load MLP at peak and degraded states
        mlp_peak = torch.load('./models_degradation/mlp_lr1.0e-05_peak54.pt')
        mlp_degraded = torch.load('./models_degradation/mlp_lr1.0e-05_peak199.pt')
        
        return kan, mlp_peak, mlp_degraded

    def _evaluate_model(self, model, x_data: torch.Tensor, y_true: np.ndarray, 
                       weights: np.ndarray, batch_size: int = 128) -> dict:
        """Evaluate model performance with timing."""
        model.eval()
        predictions = []
        inference_times = []
        
        with torch.no_grad():
            n_batches = math.ceil(len(x_data) / batch_size)
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(x_data))
                batch = x_data[start_idx:end_idx]
                
                # Time inference
                start_time = time.time()
                pred = model(batch).squeeze(-1).cpu().numpy()
                inference_time = time.time() - start_time
                
                predictions.extend(pred)
                inference_times.append(inference_time)
        
        predictions = np.array(predictions)
        
        return {
            'predictions': predictions,
            'r2_score': weighted_r2(y_true, predictions, weights),
            'inference_time': np.mean(inference_times),
            'pred_mean': np.mean(predictions),
            'pred_std': np.std(predictions)
        }

    def test_backtest_performance(self):
        """Compare model performance on backtest data."""
        print("\n==== Loading Best Models ====")
        kan, mlp_peak, mlp_degraded = self._load_best_models()
        
        # Test batch sizes
        batch_sizes = [1, 32, 128, 256]
        
        for batch_size in batch_sizes:
            print(f"\n==== Testing Batch Size: {batch_size} ====")
            
            # Evaluate KAN
            print("\n--- Evaluating KAN ---")
            kan_metrics = self._evaluate_model(
                kan, self.x_val, self.y_val.numpy(), 
                self.w_val.numpy(), batch_size
            )
            
            # Add to results
            new_row = pd.DataFrame({
                'model_type': ['KAN'],
                'batch_size': [batch_size],
                'r2_score': [kan_metrics['r2_score']],
                'inference_time': [kan_metrics['inference_time']],
                'prediction_mean': [kan_metrics['pred_mean']],
                'prediction_std': [kan_metrics['pred_std']]
            })
            self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
            
            print(f"KAN - R²: {kan_metrics['r2_score']:.4f}, "
                  f"Inference Time: {kan_metrics['inference_time']*1000:.2f}ms")
            
            # Evaluate MLP (Peak)
            print("\n--- Evaluating MLP (Peak) ---")
            mlp_peak_metrics = self._evaluate_model(
                mlp_peak, self.x_val, self.y_val.numpy(),
                self.w_val.numpy(), batch_size
            )
            
            new_row = pd.DataFrame({
                'model_type': ['MLP_Peak'],
                'batch_size': [batch_size],
                'r2_score': [mlp_peak_metrics['r2_score']],
                'inference_time': [mlp_peak_metrics['inference_time']],
                'prediction_mean': [mlp_peak_metrics['pred_mean']],
                'prediction_std': [mlp_peak_metrics['pred_std']]
            })
            self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
            
            print(f"MLP (Peak) - R²: {mlp_peak_metrics['r2_score']:.4f}, "
                  f"Inference Time: {mlp_peak_metrics['inference_time']*1000:.2f}ms")
            
            # Evaluate MLP (Degraded)
            print("\n--- Evaluating MLP (Degraded) ---")
            mlp_degraded_metrics = self._evaluate_model(
                mlp_degraded, self.x_val, self.y_val.numpy(),
                self.w_val.numpy(), batch_size
            )
            
            new_row = pd.DataFrame({
                'model_type': ['MLP_Degraded'],
                'batch_size': [batch_size],
                'r2_score': [mlp_degraded_metrics['r2_score']],
                'inference_time': [mlp_degraded_metrics['inference_time']],
                'prediction_mean': [mlp_degraded_metrics['pred_mean']],
                'prediction_std': [mlp_degraded_metrics['pred_std']]
            })
            self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
            
            print(f"MLP (Degraded) - R²: {mlp_degraded_metrics['r2_score']:.4f}, "
                  f"Inference Time: {mlp_degraded_metrics['inference_time']*1000:.2f}ms")
        
        # Save results
        self.results_df.to_csv('./results_backtest/backtest_metrics.csv', index=False)

    def test_plot_backtest_results(self):
        """Create visualization of backtest results."""
        if not os.path.exists('./results_backtest/backtest_metrics.csv'):
            self.skipTest("No results file found. Run backtest first.")
        
        results = pd.read_csv('./results_backtest/backtest_metrics.csv')
        batch_sizes = sorted(results['batch_size'].unique())
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot R² scores
        for model in ['KAN', 'MLP_Peak', 'MLP_Degraded']:
            model_results = results[results['model_type'] == model]
            ax1.plot(model_results['batch_size'], model_results['r2_score'],
                    marker='o', label=model)
        
        ax1.set_title('R² Score vs Batch Size')
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('R² Score')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot inference times
        for model in ['KAN', 'MLP_Peak', 'MLP_Degraded']:
            model_results = results[results['model_type'] == model]
            ax2.plot(model_results['batch_size'], 
                    model_results['inference_time'] * 1000,  # Convert to ms
                    marker='o', label=model)
        
        ax2.set_title('Inference Time vs Batch Size')
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Inference Time (ms)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'./results_backtest/backtest_comparison_{datetime.now()}.png',
                   bbox_inches='tight')
        print("Backtest comparison plot saved.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
