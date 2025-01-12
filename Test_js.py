import unittest
import logging
import os
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

import polars as pl
from typing import Tuple

# Your config & pipeline classes
from config import DataConfig
from data_pipeline import DataPipeline

# The QKAN code
from KAN_w_cumulative_polynomials import FixedKANConfig, FixedKAN

# Just in case you want a small MLP baseline
import torch.nn as nn


def weighted_r2(y_true: torch.Tensor, y_pred: torch.Tensor, w: torch.Tensor) -> float:
    """
    Weighted R^2 = 1 - ( sum(w_i*(y_i - y_pred_i)^2 ) / sum(w_i*y_i^2 ) )
    """
    numerator = torch.sum(w * (y_true - y_pred)**2)
    denominator = torch.sum(w * (y_true**2))
    if denominator.item() < 1e-12:
        return 0.0
    return 1.0 - (numerator / denominator).item()

def weighted_mse(y_true: torch.Tensor, y_pred: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Weighted MSE = sum(w_i*(y_i - y_pred_i)^2) / sum(w_i)
    """
    numerator = torch.sum(w * (y_true - y_pred)**2)
    denominator = torch.sum(w)
    return numerator / (denominator + 1e-12)


class TestJaneStreetQKANRegression(unittest.TestCase):
    def setUp(self):
        """
        1) Reads your DataConfig (pointing to the Parquet file, feature cols, etc.).
        2) Loads & preprocesses data via DataPipeline => train/val splits
        3) Converts polars => torch
        4) Preps QKAN config & a tiny MLP
        """
        # -------------------------
        # 1) DataConfig & Pipeline
        # -------------------------
        self.logger = logging.getLogger("TestJaneStreetQKANRegression")
        self.logger.setLevel(logging.INFO)

        # A sample DataConfig. Adjust the fields as in your YAML or pass them in some way:
        self.data_cfg = DataConfig(
            data_path="~/Interning/Kaggle/jane_street_kaggle/jane-street-real-time-market-data-forecasting/train.parquet/",
            n_rows=5000000,
            train_ratio=0.7,
            feature_cols=[f'feature_{i:02d}' for i in range(79)],  # example
            target_col="responder_6",
            weight_col="weight",
            date_col="date_id"
        )

        pipeline = DataPipeline(self.data_cfg, self.logger)

        # load_and_preprocess_data => returns polars DataFrame
        train_df, train_target, train_weight, val_df, val_target, val_weight = pipeline.load_and_preprocess_data()
        # train_df, train_target, train_weight, val_df, val_target, val_weight are polars DataFrames

        # Convert polars => numpy => torch
        self.x_train = torch.tensor(train_df.to_numpy(), dtype=torch.float32)
        self.y_train = torch.tensor(train_target.to_numpy(), dtype=torch.float32).squeeze(-1)
        self.w_train = torch.tensor(train_weight.to_numpy(), dtype=torch.float32).squeeze(-1)

        self.x_val = torch.tensor(val_df.to_numpy(), dtype=torch.float32)
        self.y_val = torch.tensor(val_target.to_numpy(), dtype=torch.float32).squeeze(-1)
        self.w_val = torch.tensor(val_weight.to_numpy(), dtype=torch.float32).squeeze(-1)

        # Decide dimension
        self.input_dim = self.x_train.shape[1]
        # For demonstration, the code has 5 features (like "feature_0..4") or 79 in your config.

        # -------------------------
        # 2) QKAN Config
        # -------------------------
        self.qkan_config = FixedKANConfig(
            network_shape=[self.input_dim, 1],
            max_degree=5,
            complexity_weight=0.0,
            trainable_coefficients=False,
            skip_qubo_for_hidden=False,  # single-layer => no hidden
            default_hidden_degree=4
        )

        # -------------------------
        # 3) Tiny MLP
        # -------------------------
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        # Create a folder to store models & plots
        os.makedirs("./models_janestreet", exist_ok=True)

    def test_1_qkan_regression(self):
        """
        QKAN => Weighted MSE => Weighted R^2 on val set. Save model & plot training curve.
        """
        print("\n==== [QKAN] Weighted MSE Regression on Jane Street Data ====")
        qkan = FixedKAN(self.qkan_config)

        # 1) QUBO => single-layer => dimension=1
        #   - We skip sample weighting in QUBO for simplicity (unweighted MSE).
        #   - So we pass (x_train, y_train)
        qkan.optimize(self.x_train, self.y_train.unsqueeze(-1))

        # 2) Weighted MSE training
        num_epochs = 100
        lr = 1e-3
        train_losses = []

        # Gather trainable params (like in your MSE test code)
        params_to_train = []
        for layer in qkan.layers:
            params_to_train.extend([layer.combine_W, layer.combine_b])
            for neuron in layer.neurons:
                params_to_train.extend([neuron.w, neuron.b])
                if self.qkan_config.trainable_coefficients and neuron.coefficients is not None:
                    params_to_train.extend(list(neuron.coefficients))

        optimizer = torch.optim.Adam(params_to_train, lr=lr)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            y_pred = qkan(self.x_train).squeeze(-1)   # shape [N]
            loss = weighted_mse(self.y_train, y_pred, self.w_train)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            if (epoch+1) % 5 == 0:
                print(f"[QKAN] Epoch {epoch+1}/{num_epochs}, Weighted MSE={loss.item():.6f}")

        # 3) Evaluate Weighted R^2 on val
        with torch.no_grad():
            y_pred_val = qkan(self.x_val).squeeze(-1)
        r2_val = weighted_r2(self.y_val, y_pred_val, self.w_val)
        print(f"[QKAN Weighted MSE] Val Weighted R^2 = {r2_val:.4f}")

        # 4) Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"./models_janestreet/qkan_reg_{r2_val:.4f}_{timestamp}.pth"
        qkan.save_model(save_path)
        print(f"QKAN regression model saved to: {save_path}")

        # 5) Plot training curve
        plt.figure()
        plt.plot(train_losses, label="Train Weighted MSE")
        plt.title(f"QKAN Weighted MSE (Val R^2={r2_val:.4f})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plot_path = f"./models_janestreet/qkan_reg_loss_{r2_val:.4f}_{timestamp}.png"
        plt.savefig(plot_path)
        print(f"QKAN training-loss plot saved to: {plot_path}")
        plt.show() # optional

    def test_2_tiny_mlp_regression(self):
        """
        Tiny MLP => Weighted MSE => Weighted R^2 on val set. Save model & plot training curve.
        """
        print("\n==== [Tiny MLP] Weighted MSE Regression on Jane Street Data ====")
        num_epochs = 50
        lr = 1e-3
        train_losses = []

        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            y_pred = self.mlp(self.x_train).squeeze(-1)   # shape [N]
            loss = weighted_mse(self.y_train, y_pred, self.w_train)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            if (epoch+1) % 5 == 0:
                print(f"[Tiny MLP] Epoch {epoch+1}/{num_epochs}, Weighted MSE={loss.item():.6f}")

        # Evaluate Weighted R^2
        with torch.no_grad():
            y_pred_val = self.mlp(self.x_val).squeeze(-1)
        r2_val = weighted_r2(self.y_val, y_pred_val, self.w_val)
        print(f"[Tiny MLP Weighted MSE] Val Weighted R^2 = {r2_val:.4f}")

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"./models_janestreet/tinymlp_reg_{r2_val:.4f}_{timestamp}.pt"
        torch.save(self.mlp.state_dict(), save_path)
        print(f"Tiny MLP regression model saved to: {save_path}")

        # Plot training curve
        plt.figure()
        plt.plot(train_losses, label="Train Weighted MSE")
        plt.title(f"Tiny MLP Weighted MSE (Val R^2={r2_val:.4f})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plot_path = f"./models_janestreet/tinymlp_reg_loss_{r2_val:.4f}_{timestamp}.png"
        plt.savefig(plot_path)
        print(f"Tiny MLP training-loss plot saved to: {plot_path}")
        plt.show()


if __name__ == "__main__":
    # Optionally configure logging:
    logging.basicConfig(level=logging.INFO)

    unittest.main(argv=["first-arg-is-ignored"], exit=False)