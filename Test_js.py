import unittest
import logging
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import lightgbm as lgb

# 1) Data pipeline & config classes
from config import DataConfig
from data_pipeline import DataPipeline

# 2) KAN code
from KAN_w_cumulative_polynomials import FixedKANConfig, FixedKAN

# --------------------------
# Utility Metrics
# --------------------------
def normal_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Unweighted MSE = average of (y_true - y_pred)^2
    """
    return float(np.mean((y_true - y_pred)**2))

def weighted_mse(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    """
    Weighted MSE = sum(w_i*(y_i - y_pred_i)^2) / sum(w_i)
    """
    numerator = np.sum(w * (y_true - y_pred)**2)
    denominator = np.sum(w)
    return float(numerator / (denominator + 1e-12))

def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    """
    Weighted R^2 = 1 - sum(w_i*(y_i - y_pred_i)^2)/sum(w_i*(y_i^2))
    """
    numerator = np.sum(w * (y_true - y_pred)**2)
    denominator = np.sum(w * (y_true**2))
    if denominator < 1e-12:
        return 0.0
    return float(1.0 - numerator/denominator)

class TestJaneStreetModels(unittest.TestCase):
    def setUp(self):
        """
        1) Create a DataConfig => load data with DataPipeline => polars DataFrames
        2) Convert polars => numpy => torch
        3) Build QKAN config, MLP, etc.
        """
        self.logger = logging.getLogger("TestJaneStreetModels")
        self.logger.setLevel(logging.INFO)

        # Example config; adjust as needed
        self.data_cfg = DataConfig(
            data_path="~/Interning/Kaggle/jane_street_kaggle/jane-street-real-time-market-data-forecasting/train.parquet/",
            n_rows=200000,        # or 1,000,000 as in your YAML
            train_ratio=0.7,
            feature_cols=[f'feature_{i:02d}' for i in range(79)],
            target_col="responder_6",
            weight_col="weight",
            date_col="date_id"
        )

        pipeline = DataPipeline(self.data_cfg, self.logger)
        train_df, train_target, train_weight, val_df, val_target, val_weight = pipeline.load_and_preprocess_data()

        # Convert polars => numpy => torch (for QKAN/MLP)
        self.x_train_np = train_df.to_numpy()          # shape [N_train, num_features]
        self.y_train_np = train_target.to_numpy()      # shape [N_train,1]
        self.w_train_np = train_weight.to_numpy()      # shape [N_train,1]

        self.x_val_np   = val_df.to_numpy()            # shape [N_val, num_features]
        self.y_val_np   = val_target.to_numpy()        # shape [N_val,1]
        self.w_val_np   = val_weight.to_numpy()        # shape [N_val,1]

        self.x_train_torch = torch.tensor(self.x_train_np, dtype=torch.float32)
        self.y_train_torch = torch.tensor(self.y_train_np, dtype=torch.float32).squeeze(-1)
        self.w_train_torch = torch.tensor(self.w_train_np, dtype=torch.float32).squeeze(-1)

        self.x_val_torch   = torch.tensor(self.x_val_np,   dtype=torch.float32)
        self.y_val_torch   = torch.tensor(self.y_val_np,   dtype=torch.float32).squeeze(-1)
        self.w_val_torch   = torch.tensor(self.w_val_np,   dtype=torch.float32).squeeze(-1)

        self.input_dim = self.x_train_np.shape[1]

        # QKAN config
        self.qkan_config = FixedKANConfig(
            network_shape=[self.input_dim,10, 1],
            max_degree=3,
            complexity_weight=0.0,
            trainable_coefficients=False,
            skip_qubo_for_hidden=False,
            default_hidden_degree=3
        )

        # Tiny MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Ensure directory for saving
        os.makedirs("./models_janestreet", exist_ok=True)

    def test_1_qkan_regression(self):
        """
        QKAN => Weighted MSE => measure normal MSE, Weighted MSE, Weighted R^2 on val => save & plot
        """
        print("\n==== [QKAN] Weighted MSE Regression ====")
        qkan = FixedKAN(self.qkan_config)

        # 1) QUBO => ignoring sample weights => unweighted least squares on (x,y)
        qkan.optimize(self.x_train_torch, self.y_train_torch.unsqueeze(-1))

        # 2) Weighted MSE train
        num_epochs = 500
        lr = 1e-3
        train_losses = []

        # gather QKAN params
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
            y_pred = qkan(self.x_train_torch).squeeze(-1)  # shape [N]
            # Weighted MSE
            numerator = torch.sum(self.w_train_torch * (self.y_train_torch - y_pred)**2)
            denominator = torch.sum(self.w_train_torch)
            loss = numerator / (denominator + 1e-12)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            if (epoch+1) % 5 == 0:
                print(f"[QKAN] Epoch {epoch+1}/{num_epochs}, Weighted MSE={loss.item():.6f}")

        # Evaluate on val set
        with torch.no_grad():
            y_pred_val = qkan(self.x_val_torch).squeeze(-1).cpu().numpy()

        # Compute metrics
        y_true_val = self.y_val_torch.cpu().numpy()
        w_val      = self.w_val_torch.cpu().numpy()

        nmse_val   = normal_mse(y_true_val, y_pred_val)
        wmse_val   = weighted_mse(y_true_val, y_pred_val, w_val)
        r2_val     = weighted_r2(y_true_val, y_pred_val, w_val)
        print(f"QKAN => Normal MSE={nmse_val:.6f}, Weighted MSE={wmse_val:.6f}, Weighted R^2={r2_val:.4f}")

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"./models_janestreet/qkan_{r2_val:.4f}_{timestamp}.pth"
        qkan.save_model(save_path)
        print(f"QKAN model saved to: {save_path}")

        # Plot training curve
        plt.figure()
        plt.plot(train_losses, label="Train Weighted MSE")
        plt.title(f"QKAN Weighted MSE (Val R^2={r2_val:.4f})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plot_path = f"./models_janestreet/qkan_loss_{r2_val:.4f}_{timestamp}.png"
        plt.savefig(plot_path)
        print(f"Saved QKAN training-loss plot: {plot_path}")

    def test_2_mlp_regression(self):
        """
        Tiny MLP => Weighted MSE => measure normal MSE, Weighted MSE, Weighted R^2 on val => save & plot
        """
        print("\n==== [Tiny MLP] Weighted MSE Regression ====")
        num_epochs = 500
        lr = 1e-3
        train_losses = []

        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            y_pred = self.mlp(self.x_train_torch).squeeze(-1)
            numerator = torch.sum(self.w_train_torch * (self.y_train_torch - y_pred)**2)
            denominator = torch.sum(self.w_train_torch)
            loss = numerator / (denominator + 1e-12)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            if (epoch+1) % 5 == 0:
                print(f"[MLP] Epoch {epoch+1}/{num_epochs}, Weighted MSE={loss.item():.6f}")

        # Evaluate
        self.mlp.eval()
        with torch.no_grad():
            y_pred_val = self.mlp(self.x_val_torch).squeeze(-1).cpu().numpy()

        y_true_val = self.y_val_torch.cpu().numpy()
        w_val      = self.w_val_torch.cpu().numpy()

        nmse_val = normal_mse(y_true_val, y_pred_val)
        wmse_val = weighted_mse(y_true_val, y_pred_val, w_val)
        r2_val   = weighted_r2(y_true_val, y_pred_val, w_val)
        print(f"MLP => Normal MSE={nmse_val:.6f}, Weighted MSE={wmse_val:.6f}, Weighted R^2={r2_val:.4f}")

        # Save MLP model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"./models_janestreet/tiny_mlp_{r2_val:.4f}_{timestamp}.pt"
        torch.save(self.mlp.state_dict(), save_path)
        print(f"Tiny MLP model saved to: {save_path}")

        # Plot training curve
        plt.figure()
        plt.plot(train_losses, label="Train Weighted MSE")
        plt.title(f"Tiny MLP Weighted MSE (Val R^2={r2_val:.4f})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plot_path = f"./models_janestreet/tiny_mlp_loss_{r2_val:.4f}_{timestamp}.png"
        plt.savefig(plot_path)
        print(f"Saved MLP training-loss plot: {plot_path}")

    def test_3_lightgbm_regression(self):
        """
        LightGBM => Weighted MSE => measure normal MSE, Weighted MSE, Weighted R^2 => save model & partial training curve
        """
        print("\n==== [LightGBM] Weighted MSE Regression ====")
        # Convert to lgb.Dataset => pass sample_weight for training
        dtrain = lgb.Dataset(self.x_train_np, label=self.y_train_np.squeeze(-1), weight=self.w_train_np.squeeze(-1))
        # We'll do a small valid set => you can pass or just do your own eval
        dval = lgb.Dataset(self.x_val_np, label=self.y_val_np.squeeze(-1), weight=self.w_val_np.squeeze(-1))

        # We'll store results in evals_result for plotting
        evals_result = {}
        params = {
            'objective': 'regression_l2',
            'learning_rate': 0.01,
            'metric': 'rmse',   # or 'rmse'
            'verbosity': -1
        }
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dtrain, dval],
            valid_names=['train','val'],
            #evals_result=evals_result,
            #early_stopping_rounds=30
        )

        # Evaluate
        y_pred_val = model.predict(self.x_val_np)
        y_true_val = self.y_val_np.squeeze(-1)
        w_val      = self.w_val_np.squeeze(-1)

        nmse_val = normal_mse(y_true_val, y_pred_val)
        wmse_val = weighted_mse(y_true_val, y_pred_val, w_val)
        r2_val   = weighted_r2(y_true_val, y_pred_val, w_val)
        print(f"LightGBM => Normal MSE={nmse_val:.6f}, Weighted MSE={wmse_val:.6f}, Weighted R^2={r2_val:.4f}")

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"./models_janestreet/lightgbm_{r2_val:.4f}_{timestamp}.txt"
        model.save_model(model_path)
        print(f"LightGBM model saved to: {model_path}")

        # Plot partial training curve
        # LightGBM stores 'l2' or 'rmse' in evals_result, let's see what we have
        train_metric = evals_result['train']['l2'] if 'l2' in evals_result['train'] else None
        val_metric   = evals_result['val']['l2']   if 'l2' in evals_result['val']   else None
        if train_metric is not None:
            plt.figure()
            plt.plot(train_metric, label='train l2')
            plt.plot(val_metric, label='val l2')
            plt.title(f"LightGBM (Val R^2={r2_val:.4f})")
            plt.xlabel("Iteration")
            plt.ylabel("L2 Loss")
            plt.legend()
            plot_path = f"./models_janestreet/lightgbm_loss_{r2_val:.4f}_{timestamp}.png"
            plt.savefig(plot_path)
            print(f"Saved LightGBM training curve: {plot_path}")
            # plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main(argv=["first-arg-is-ignored"], exit=False)