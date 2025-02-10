import math
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
from data_pipeline_js_config import DataConfig
from data_pipeline import DataPipeline

# 2) KAN code
from CP_KAN import FixedKANConfig, FixedKAN

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

def count_parameters(module: nn.Module) -> int:
    """
    Counts total number of trainable parameters in a PyTorch module.
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def build_mlp_for_depth(input_dim: int, output_dim: int, depth: int, target_params: int) -> nn.Module:
    """
    Build an MLP of a given depth (i.e. number of hidden layers),
    trying to keep total param count close to `target_params`.

    We'll do a naive approach: assume all hidden layers have the same width `H`.
    Then solve for `H` so that total param_count ~ target_params.

    Depth = number of hidden layers, not counting the output layer.
    E.g. depth=2 => MLP with 2 hidden layers + 1 final linear layer.

    Returns nn.Sequential with [Linear -> ReLU -> ... -> Linear -> ReLU -> Linear].
    """
    # We'll define a function that, given a hidden size h, calculates total param count for this depth.
    def calc_param_count_for_h(h):
        # Build a temporary MLP
        layers = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        # final
        layers.append(nn.Linear(in_dim, output_dim))

        temp_mlp = nn.Sequential(*layers)
        return count_parameters(temp_mlp)

    # We do a simple search to find the hidden width that gets us near `target_params`.
    best_h = 4
    best_diff = float('inf')
    for h in range(4, 1024, 4):  # step by 4 or whatever step you like
        pc = calc_param_count_for_h(h)
        diff = abs(pc - target_params)
        if diff < best_diff:
            best_diff = diff
            best_h = h

    # Now build the final MLP with the best hidden size
    mlp_layers = []
    in_dim = input_dim
    for _ in range(depth):
        mlp_layers.append(nn.Linear(in_dim, best_h))
        mlp_layers.append(nn.ReLU())
        in_dim = best_h
    mlp_layers.append(nn.Linear(in_dim, output_dim))

    return nn.Sequential(*mlp_layers)




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
            max_degree=5,
            complexity_weight=0.0,
            trainable_coefficients=False,
            skip_qubo_for_hidden=False,
            default_hidden_degree=3
        )

        # Tiny MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Ensure directory for saving
        os.makedirs("../models_janestreet", exist_ok=True)

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
        losses_array = np.array(train_losses, dtype=np.float32)
        np.save(f"./models_janestreet/qkan_train_losses_{r2_val:.4f}.npy", losses_array)
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
        # MLP training loop is done
        # ...
        losses_array = np.array(train_losses, dtype=np.float32)
        np.save(f"./models_janestreet/mlp_train_losses_{r2_val:.4f}.npy", losses_array)
    def test_3_lightgbm_regression(self):
        """
        LightGBM => Weighted MSE => measure normal MSE, Weighted MSE, Weighted R^2 => save model & partial training curve
        """
        print("\n==== [LightGBM] Weighted MSE Regression ====")
        # Convert to lgb.Dataset => pass sample_weight for training
        dtrain = lgb.Dataset(self.x_train_np, label=self.y_train_np.squeeze(-1), weight=self.w_train_np.squeeze(-1))
        # We'll do a small valid set => you can pass or just do your own eval
        dval = lgb.Dataset(self.x_val_np, label=self.y_val_np.squeeze(-1), weight=self.w_val_np.squeeze(-1))

        # We'll store results_js in evals_result for plotting
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
            plt.show()
    def test_4_compare_kan_mlp_depths(self):
        """
        Compare CP-KAN vs MLPs of depths 2,3,4,5 on Weighted MSE + Weighted R².
        We'll produce a single figure with R² vs. epochs (both train and val).
        """

        # ------------------------
        # 1) Prepare CP-KAN
        # ------------------------
        print("\n==== [CP-KAN] Weighted MSE Regression ====")
        qkan = FixedKAN(self.qkan_config)

        # Do QUBO-based degree selection (unweighted least squares)
        qkan.optimize(self.x_train_torch, self.y_train_torch.unsqueeze(-1))

        # We'll train for 500 epochs, same as your existing code
        num_epochs = 500
        lr = 1e-3

        # Gather all trainable params
        params_to_train = []
        for layer in qkan.layers:
            params_to_train.extend([layer.combine_W, layer.combine_b])
            for neuron in layer.neurons:
                params_to_train.extend([neuron.w, neuron.b])
                if (self.qkan_config.trainable_coefficients and
                        neuron.coefficients is not None):
                    params_to_train.extend(list(neuron.coefficients))
        cpk_param_count = sum(p.numel() for p in params_to_train)
        print(f"[CP-KAN] param_count={cpk_param_count}")

        optimizer = torch.optim.Adam(params_to_train, lr=lr)

        # We'll track Weighted R² each epoch for train & val
        qkan_train_r2 = []
        qkan_val_r2   = []

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            y_pred = qkan(self.x_train_torch).squeeze(-1)
            numerator = torch.sum(self.w_train_torch * (self.y_train_torch - y_pred)**2)
            denominator = torch.sum(self.w_train_torch)
            loss = numerator / (denominator + 1e-12)
            loss.backward()
            optimizer.step()

            # Compute Weighted R² on train
            with torch.no_grad():
                y_pred_train_np = y_pred.detach().cpu().numpy()
                y_true_train_np = self.y_train_torch.cpu().numpy()
                w_train_np      = self.w_train_torch.cpu().numpy()
                r2_train = weighted_r2(y_true_train_np, y_pred_train_np, w_train_np)
                qkan_train_r2.append(r2_train)

                # Weighted R² on validation
                y_pred_val = qkan(self.x_val_torch).squeeze(-1).cpu().numpy()
                y_true_val = self.y_val_torch.cpu().numpy()
                w_val      = self.w_val_torch.cpu().numpy()
                r2_val     = weighted_r2(y_true_val, y_pred_val, w_val)
                qkan_val_r2.append(r2_val)

            if (epoch+1) % 50 == 0:
                print(f"[CP-KAN] Epoch {epoch+1}/{num_epochs}, Weighted MSE={loss.item():.6f}, Train R²={r2_train:.4f}, Val R²={r2_val:.4f}")

        # Final CP-KAN metrics on val
        final_r2_val = qkan_val_r2[-1]
        print(f"[CP-KAN] Final Val Weighted R²={final_r2_val:.4f}")

        # ========== 2) MLPs of Depth=2..5 (Fixed hidden_size=64) ==========

        def build_simple_mlp(in_dim, out_dim, depth=2, hidden_size=64):
            """
            Build a straightforward MLP with `depth` hidden layers, each of width `hidden_size`.
            No param search.
            """
            layers = []
            curr_dim = in_dim
            for _ in range(depth):
                layers.append(nn.Linear(curr_dim, hidden_size))
                layers.append(nn.ReLU())
                curr_dim = hidden_size
            layers.append(nn.Linear(curr_dim, out_dim))
            return nn.Sequential(*layers)

        mlp_depths = [2, 3, 4, 5]
        mlp_models = {}
        mlp_param_counts = {}
        batch_size = 16
        # Build each MLP
        for d in mlp_depths:
            mlp_d = build_simple_mlp(self.input_dim, 1, depth=d, hidden_size=64)
            pc = count_parameters(mlp_d)
            mlp_models[d] = mlp_d
            mlp_param_counts[d] = pc
            print(f"[MLP-depth={d}] param_count={pc}")

        # We'll store Weighted R² histories
        mlp_r2_histories = {d: {"train": [], "val": []} for d in mlp_depths}

        # ========== 3) Train each MLP with mini-batches ==========
        for d in mlp_depths:
            print(f"\n==== [MLP-depth={d}] Weighted MSE Regression ====")
            mlp = mlp_models[d]
            optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=lr)

            train_r2_list = []
            val_r2_list   = []

            for epoch in range(num_epochs):
                # Mini-batch Weighted MSE
                mlp.train()
                n_train = len(self.x_train_torch)
                n_batches = math.ceil(n_train / batch_size)

                for i in range(n_batches):
                    start = i*batch_size
                    end   = (i+1)*batch_size
                    x_batch = self.x_train_torch[start:end]
                    y_batch = self.y_train_torch[start:end]
                    w_batch = self.w_train_torch[start:end]

                    optimizer_mlp.zero_grad()
                    pred_batch = mlp(x_batch).squeeze(-1)
                    numerator = torch.sum(w_batch*(y_batch - pred_batch)**2)
                    denominator = torch.sum(w_batch)
                    loss = numerator/(denominator+1e-12)
                    loss.backward()
                    optimizer_mlp.step()

                # Evaluate Weighted R² once per epoch
                mlp.eval()
                with torch.no_grad():
                    y_pred_train = mlp(self.x_train_torch).squeeze(-1).cpu().numpy()
                    r2_train = weighted_r2(self.y_train_np, y_pred_train, self.w_train_np)
                    train_r2_list.append(r2_train)

                    y_pred_val = mlp(self.x_val_torch).squeeze(-1).cpu().numpy()
                    r2_val = weighted_r2(self.y_val_np, y_pred_val, self.w_val_np)
                    val_r2_list.append(r2_val)

                if (epoch+1) % 50 == 0:
                    print(f"[MLP-depth={d}] epoch {epoch+1}/{num_epochs}, Train R²={r2_train:.4f}, Val R²={r2_val:.4f}")

            mlp_r2_histories[d]["train"] = train_r2_list
            mlp_r2_histories[d]["val"]   = val_r2_list

        # ========== 4) Plot all on one figure ==========
        epochs_range = range(1, num_epochs+1)
        plt.figure(figsize=(8,6))

        # CP-KAN lines
        plt.plot(epochs_range, qkan_train_r2,
                 label=f"CP-KAN (train) [{cpk_param_count} params]", color='blue')
        plt.plot(epochs_range, qkan_val_r2,
                 label="CP-KAN (val)", color='blue', linestyle='--')

        color_map = {2:'orange', 3:'green', 4:'red', 5:'purple'}

        # MLP lines
        for d in mlp_depths:
            train_r2 = mlp_r2_histories[d]["train"]
            val_r2   = mlp_r2_histories[d]["val"]
            param_str = f"[{mlp_param_counts[d]} params]"
            plt.plot(epochs_range, train_r2,
                     label=f"MLP(d={d}) train {param_str}",
                     color=color_map[d])
            plt.plot(epochs_range, val_r2,
                     label=f"MLP(d={d}) val",
                     color=color_map[d], linestyle='--')

        plt.title("CP-KAN vs. MLP Depths (Weighted R² vs. Epoch)\nJane Street Example (No Param Search)")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted R²")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_path = f"./models_janestreet/comparison_cpk_mlp_depths_simple.png"
        plt.savefig(plot_path)
        plt.show()
        print(f"[Done] Combined plot saved => {plot_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main(argv=["first-arg-is-ignored"], exit=False)