#!/usr/bin/env python3
"""
Updated script with mini-batch training for MLP + QKAN
on classification (covertype) and regression (house_sales).
We also log each model's parameter count, and use log-target for house_sales.
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Tuple

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb

# KAN code
from CP_KAN import FixedKANConfig, FixedKAN

#############################################################################
# 1) Helper: parameter counts & shape approximations
#############################################################################
def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def approximate_kan_shape_for_param_count(
        input_dim: int,
        target_dim: int,
        desired_params: int,
        max_degree: int = 3
    ) -> list:
    """
    Guess a KAN architecture (single hidden layer) that yields param_count ~ desired_params.
    """
    def param_count_kan_shape(h: int) -> int:
        config = FixedKANConfig(
            network_shape=[input_dim, h, target_dim],
            max_degree=max_degree,
            complexity_weight=0.0
        )
        model = FixedKAN(config)
        return count_parameters(model)

    best_h = 1
    best_diff = float("inf")
    for hdim in range(1, 501, 5):
        pc = param_count_kan_shape(hdim)
        diff = abs(pc - desired_params)
        if diff < best_diff:
            best_diff = diff
            best_h = hdim
        # Stop if we are far beyond the target
        if pc > desired_params * 2:
            break
    return [input_dim, best_h, target_dim]

def approximate_mlp_for_param_count(
        input_dim: int,
        target_dim: int,
        desired_params: int,
        hidden_init: int = 64
    ) -> nn.Module:
    """
    Build a small MLP with ~ desired_params by adjusting a single hidden layer size.
    """
    class TinyMLP(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )
        def forward(self, x):
            return self.model(x)

    best_h = hidden_init
    best_diff = float("inf")
    best_model = None

    for hdim in range(1, 501, 5):
        mlp_temp = TinyMLP(input_dim, hdim, target_dim)
        pc = count_parameters(mlp_temp)
        diff = abs(pc - desired_params)
        if diff < best_diff:
            best_diff = diff
            best_model = mlp_temp
            best_h = hdim
        if pc > desired_params * 2:
            break
    return best_model

#############################################################################
# 2) Mini-Batch Training Helpers
#############################################################################
def train_mlp_classification_minibatch(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    loss_fn = nn.CrossEntropyLoss()
):
    """
    One epoch of mini-batch training for classification (Cross Entropy).
    """
    model.train()
    n_samples = len(X_train)
    n_batches = math.ceil(n_samples / batch_size)

    for i in range(n_batches):
        start = i*batch_size
        end   = (i+1)*batch_size
        xb = X_train[start:end]
        yb = y_train[start:end]

        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

def train_mlp_regression_minibatch(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    loss_fn = nn.MSELoss()
):
    """
    One epoch of mini-batch training for regression (MSE).
    """
    model.train()
    n_samples = len(X_train)
    n_batches = math.ceil(n_samples / batch_size)

    for i in range(n_batches):
        start = i*batch_size
        end   = (i+1)*batch_size
        xb = X_train[start:end]
        yb = y_train[start:end]

        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()

def train_kan_classification_minibatch(
    kan: FixedKAN,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    loss_fn = nn.CrossEntropyLoss()
):
    """
    One epoch of mini-batch training for KAN classification (Cross Entropy).
    """
    kan.train()
    n_samples = len(X_train)
    n_batches = math.ceil(n_samples / batch_size)

    for i in range(n_batches):
        start = i*batch_size
        end   = (i+1)*batch_size
        xb = X_train[start:end]
        yb = y_train[start:end]

        optimizer.zero_grad()
        logits = kan(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

def train_kan_regression_minibatch(
    kan: FixedKAN,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    loss_fn = nn.MSELoss()
):
    """
    One epoch of mini-batch training for KAN regression (MSE).
    """
    kan.train()
    n_samples = len(X_train)
    n_batches = math.ceil(n_samples / batch_size)

    for i in range(n_batches):
        start = i*batch_size
        end   = (i+1)*batch_size
        xb = X_train[start:end]
        yb = y_train[start:end]

        optimizer.zero_grad()
        preds = kan(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()

#############################################################################
# 3) Classification on "covertype"
#############################################################################
def load_covertype_data() -> Tuple[np.ndarray, np.ndarray]:
    dataset = load_dataset(
        "inria-soda/tabular-benchmark",
        data_files="clf_num/covertype.csv",
        split="train"
    )
    df = pd.DataFrame(dataset)
    label_col = "target" if "target" in df.columns else df.columns[-1]

    y = df[label_col].values
    X = df.drop(columns=[label_col]).values
    print(f"[covertype] shape={df.shape}, label_col='{label_col}'")
    return X, y

def experiment_covertype_classification(batch_size=4096, num_epochs=50):
    print("\n=== Classification on covertype (numeric) ===")
    X, y = load_covertype_data()

    # Basic train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Convert to torch
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long)

    # Desired param count
    desired_params = 10000

    # ---------------------------
    # 1) MLP with ~ desired_params
    # ---------------------------
    mlp_model = approximate_mlp_for_param_count(
        input_dim=X_train.shape[1],
        target_dim=len(np.unique(y)),
        desired_params=desired_params,
        hidden_init=64
    )
    mlp_params = count_parameters(mlp_model)
    print(f"[MLP] param_count = {mlp_params}")

    mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-3)
    loss_ce = nn.CrossEntropyLoss()

    mlp_train_acc = []
    mlp_val_acc   = []

    print(f"[MLP] Training for {num_epochs} epochs, batch_size={batch_size}")
    for epoch in range(num_epochs):
        # 1 epoch of mini-batch
        train_mlp_classification_minibatch(
            mlp_model, X_train_t, y_train_t, batch_size, mlp_optimizer, loss_ce
        )
        # Evaluate accuracy
        mlp_model.eval()
        with torch.no_grad():
            train_preds = mlp_model(X_train_t).argmax(dim=1).numpy()
            val_preds   = mlp_model(X_val_t).argmax(dim=1).numpy()
        acc_tr = accuracy_score(y_train, train_preds)
        acc_va = accuracy_score(y_val,   val_preds)
        mlp_train_acc.append(acc_tr)
        mlp_val_acc.append(acc_va)

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, MLP train_acc={acc_tr:.4f}, val_acc={acc_va:.4f}")

    # ---------------------------
    # 2) QKAN with ~ desired_params
    # ---------------------------
    kan_shape = approximate_kan_shape_for_param_count(
        input_dim=X_train.shape[1],
        target_dim=len(np.unique(y)),
        desired_params=desired_params,
        max_degree=10
    )
    config = FixedKANConfig(
        network_shape=kan_shape,
        max_degree=10,
        complexity_weight=0.0,
        trainable_coefficients=False,
        skip_qubo_for_hidden=False,
        default_hidden_degree=2
    )
    qkan = FixedKAN(config)
    qkan_params = count_parameters(qkan)
    print(f"[QKAN] param_count = {qkan_params}")

    # QUBO optimize (unweighted)
    from torch.nn.functional import one_hot
    y_train_onehot = one_hot(y_train_t, num_classes=len(np.unique(y))).float()
    qkan.optimize(X_train_t, y_train_onehot)  # picks degrees

    # Then mini-batch cross-entropy
    qkan_optimizer = torch.optim.Adam(
        [p for p in qkan.parameters() if p.requires_grad],
        lr=1e-3
    )
    qkan_train_acc = []
    qkan_val_acc   = []

    print(f"[QKAN] Training for {num_epochs} epochs, batch_size={batch_size}")
    for epoch in range(num_epochs):
        train_kan_classification_minibatch(
            qkan, X_train_t, y_train_t, batch_size, qkan_optimizer, loss_ce
        )
        # Evaluate accuracy
        qkan.eval()
        with torch.no_grad():
            train_preds = qkan(X_train_t).argmax(dim=1).numpy()
            val_preds   = qkan(X_val_t).argmax(dim=1).numpy()
        acc_tr = accuracy_score(y_train, train_preds)
        acc_va = accuracy_score(y_val,   val_preds)
        qkan_train_acc.append(acc_tr)
        qkan_val_acc.append(acc_va)

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, QKAN train_acc={acc_tr:.4f}, val_acc={acc_va:.4f}")

    # Final accuracies
    final_mlp_acc  = mlp_val_acc[-1]
    final_qkan_acc = qkan_val_acc[-1]
    print(f"[MLP] final val_acc={final_mlp_acc:.4f}")
    print(f"[QKAN] final val_acc={final_qkan_acc:.4f}")

    # 3) RandomForest / XGBoost (no param count measure here)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_val, rf.predict(X_val))
    print(f"[RF] covertype ACC={rf_acc:.4f}")

    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_acc = accuracy_score(y_val, xgb_model.predict(X_val))
    print(f"[XGB] covertype ACC={xgb_acc:.4f}")

    # 4) Plot MLP vs. QKAN (train+val accuracy)
    epochs_range = range(1, num_epochs+1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs_range, mlp_train_acc, label=f"MLP(train) [{mlp_params} params]", color='blue')
    plt.plot(epochs_range, mlp_val_acc,   label="MLP(val)", color='blue', linestyle='--')

    plt.plot(epochs_range, qkan_train_acc, label=f"QKAN(train) [{qkan_params} params]", color='orange')
    plt.plot(epochs_range, qkan_val_acc,   label="QKAN(val)", color='orange', linestyle='--')

    plt.title(f"covertype Classification\n(BatchSize={batch_size}, Epochs={num_epochs})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    os.makedirs("models_tabular_data", exist_ok=True)
    plt.savefig("models_tabular_data/covertype_mlp_qkan_minibatch.png")
    plt.show()
    print("[Done] Saved => models_tabular_data/covertype_mlp_qkan_minibatch.png")


#############################################################################
# 4) Regression on "house_sales" (log-target)
#############################################################################
def load_house_sales_data(normalize: bool = True, log_target: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    dataset = load_dataset(
        "inria-soda/tabular-benchmark",
        data_files="reg_num/house_sales.csv",
        split="train"
    )
    df = pd.DataFrame(dataset)
    label_col = "target" if "target" in df.columns else df.columns[-1]

    y = df[label_col].values.astype(np.float32)  # float32
    X = df.drop(columns=[label_col]).values.astype(np.float32)

    # Optionally log-transform
    if log_target:
        y = np.log1p(y)  # log(1+y)

    # Optionally normalize
    if normalize:
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)

    print(f"[house_sales] shape={df.shape}, label_col='{label_col}'")
    print(f"[house_sales] X.shape={X.shape}, y.shape={y.shape}")
    return X, y

def experiment_house_sales_regression(batch_size=4096, num_epochs=100):
    print("\n=== Regression on house_sales (numeric) ===")
    X, y = load_house_sales_data(normalize=True, log_target=True)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape)
    # Convert to torch
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(-1)

    desired_params = 10000

    # ---------------------------
    # 1) MLP
    # ---------------------------
    mlp_model = approximate_mlp_for_param_count(
        input_dim=X_train.shape[1],
        target_dim=1,
        desired_params=desired_params,
        hidden_init=64
    )
    mlp_params = count_parameters(mlp_model)
    print(f"[MLP] param_count = {mlp_params}")

    mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-3)
    loss_mse = nn.MSELoss()

    mlp_train_mse = []
    mlp_val_mse   = []

    print(f"[MLP] Training for {num_epochs} epochs, batch_size={batch_size}")
    for epoch in range(num_epochs):
        # One epoch of mini-batch
        train_mlp_regression_minibatch(
            mlp_model, X_train_t, y_train_t, batch_size, mlp_optimizer, loss_mse
        )
        # Evaluate MSE
        mlp_model.eval()
        with torch.no_grad():
            pred_tr = mlp_model(X_train_t)
            mse_tr  = loss_mse(pred_tr, y_train_t).item()
            pred_va = mlp_model(X_val_t)
            mse_va  = loss_mse(pred_va, y_val_t).item()
        mlp_train_mse.append(mse_tr)
        mlp_val_mse.append(mse_va)

        if (epoch+1) % 10 == 0:
            print(f"[MLP] epoch {epoch+1}/{num_epochs}, train_mse={mse_tr:.4f}, val_mse={mse_va:.4f}")

    # ---------------------------
    # 2) QKAN
    # ---------------------------
    kan_shape = approximate_kan_shape_for_param_count(
        input_dim=X_train.shape[1],
        target_dim=1,
        desired_params=desired_params,
        max_degree=7
    )
    config = FixedKANConfig(
        network_shape=kan_shape,
        max_degree=7,
        complexity_weight=0.0,
        trainable_coefficients=True,
        skip_qubo_for_hidden=False,
        default_hidden_degree=5
    )
    qkan = FixedKAN(config)
    qkan_params = count_parameters(qkan)
    print(f"[QKAN] param_count = {qkan_params}")

    # QUBO step (unweighted)
    qkan.optimize(X_train_t, y_train_t)

    qkan_optimizer = torch.optim.Adam(
        [p for p in qkan.parameters() if p.requires_grad],
        lr=1e-4
    )

    qkan_train_mse = []
    qkan_val_mse   = []

    print(f"[QKAN] Training for {num_epochs} epochs, batch_size={batch_size}")
    for epoch in range(num_epochs):
        train_kan_regression_minibatch(
            qkan, X_train_t, y_train_t, batch_size, qkan_optimizer, loss_mse
        )
        # Evaluate MSE
        qkan.eval()
        with torch.no_grad():
            pred_tr = qkan(X_train_t)
            mse_tr  = loss_mse(pred_tr, y_train_t).item()
            pred_va = qkan(X_val_t)
            mse_va  = loss_mse(pred_va, y_val_t).item()
        qkan_train_mse.append(mse_tr)
        qkan_val_mse.append(mse_va)

        if (epoch+1) % 10 == 0:
            print(f"[QKAN] epoch {epoch+1}/{num_epochs}, train_mse={mse_tr:.4f}, val_mse={mse_va:.4f}")

    # Final MSE
    final_mlp_mse  = mlp_val_mse[-1]
    final_qkan_mse = qkan_val_mse[-1]
    print(f"[MLP] final val_mse={final_mlp_mse:.4f}")
    print(f"[QKAN] final val_mse={final_qkan_mse:.4f}")

    # ---------------------------
    # 3) RandomForest / XGBoost
    # ---------------------------
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_mse = mean_squared_error(y_val, rf.predict(X_val))
    print(f"[RF] house_sales MSE={rf_mse:.4f}")

    xgb_reg = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_reg.fit(X_train, y_train)
    xgb_mse = mean_squared_error(y_val, xgb_reg.predict(X_val))
    print(f"[XGB] house_sales MSE={xgb_mse:.4f}")

    # ---------------------------
    # 4) Plot MLP vs. QKAN MSE
    # ---------------------------
    epochs_range = range(1, num_epochs+1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs_range, mlp_train_mse, label=f"MLP(train) [{mlp_params} params]", color='blue')
    plt.plot(epochs_range, mlp_val_mse,   label="MLP(val)", color='blue', linestyle='--')

    plt.plot(epochs_range, qkan_train_mse, label=f"QKAN(train) [{qkan_params} params]", color='orange')
    plt.plot(epochs_range, qkan_val_mse,   label="QKAN(val)", color='orange', linestyle='--')

    plt.title(f"House Sales Regression (log-target)\n(BatchSize={batch_size}, Epochs={num_epochs})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE(log scale)")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    os.makedirs("models_tabular_data", exist_ok=True)
    plt.savefig("models_tabular_data/house_sales_mlp_qkan_minibatch.png")
    plt.show()
    print("[Done] Saved => models_tabular_data/house_sales_mlp_qkan_minibatch.png")


#############################################################################
# Main
#############################################################################
def main():
    print("===== BEGIN TABULAR BENCHMARK EXPERIMENTS (Mini-Batch) =====")

    # #1) Classification test on covertype
    # experiment_covertype_classification(
    #     batch_size=4096,  # adjust if OOM
    #     num_epochs=500
    # )

    #2) Regression test on house_sales
    experiment_house_sales_regression(
        batch_size=4096,  # adjust if OOM
        num_epochs=500
    )

    print("===== END =====")


if __name__ == "__main__":
    main()