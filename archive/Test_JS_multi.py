import math
import unittest

import torch
from torch import nn

from CP_KAN import FixedKAN
from Test_js import weighted_r2


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
        Same as before: load data, build self.x_train_torch, self.y_train_torch, etc.
        Also build your CP-KAN (qkan) and a "tiny" MLP if needed.
        """
        # ... your existing setUp code ...
        pass

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

        # ------------------------
        # 2) Prepare MLPs of depths 2..5
        #    Keep param count ~ the same.
        # ------------------------
        # Let's get CP-KAN param count to use as a baseline
        cpk_param_count = 0
        for p in params_to_train:
            cpk_param_count += p.numel()

        # Alternatively, you can pick a fixed "target_params", e.g. 50k, 100k, etc.
        # For demonstration, we match CP-KAN's param count:
        target_params = cpk_param_count

        mlp_depths = [2, 3, 4, 5]
        mlp_models = {}
        for d in mlp_depths:
            mlp_models[d] = build_mlp_for_depth(
                input_dim=self.input_dim,
                output_dim=1,
                depth=d,
                target_params=target_params
            )
            print(f"MLP(depth={d}): param_count={count_parameters(mlp_models[d])}")

        # We'll store train/val R² for each depth
        mlp_r2_histories = {d: {"train": [], "val": []} for d in mlp_depths}

        # ------------------------
        # 3) Train each MLP
        # ------------------------
        for d in mlp_depths:
            print(f"\n==== [MLP-depth={d}] Weighted MSE Regression ====")
            mlp = mlp_models[d]
            optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=lr)

            for epoch in range(num_epochs):
                mlp.train()
                optimizer_mlp.zero_grad()
                y_pred = mlp(self.x_train_torch).squeeze(-1)
                # Weighted MSE
                numerator = torch.sum(self.w_train_torch * (self.y_train_torch - y_pred)**2)
                denominator = torch.sum(self.w_train_torch)
                loss = numerator / (denominator + 1e-12)
                loss.backward()
                optimizer_mlp.step()

                # Evaluate Weighted R² on train & val
                with torch.no_grad():
                    # train
                    y_pred_train = y_pred.cpu().numpy()
                    r2_train = weighted_r2(self.y_train_np, y_pred_train, self.w_train_np)
                    mlp_r2_histories[d]["train"].append(r2_train)

                    # val
                    y_pred_val = mlp(self.x_val_torch).squeeze(-1).cpu().numpy()
                    r2_val = weighted_r2(self.y_val_np, y_pred_val, self.w_val_np)
                    mlp_r2_histories[d]["val"].append(r2_val)

                if (epoch+1) % 50 == 0:
                    print(f"[MLP-depth={d}] Epoch {epoch+1}/{num_epochs}, Weighted MSE={loss.item():.6f}, Train R²={r2_train:.4f}, Val R²={r2_val:.4f}")

        # ------------------------
        # 4) Plot all on one figure
        #    We show Weighted R² vs. epochs (both train and val).
        # ------------------------
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))

        # Plot CP-KAN
        epochs = range(1, num_epochs+1)
        plt.plot(epochs, qkan_train_r2, label="CP-KAN (train)", color='blue')
        plt.plot(epochs, qkan_val_r2,   label="CP-KAN (val)",   color='blue', linestyle='--')

        # Colors for MLP lines
        color_map = {
            2: 'orange',
            3: 'green',
            4: 'red',
            5: 'purple'
        }

        for d in mlp_depths:
            train_r2 = mlp_r2_histories[d]["train"]
            val_r2   = mlp_r2_histories[d]["val"]
            plt.plot(epochs, train_r2, label=f"MLP(d={d}) train", color=color_map[d])
            plt.plot(epochs, val_r2,   label=f"MLP(d={d}) val",   color=color_map[d], linestyle='--')

        plt.title("CP-KAN vs. MLP Depths on Jane Street\n(Weighted R² vs. Epoch)")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted R²")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_path = "./models_janestreet/comparison_cpk_mlp_depths.png"
        plt.savefig(plot_path)
        plt.show()
        print(f"[Done] Combined plot saved => {plot_path}")