import unittest
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import transforms

from KAN_w_cumulative_polynomials import FixedKANConfig, FixedKAN


class TestQKANonCIFAR10(unittest.TestCase):
    def setUp(self):
        """
        Sets up:
         1) QKAN config => [3072 -> 32 -> 16 -> 10]
         2) Tiny MLP
         3) CIFAR-10 train subset + official test set
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------------------------
        # 1) QKAN Config
        # ---------------------------
        self.qkan_config = FixedKANConfig(
            network_shape=[3072, 32, 32, 10],
            max_degree=7,
            complexity_weight=0.0,
            trainable_coefficients=False,
            skip_qubo_for_hidden=False
        )

        # ---------------------------
        # 2) Tiny MLP Baseline
        # ---------------------------
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3072, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10)
        ).to(self.device)

        # ---------------------------
        # 3) CIFAR-10 Data
        # ---------------------------
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # === A) Train Subset ===
        cifar_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        subset_indices = list(range(25000))  # e.g. 20k out of 50k
        train_subset = Subset(cifar_train, subset_indices)

        loader_train = DataLoader(train_subset, batch_size=len(train_subset), shuffle=False)
        images_train, labels_train = next(iter(loader_train))

        self.x_train = images_train.view(images_train.size(0), -1).float().to(self.device)  # [N, 3072]
        self.labels_train = labels_train.to(self.device)                                    # [N]

        # For QUBO: one-hot
        num_classes = 10
        y_data_oh = torch.zeros((labels_train.size(0), num_classes), device=self.device)
        y_data_oh.scatter_(1, labels_train.unsqueeze(1), 1.0)
        self.y_train_onehot = y_data_oh  # [N, 10]

        # === B) Test Set (official 10k) ===
        cifar_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        loader_test = DataLoader(cifar_test, batch_size=len(cifar_test), shuffle=False)
        images_test, labels_test = next(iter(loader_test))

        self.x_test = images_test.view(images_test.size(0), -1).float().to(self.device)  # [N_test, 3072]
        self.labels_test = labels_test.to(self.device)                                   # [N_test]

        # Make sure models directory exists
        os.makedirs("./models", exist_ok=True)

    def test_1_qkan_training_mse(self):
        """
        1) QUBO => MSE-based training on the 20k train subset.
        2) Evaluate on official 10k test set => final test accuracy.
        3) Save the QKAN model + plot the training loss.
        """
        print("\n==== [QKAN] MSE-based training => Evaluate on official test set ====")
        qkan = FixedKAN(self.qkan_config).to(self.device)

        # 1) QUBO => picks polynomial degrees
        qkan.optimize(self.x_train, self.y_train_onehot)

        # 2) MSE training
        num_epochs = 500
        lr = 1e-4
        train_losses = []
        optimizer = None  # We'll replicate the train loop manually to record losses

        # ---------------------------
        # Manual training loop
        # (We override qkan.train_model to store the loss for plotting)
        # ---------------------------
        from torch.optim import Adam
        optimizer = Adam(self._gather_trainable_params(qkan), lr=lr)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            y_pred = qkan.forward(self.x_train)  # [N,10]
            mse = torch.mean((y_pred - self.y_train_onehot)**2)
            loss = mse  # no complexity_weight for now

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, MSE={mse.item():.6f}")

        # Evaluate on test set
        acc_test = self._compute_accuracy(qkan, self.x_test, self.labels_test)
        print(f"[QKAN MSE] Test-set accuracy: {acc_test:.2f}%")

        # Save model with test accuracy in filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"./models/qkan_mse_{acc_test:.2f}_{timestamp}.pth"
        qkan.save_model(save_path)
        print(f"QKAN MSE model saved to {save_path}")

        # Plot training loss
        plt.figure()
        plt.plot(train_losses, label="Train Loss (MSE)")
        plt.title(f"QKAN MSE Training Loss (Test Acc={acc_test:.2f}%)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plot_path = f"./plots/qkan_mse_loss_{acc_test:.2f}_{timestamp}.png"
        plt.savefig(plot_path)
        print(f"Saved loss plot to {plot_path}")
        plt.show()  # optional if you want interactive

    def test_2_tiny_mlp_training_mse(self):
        """
        Tiny MLP => MSE-based training => Evaluate on official test set.
        Save model + plot training loss.
        """
        print("\n==== [Tiny MLP] MSE-based training => Evaluate on official test set ====")
        mlp_optimizer = torch.optim.Adam(self.mlp.parameters(), lr=1e-3)
        num_epochs = 500
        train_losses = []

        for epoch in range(num_epochs):
            mlp_optimizer.zero_grad()
            preds = self.mlp(self.x_train)  # [N,10]
            mse = torch.mean((preds - self.y_train_onehot)**2)
            mse.backward()
            mlp_optimizer.step()
            train_losses.append(mse.item())

            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, MSE={mse.item():.6f}")

        # Evaluate on test set
        acc_test = self._compute_accuracy(self.mlp, self.x_test, self.labels_test)
        print(f"[Tiny MLP MSE] Test-set accuracy: {acc_test:.2f}%")

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"./models/tinymlp_mse_{acc_test:.2f}_{timestamp}.pt"
        torch.save(self.mlp.state_dict(), save_path)
        print(f"Tiny MLP (MSE) model saved to {save_path}")

        # Plot training loss
        plt.figure()
        plt.plot(train_losses, label="Train Loss (MSE)")
        plt.title(f"Tiny MLP MSE Training Loss (Test Acc={acc_test:.2f}%)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plot_path = f"./plots/tinymlp_mse_loss_{acc_test:.2f}_{timestamp}.png"
        plt.savefig(plot_path)
        print(f"Saved MLP loss plot to {plot_path}")
        plt.show()

    def test_3_qkan_training_cross_entropy(self):
        """
        QUBO => cross-entropy training => test set evaluation.
        Save model + plot CE loss.
        """
        print("\n==== [QKAN] Cross-Entropy => Evaluate on official test set ====")
        qkan = FixedKAN(self.qkan_config).to(self.device)

        # QUBO using one-hot
        qkan.optimize(self.x_train, self.y_train_onehot)

        # Now cross-entropy training on integer labels
        num_epochs = 500
        lr = 1e-4
        train_losses = []
        from torch.optim import Adam
        optimizer = Adam(self._gather_trainable_params(qkan), lr=lr)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            logits = qkan(self.x_train)  # [N,10]
            ce_loss = torch.nn.functional.cross_entropy(logits, self.labels_train.long())

            loss = ce_loss  # no complexity penalty for now
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            if (epoch+1) % 5 == 0:
                print(f"[CE Training] Epoch {epoch+1}/{num_epochs}, CE={ce_loss.item():.6f}")

        # Evaluate on test set
        acc_test = self._compute_accuracy(qkan, self.x_test, self.labels_test)
        print(f"[QKAN CE] Test-set accuracy: {acc_test:.2f}%")

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"./models/qkan_ce_{acc_test:.2f}_{timestamp}.pth"
        qkan.save_model(save_path)
        print(f"QKAN CE model saved to {save_path}")

        # Plot training loss
        plt.figure()
        plt.plot(train_losses, label="Train Loss (CE)")
        plt.title(f"QKAN CE Training Loss (Test Acc={acc_test:.2f}%)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plot_path = f"./plots/qkan_ce_loss_{acc_test:.2f}_{timestamp}.png"
        plt.savefig(plot_path)
        print(f"Saved CE loss plot to {plot_path}")
        plt.show()

    def _compute_accuracy(self, model: torch.nn.Module, x_data: torch.Tensor, labels_int: torch.Tensor):
        """
        Evaluate classification accuracy => (pred == label_int).mean().
        """
        with torch.no_grad():
            preds = model(x_data)  # [N,10]
            predicted_labels = preds.argmax(dim=1)
            acc = (predicted_labels == labels_int).float().mean().item() * 100
        return acc

    def _gather_trainable_params(self, qkan: FixedKAN):
        """
        Helper to gather (w, b, combine_W, combine_b)
        and optionally polynomial coefficients if trainable_coefficients=True.
        """
        params = []
        for layer in qkan.layers:
            params.extend([layer.combine_W, layer.combine_b])
            for neuron in layer.neurons:
                params.extend([neuron.w, neuron.b])
                if qkan.config.trainable_coefficients and neuron.coefficients is not None:
                    params.extend(list(neuron.coefficients))
        return params

    def _visualize_predictions(self, model, title="[Model] Predictions"):
        """
        If you want to show some images + predictions from test set, do so here.
        For brevity, we skip this to keep the code short.
        """
        pass


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)