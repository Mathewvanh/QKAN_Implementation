import torch
import torch.nn as nn
import torch.nn.functional as F
import polars as pl
import torch.optim as optim
from torch import device
from torch.utils.data import TensorDataset, DataLoader


def train_mlp(model, x_train: pl.DataFrame, y_train: pl.DataFrame, weights: pl.DataFrame=None,
              x_val: pl.DataFrame=None, y_val: pl.DataFrame=None, w_val: pl.DataFrame=None,
              batch_size=32, n_epochs=10):
    """Train MLP model with improved stability"""
    x_tensor = x_train.to_torch()
    y_tensor = y_train.to_torch(dtype=pl.Float64)

    if weights is not None:
        # Normalize weights to avoid extreme values
        weights_tensor = weights.to_torch(dtype=pl.Float64)
        weights_tensor = weights_tensor / weights_tensor.mean()
        dataset = TensorDataset(x_tensor, y_tensor, weights_tensor)
    else:
        dataset = TensorDataset(x_tensor, y_tensor)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Convert model to double precision and add batch norm
    model = nn.Sequential(
        nn.BatchNorm1d(x_tensor.shape[1]),
        *[layer for pair in zip(
            list(model.children())[:-1],
            [nn.Tanh() for _ in range(len(list(model.children()))-1)]
        ) for layer in pair],
        list(model.children())[-1]  # Last linear layer
    ).double()

    # Use smaller learning rate with scheduler
    optimizer = optim.Adam(model.parameters(), lr=8e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Add gradient clipping
    max_grad_norm = 1.0

    model.train()
    early_stopping_patience = 10
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    scores = []
    compr2_scores = []

    for epoch in range(n_epochs):
        total_loss = 0
        for batch in loader:
            if weights is not None:
                x_batch, y_batch, w_batch = batch
            else:
                x_batch, y_batch = batch
                w_batch = None

            optimizer.zero_grad()
            pred = model(x_batch)

            if w_batch is not None:
                # Modified weighted loss calculation
                loss = torch.mean(w_batch.view(-1, 1) * (pred - y_batch.view(-1,1))**2)
            else:
                loss = F.mse_loss(pred, y_batch.view(-1,1))

            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            total_loss += loss.item()

            # Monitor for debugging
            # if epoch % 2 == 0:
            #     with torch.no_grad():
            #         for name, param in model.named_parameters():
            #             if param.grad is not None:
            #                 print(f"{name} - grad stats: mean={param.grad.mean():.2e}, "
            #                       f"std={param.grad.std():.2e}, max={param.grad.max():.2e}")

        # Validation step
        if x_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                x_val_tensor = x_val.to_torch()
                y_val_tensor = y_val.to_torch(dtype=pl.Float64)
                val_pred = model(x_val_tensor)

                val_mse = F.mse_loss(val_pred, y_val_tensor.view(-1,1)).item()

                if w_val is not None:
                    w_val_tensor = w_val.to_torch(dtype=pl.Float64)
                    w_val_tensor = w_val_tensor / w_val_tensor.mean()  # Normalize validation weights too
                    weighted_squared_errors = w_val_tensor.view(-1, 1) * (y_val_tensor.view(-1,1) - val_pred) ** 2
                    weighted_y_squared = w_val_tensor.view(-1, 1) * y_val_tensor.view(-1,1) ** 2
                    compr2 = 1 - weighted_squared_errors.sum() / weighted_y_squared.sum()
                    compr2_scores.append(compr2.item())

                scores.append(val_mse)
                if val_mse < best_val_loss:
                    best_val_loss = val_mse
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    print(f'Early stopping triggered at epoch {epoch}')
                    break

                # Save best model
                if val_mse < best_val_loss:
                    best_val_loss = val_mse
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Update learning rate based on validation loss
            scheduler.step(val_mse)
            model.train()

        print(f'Epoch {epoch}, Loss: {total_loss/len(loader):.4f}, '
              f'Val MSE: {val_mse:.4f}, LR: {optimizer.param_groups[0]["lr"]:.2e}',
              f'Val comprR2: {compr2_scores[-1]:.4f}')

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return scores, compr2_scores
