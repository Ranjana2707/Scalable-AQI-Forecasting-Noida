"""
src/deep_learning/trainer.py
==============================
Unified training loop for LSTM/GRU/CNN-LSTM models with:
- Mini-batch gradient descent (Adam optimiser)
- Early stopping on validation RMSE (patience-based, Keras semantics)
- Best-model checkpointing (restores best weights at the end)
- MSE loss with gradient computed on the *standardised* AQI target
  (target is z-scored using the SAME scaler fit on training y, to keep
  gradients well-scaled for the tanh/sigmoid-based recurrent units)
"""
from __future__ import annotations
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from src.deep_learning.layers import AdamOptimizer


@dataclass
class TrainingHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss:   List[float] = field(default_factory=list)
    val_rmse:   List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_epoch: int = 0


def train_model(
    model,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
    target_mean: float = 0.0,
    target_std: float = 1.0,
    verbose: bool = True,
    seed: int = 42,
) -> TrainingHistory:
    """
    Train a sequence model with mini-batch Adam, early stopping, and
    best-weight checkpointing.

    Parameters
    ----------
    model : LSTMModel | GRUModel | CNNLSTMModel
        Must implement forward(), backward(), get_all_params(), set_params().
    X_train, y_train, X_val, y_val : np.ndarray
        Pre-built sequences (y in raw AQI units).
    epochs : int
        Maximum training epochs.
    batch_size : int
    lr : float
        Adam learning rate.
    patience : int
        Epochs to wait for val RMSE improvement before stopping.
    target_mean, target_std : float
        Standardisation stats for the target (fit on training y).
    verbose : bool
    seed : int

    Returns
    -------
    TrainingHistory
        Also mutates `model` in-place to hold the best checkpointed weights.
    """
    rng = np.random.default_rng(seed)
    opt = AdamOptimizer(lr=lr)
    history = TrainingHistory()

    y_train_z = (y_train - target_mean) / target_std
    y_val_z   = (y_val - target_mean) / target_std

    n = len(X_train)
    best_val_rmse = np.inf
    best_params = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        idx = rng.permutation(n)
        X_shuf, y_shuf = X_train[idx], y_train_z[idx]

        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            xb = X_shuf[start:start+batch_size]
            yb = y_shuf[start:start+batch_size].reshape(-1, 1)

            yhat, cache = model.forward(xb)
            diff = yhat - yb
            loss = float(np.mean(diff ** 2))
            dyhat = (2.0 / len(xb)) * diff

            grads = model.backward(dyhat, cache)
            opt.step(model.get_all_params(), grads)

            epoch_loss += loss
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        yhat_val, _ = model.forward(X_val)
        val_diff = yhat_val.ravel() - y_val_z
        val_loss = float(np.mean(val_diff ** 2))
        val_pred_raw = yhat_val.ravel() * target_std + target_mean
        val_rmse = float(np.sqrt(np.mean((val_pred_raw - y_val) ** 2)))

        epoch_time = time.perf_counter() - t0
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_rmse.append(val_rmse)
        history.epoch_times.append(epoch_time)

        improved = val_rmse < best_val_rmse - 1e-4
        if improved:
            best_val_rmse = val_rmse
            best_params = {k: v.copy() for k, v in model.get_all_params().items()}
            history.best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            marker = " *" if improved else ""
            print(f"  Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.5f} | "
                  f"val_RMSE={val_rmse:7.3f}{marker} | {epoch_time:.2f}s")

        if epochs_no_improve >= patience:
            history.stopped_epoch = epoch
            if verbose:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs). "
                      f"Best epoch: {history.best_epoch} (val_RMSE={best_val_rmse:.3f})")
            break
    else:
        history.stopped_epoch = epochs

    # Restore best checkpoint
    if best_params is not None:
        model.set_params(best_params)

    return history


def predict_raw(model, X: np.ndarray, target_mean: float, target_std: float,
                batch_size: int = 256) -> np.ndarray:
    """Predict in raw AQI units (de-standardised)."""
    preds_z = model.predict(X, batch_size=batch_size)
    return preds_z * target_std + target_mean
