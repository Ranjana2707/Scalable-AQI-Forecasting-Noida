"""
src/deep_learning/models.py
==============================
Three sequence model architectures for AQI forecasting:
LSTM, GRU, and CNN-LSTM, all built from the layers in layers.py.

Each model:
- Takes (batch, window_size, n_features) input
- Outputs a single scalar AQI prediction per sample (last timestep's
  hidden state -> Dense(1))
- Supports forward(), backward(), and a unified train loop with
  early stopping + checkpointing in trainer.py
"""
from __future__ import annotations
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional

from src.deep_learning.layers import (
    LSTMLayer, GRULayer, Conv1DLayer, DenseLayer, AdamOptimizer
)


class BaseSequenceModel:
    """Shared interface for LSTM / GRU / CNN-LSTM models."""

    def get_all_params(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        preds = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            yhat, _ = self.forward(batch)
            preds.append(yhat.ravel())
        return np.concatenate(preds)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "BaseSequenceModel":
        with open(path, "rb") as f:
            return pickle.load(f)


class LSTMModel(BaseSequenceModel):
    """LSTM -> last-timestep hidden state -> Dense(1) regression head."""

    def __init__(self, n_features: int, hidden_dim: int = 32, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.lstm  = LSTMLayer(n_features, hidden_dim, rng)
        self.dense = DenseLayer(hidden_dim, 1, rng)
        self.name = f"LSTM(hidden={hidden_dim})"

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        h_seq, lstm_cache = self.lstm.forward(X)
        h_last = h_seq[:, -1, :]
        yhat, dense_cache = self.dense.forward(h_last)
        return yhat, {"lstm": lstm_cache, "dense": dense_cache, "h_seq_shape": h_seq.shape}

    def backward(self, dyhat: np.ndarray, cache: Dict) -> Dict:
        dense_grads, dh_last = self.dense.backward(dyhat, cache["dense"])
        B, T, d = cache["h_seq_shape"]
        dh_seq = np.zeros((B, T, d))
        dh_seq[:, -1, :] = dh_last
        lstm_grads, _ = self.lstm.backward(dh_seq, cache["lstm"])
        return {**{f"lstm.{k}": v for k,v in lstm_grads.items()},
                **{f"dense.{k}": v for k,v in dense_grads.items()}}

    def get_all_params(self) -> Dict[str, np.ndarray]:
        return {**{f"lstm.{k}": v for k,v in self.lstm.params.items()},
                **{f"dense.{k}": v for k,v in self.dense.params.items()}}

    def set_params(self, flat: Dict[str, np.ndarray]) -> None:
        for k, v in flat.items():
            layer, pname = k.split(".")
            getattr(self, layer).params[pname] = v


class GRUModel(BaseSequenceModel):
    """GRU -> last-timestep hidden state -> Dense(1) regression head."""

    def __init__(self, n_features: int, hidden_dim: int = 32, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.gru   = GRULayer(n_features, hidden_dim, rng)
        self.dense = DenseLayer(hidden_dim, 1, rng)
        self.name = f"GRU(hidden={hidden_dim})"

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        h_seq, gru_cache = self.gru.forward(X)
        h_last = h_seq[:, -1, :]
        yhat, dense_cache = self.dense.forward(h_last)
        return yhat, {"gru": gru_cache, "dense": dense_cache, "h_seq_shape": h_seq.shape}

    def backward(self, dyhat: np.ndarray, cache: Dict) -> Dict:
        dense_grads, dh_last = self.dense.backward(dyhat, cache["dense"])
        B, T, d = cache["h_seq_shape"]
        dh_seq = np.zeros((B, T, d))
        dh_seq[:, -1, :] = dh_last
        gru_grads, _ = self.gru.backward(dh_seq, cache["gru"])
        return {**{f"gru.{k}": v for k,v in gru_grads.items()},
                **{f"dense.{k}": v for k,v in dense_grads.items()}}

    def get_all_params(self) -> Dict[str, np.ndarray]:
        return {**{f"gru.{k}": v for k,v in self.gru.params.items()},
                **{f"dense.{k}": v for k,v in self.dense.params.items()}}

    def set_params(self, flat: Dict[str, np.ndarray]) -> None:
        for k, v in flat.items():
            layer, pname = k.split(".")
            getattr(self, layer).params[pname] = v


class CNNLSTMModel(BaseSequenceModel):
    """
    Conv1D(causal local feature extraction) -> LSTM -> Dense(1).

    The convolution extracts short-range local patterns (e.g., 3-day
    pollution episode shapes) before the LSTM models longer-range
    temporal dependencies over the resulting feature sequence.
    """

    def __init__(self, n_features: int, n_filters: int = 16, kernel_size: int = 3,
                hidden_dim: int = 32, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.conv  = Conv1DLayer(n_features, n_filters, kernel_size, rng)
        self.lstm  = LSTMLayer(n_filters, hidden_dim, rng)
        self.dense = DenseLayer(hidden_dim, 1, rng)
        self.name = f"CNN-LSTM(filters={n_filters},k={kernel_size},hidden={hidden_dim})"

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        conv_out, conv_cache = self.conv.forward(X)
        h_seq, lstm_cache = self.lstm.forward(conv_out)
        h_last = h_seq[:, -1, :]
        yhat, dense_cache = self.dense.forward(h_last)
        return yhat, {
            "conv": conv_cache, "lstm": lstm_cache, "dense": dense_cache,
            "h_seq_shape": h_seq.shape,
        }

    def backward(self, dyhat: np.ndarray, cache: Dict) -> Dict:
        dense_grads, dh_last = self.dense.backward(dyhat, cache["dense"])
        B, T, d = cache["h_seq_shape"]
        dh_seq = np.zeros((B, T, d))
        dh_seq[:, -1, :] = dh_last
        lstm_grads, dconv_out = self.lstm.backward(dh_seq, cache["lstm"])
        conv_grads, _ = self.conv.backward(dconv_out, cache["conv"])
        return {**{f"conv.{k}": v for k,v in conv_grads.items()},
                **{f"lstm.{k}": v for k,v in lstm_grads.items()},
                **{f"dense.{k}": v for k,v in dense_grads.items()}}

    def get_all_params(self) -> Dict[str, np.ndarray]:
        return {**{f"conv.{k}": v for k,v in self.conv.params.items()},
                **{f"lstm.{k}": v for k,v in self.lstm.params.items()},
                **{f"dense.{k}": v for k,v in self.dense.params.items()}}

    def set_params(self, flat: Dict[str, np.ndarray]) -> None:
        for k, v in flat.items():
            layer, pname = k.split(".")
            getattr(self, layer).params[pname] = v
