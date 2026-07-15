"""
src/deep_learning/layers.py
==============================
NumPy implementations of LSTM, GRU, Conv1D layers with full forward
and backward (BPTT) passes, plus an Adam optimiser.

All layers follow the standard recurrent-network equations
(Hochreiter & Schmidhuber 1997 for LSTM; Cho et al. 2014 for GRU)
implemented with vectorised batch operations for reasonable CPU speed.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, List, Optional


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))

def dsigmoid(y: np.ndarray) -> np.ndarray:
    """Derivative w.r.t. pre-activation, given y = sigmoid(x)."""
    return y * (1 - y)

def dtanh(y: np.ndarray) -> np.ndarray:
    """Derivative w.r.t. pre-activation, given y = tanh(x)."""
    return 1 - y ** 2

def xavier_init(shape: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    fan_in, fan_out = shape[0], shape[-1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape)


class AdamOptimizer:
    """Adam optimiser shared across all layers in a model."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}
        self.t = 0

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        self.t += 1
        for k in params:
            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])
            g = np.clip(grads[k], -5.0, 5.0)  # gradient clipping
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g ** 2)
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class LSTMLayer:
    """
    Single-layer LSTM with full BPTT.

    Gates: forget (f), input (i), candidate (g), output (o).
    Parameters stored as combined weight matrices for efficiency.
    """

    def __init__(self, input_dim: int, hidden_dim: int, rng: np.random.Generator):
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        d = hidden_dim
        # Combined weight: [Wf, Wi, Wg, Wo] each (input_dim+hidden_dim, hidden_dim)
        self.params = {
            "Wf": xavier_init((input_dim + d, d), rng), "bf": np.ones(d) * 1.0,  # forget bias init=1 (standard trick)
            "Wi": xavier_init((input_dim + d, d), rng), "bi": np.zeros(d),
            "Wg": xavier_init((input_dim + d, d), rng), "bg": np.zeros(d),
            "Wo": xavier_init((input_dim + d, d), rng), "bo": np.zeros(d),
        }

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        X: (batch, T, input_dim)
        Returns: h_seq (batch, T, hidden_dim), cache for backward
        """
        B, T, _ = X.shape
        d = self.hidden_dim
        h = np.zeros((B, d)); c = np.zeros((B, d))
        h_seq = np.zeros((B, T, d))
        cache = {"X": X, "h": [h.copy()], "c": [c.copy()],
                 "f": [], "i": [], "g": [], "o": []}

        for t in range(T):
            xt = X[:, t, :]
            concat = np.concatenate([xt, h], axis=1)
            f = sigmoid(concat @ self.params["Wf"] + self.params["bf"])
            i = sigmoid(concat @ self.params["Wi"] + self.params["bi"])
            g = np.tanh(concat @ self.params["Wg"] + self.params["bg"])
            o = sigmoid(concat @ self.params["Wo"] + self.params["bo"])
            c = f * c + i * g
            h = o * np.tanh(c)
            h_seq[:, t, :] = h
            cache["h"].append(h.copy()); cache["c"].append(c.copy())
            cache["f"].append(f); cache["i"].append(i)
            cache["g"].append(g); cache["o"].append(o)

        return h_seq, cache

    def backward(self, dh_seq: np.ndarray, cache: Dict) -> Tuple[Dict, np.ndarray]:
        """
        dh_seq: (batch, T, hidden_dim) gradient w.r.t. each timestep's output
        Returns: grads dict, dX (batch, T, input_dim)
        """
        X = cache["X"]; B, T, in_dim = X.shape; d = self.hidden_dim
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        dX = np.zeros_like(X)
        dh_next = np.zeros((B, d)); dc_next = np.zeros((B, d))

        for t in reversed(range(T)):
            h_prev = cache["h"][t]; c_prev = cache["c"][t]; c_t = cache["c"][t+1]
            f, i, g, o = cache["f"][t], cache["i"][t], cache["g"][t], cache["o"][t]
            xt = X[:, t, :]
            concat = np.concatenate([xt, h_prev], axis=1)

            dh = dh_seq[:, t, :] + dh_next
            do = dh * np.tanh(c_t)
            dc = dh * o * dtanh(np.tanh(c_t)) + dc_next

            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dc_next = dc * f

            df_pre = df * dsigmoid(f)
            di_pre = di * dsigmoid(i)
            dg_pre = dg * dtanh(g)
            do_pre = do * dsigmoid(o)

            grads["Wf"] += concat.T @ df_pre; grads["bf"] += df_pre.sum(0)
            grads["Wi"] += concat.T @ di_pre; grads["bi"] += di_pre.sum(0)
            grads["Wg"] += concat.T @ dg_pre; grads["bg"] += dg_pre.sum(0)
            grads["Wo"] += concat.T @ do_pre; grads["bo"] += do_pre.sum(0)

            dconcat = (df_pre @ self.params["Wf"].T + di_pre @ self.params["Wi"].T +
                      dg_pre @ self.params["Wg"].T + do_pre @ self.params["Wo"].T)
            dX[:, t, :] = dconcat[:, :in_dim]
            dh_next = dconcat[:, in_dim:]

        return grads, dX


class GRULayer:
    """Single-layer GRU with full BPTT (Cho et al. 2014 formulation)."""

    def __init__(self, input_dim: int, hidden_dim: int, rng: np.random.Generator):
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        d = hidden_dim
        self.params = {
            "Wz": xavier_init((input_dim + d, d), rng), "bz": np.zeros(d),
            "Wr": xavier_init((input_dim + d, d), rng), "br": np.zeros(d),
            "Wh": xavier_init((input_dim + d, d), rng), "bh": np.zeros(d),
        }

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        B, T, _ = X.shape; d = self.hidden_dim
        h = np.zeros((B, d))
        h_seq = np.zeros((B, T, d))
        cache = {"X": X, "h": [h.copy()], "z": [], "r": [], "hhat": []}

        for t in range(T):
            xt = X[:, t, :]
            concat = np.concatenate([xt, h], axis=1)
            z = sigmoid(concat @ self.params["Wz"] + self.params["bz"])
            r = sigmoid(concat @ self.params["Wr"] + self.params["br"])
            concat_r = np.concatenate([xt, r * h], axis=1)
            hhat = np.tanh(concat_r @ self.params["Wh"] + self.params["bh"])
            h = (1 - z) * h + z * hhat
            h_seq[:, t, :] = h
            cache["h"].append(h.copy())
            cache["z"].append(z); cache["r"].append(r); cache["hhat"].append(hhat)

        return h_seq, cache

    def backward(self, dh_seq: np.ndarray, cache: Dict) -> Tuple[Dict, np.ndarray]:
        X = cache["X"]; B, T, in_dim = X.shape; d = self.hidden_dim
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        dX = np.zeros_like(X)
        dh_next = np.zeros((B, d))

        for t in reversed(range(T)):
            h_prev = cache["h"][t]
            z, r, hhat = cache["z"][t], cache["r"][t], cache["hhat"][t]
            xt = X[:, t, :]
            concat = np.concatenate([xt, h_prev], axis=1)

            dh = dh_seq[:, t, :] + dh_next
            dz = dh * (hhat - h_prev)
            dhhat = dh * z
            dh_prev_partial = dh * (1 - z)

            dhhat_pre = dhhat * dtanh(hhat)
            concat_r = np.concatenate([xt, r * h_prev], axis=1)
            grads["Wh"] += concat_r.T @ dhhat_pre
            grads["bh"] += dhhat_pre.sum(0)
            dconcat_r = dhhat_pre @ self.params["Wh"].T
            dxt_from_h = dconcat_r[:, :in_dim]
            dr_h = dconcat_r[:, in_dim:]
            dr = dr_h * h_prev
            dh_prev_from_r = dr_h * r

            dz_pre = dz * dsigmoid(z)
            dr_pre = dr * dsigmoid(r)

            grads["Wz"] += concat.T @ dz_pre; grads["bz"] += dz_pre.sum(0)
            grads["Wr"] += concat.T @ dr_pre; grads["br"] += dr_pre.sum(0)

            dconcat_zr = dz_pre @ self.params["Wz"].T + dr_pre @ self.params["Wr"].T
            dxt_from_zr = dconcat_zr[:, :in_dim]
            dh_prev_from_zr = dconcat_zr[:, in_dim:]

            dX[:, t, :] = dxt_from_h + dxt_from_zr
            dh_next = dh_prev_partial + dh_prev_from_r + dh_prev_from_zr

        return grads, dX


class Conv1DLayer:
    """
    Simple 1D convolution along the time axis (valid padding),
    used as the CNN front-end in CNN-LSTM.
    """

    def __init__(self, input_dim: int, n_filters: int, kernel_size: int,
                rng: np.random.Generator):
        self.input_dim, self.n_filters, self.k = input_dim, n_filters, kernel_size
        self.params = {
            "W": xavier_init((kernel_size * input_dim, n_filters), rng),
            "b": np.zeros(n_filters),
        }

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """X: (batch, T, input_dim) -> out: (batch, T-k+1, n_filters), ReLU applied."""
        B, T, C = X.shape; k = self.k
        T_out = T - k + 1
        # im2col
        cols = np.zeros((B, T_out, k * C))
        for t in range(T_out):
            cols[:, t, :] = X[:, t:t+k, :].reshape(B, -1)
        z = cols @ self.params["W"] + self.params["b"]
        out = np.maximum(z, 0)  # ReLU
        cache = {"X": X, "cols": cols, "z": z, "out": out}
        return out, cache

    def backward(self, dout: np.ndarray, cache: Dict) -> Tuple[Dict, np.ndarray]:
        X, cols, z = cache["X"], cache["cols"], cache["z"]
        B, T, C = X.shape; k = self.k; T_out = T - k + 1
        drelu = dout * (z > 0)
        grads = {
            "W": cols.reshape(-1, k*C).T @ drelu.reshape(-1, self.n_filters),
            "b": drelu.sum(axis=(0,1)),
        }
        dcols = drelu @ self.params["W"].T  # (B, T_out, k*C)
        dX = np.zeros_like(X)
        for t in range(T_out):
            dX[:, t:t+k, :] += dcols[:, t, :].reshape(B, k, C)
        return grads, dX


class DenseLayer:
    """Fully connected output layer (linear, no activation — regression head)."""

    def __init__(self, input_dim: int, output_dim: int, rng: np.random.Generator):
        self.params = {
            "W": xavier_init((input_dim, output_dim), rng),
            "b": np.zeros(output_dim),
        }

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        out = X @ self.params["W"] + self.params["b"]
        return out, {"X": X}

    def backward(self, dout: np.ndarray, cache: Dict) -> Tuple[Dict, np.ndarray]:
        X = cache["X"]
        grads = {"W": X.T @ dout, "b": dout.sum(axis=0)}
        dX = dout @ self.params["W"].T
        return grads, dX
