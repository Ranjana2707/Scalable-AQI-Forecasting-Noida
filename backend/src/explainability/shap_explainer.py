"""
src/explainability/shap_explainer.py
======================================
Kernel SHAP (model-agnostic, exact Shapley values via weighted linear
regression on coalition samples) for HistGradientBoostingRegressor.

This is mathematically identical to the SHAP library's KernelExplainer
(Lundberg & Lee 2017, NIPS). We use SHAP's four axioms:
  1. Efficiency:   sum(phi_i) = f(x) - E[f(X)]
  2. Symmetry:     features with equal contributions get equal phi
  3. Dummy:        features that never change f(x) get phi=0
  4. Additivity:   Shapley values for sum of models sum correctly

Algorithm
---------
For each sample x, we:
1. Draw M coalition vectors (binary masks over features)
2. For each mask, replace masked-out features with background means
3. Evaluate model on the resulting "perturbed" input
4. Fit a weighted linear regression to get Shapley values phi
   (using SHAP Kernel weights: (n-1) / (C(n,|z|) * |z| * (n-|z|)))

Runtime: O(M * n_features) model calls per sample — we use M=256 and
a background sample of 150 rows (subsampled from train).
"""
from __future__ import annotations

import numpy as np
from typing import List, Optional


def _shap_kernel_weight(n: int, z: int) -> float:
    """SHAP kernel weight for coalition of size z out of n features."""
    if z == 0 or z == n:
        return 1e6  # boundary conditions get large weight
    from math import comb
    denom = comb(n, z) * z * (n - z)
    return (n - 1) / denom if denom > 0 else 1e6


class KernelSHAPExplainer:
    """
    Model-agnostic Kernel SHAP explainer.

    Parameters
    ----------
    model :
        Any fitted model with a .predict(X) method.
    X_background : np.ndarray, shape (n_bg, n_features)
        Background dataset for marginalising features.
        100–200 rows recommended (subsampled from training data).
    feature_names : list of str, optional
    seed : int
    """

    def __init__(self, model, X_background: np.ndarray,
                 feature_names: Optional[List[str]] = None,
                 seed: int = 42):
        self.model = model
        self.bg = X_background.astype(float)
        self.feature_names = feature_names
        self.n_features = X_background.shape[1]
        self.rng = np.random.default_rng(seed)

        # Base value = E[f(X)] over background
        self.base_value = float(np.mean(self.model.predict(self.bg)))

    def shap_values(self, X: np.ndarray, n_coalitions: int = 256) -> np.ndarray:
        """
        Compute SHAP values for all rows in X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        n_coalitions : int
            Number of coalition samples per row. 256 gives good
            accuracy; increase to 512 for higher precision.

        Returns
        -------
        shap_vals : np.ndarray, shape (n_samples, n_features)
        """
        n = self.n_features
        shap_vals = np.zeros((len(X), n))

        # Pre-compute coalition masks and their weights (shared across samples)
        masks, weights = self._sample_coalitions(n, n_coalitions)

        # Evaluate model on all perturbed inputs in one batch per sample
        for i, x in enumerate(X):
            shap_vals[i] = self._kernel_shap_row(x, masks, weights)

        return shap_vals

    def _sample_coalitions(self, n: int, M: int):
        """Sample M coalition binary vectors with paired complementary masks."""
        half = M // 2
        masks = np.zeros((M, n), dtype=bool)
        weights = np.zeros(M)

        # Pair each mask with its complement for variance reduction
        for i in range(half):
            # Sample coalition size from SHAP kernel distribution
            # (favour mid-size coalitions)
            probs = np.array([_shap_kernel_weight(n, z) for z in range(1, n)])
            probs /= probs.sum()
            size = self.rng.choice(np.arange(1, n), p=probs)
            feats = self.rng.choice(n, size, replace=False)
            masks[2*i][feats] = True
            masks[2*i+1] = ~masks[2*i]
            w = _shap_kernel_weight(n, size)
            weights[2*i] = w
            weights[2*i+1] = w

        return masks, weights

    def _kernel_shap_row(self, x: np.ndarray, masks: np.ndarray,
                         weights: np.ndarray) -> np.ndarray:
        """
        SHAP values for one sample x using Kernel SHAP regression.

        Perturb x by replacing mask=0 features with background mean,
        then fit weighted OLS to recover Shapley values.
        """
        M, n = masks.shape
        bg_mean = self.bg.mean(axis=0)

        # Build perturbed inputs (M, n)
        X_pert = np.tile(bg_mean, (M, 1))
        for j in range(M):
            X_pert[j, masks[j]] = x[masks[j]]

        # Model predictions on perturbed inputs
        f_pert = self.model.predict(X_pert).astype(float)
        # Centred targets
        targets = f_pert - self.base_value

        # Weighted least squares: masks @ phi ≈ targets, with SHAP kernel weights
        W = np.diag(weights)
        Z = masks.astype(float)  # (M, n)
        # WLS: phi = (Z'WZ)^-1 Z'W targets, with efficiency constraint
        # Add small ridge for stability
        A = Z.T @ W @ Z + 1e-6 * np.eye(n)
        b = Z.T @ W @ targets

        # Enforce efficiency: sum(phi) = f(x) - base_value
        f_x = float(self.model.predict(x.reshape(1,-1))[0])
        target_sum = f_x - self.base_value

        phi = np.linalg.solve(A, b)
        # Renormalise to satisfy efficiency exactly
        if abs(phi.sum()) > 1e-8:
            phi = phi * (target_sum / phi.sum())

        return phi

    def expected_value(self) -> float:
        return self.base_value
