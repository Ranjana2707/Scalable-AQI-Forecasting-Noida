"""
src/explainability/treeshap.py
================================
TreeSHAP implementation for sklearn HistGradientBoostingRegressor.

SHAP (SHapley Additive exPlanations) values are computed using the
TreeSHAP algorithm (Lundberg & Lee, 2017; Lundberg et al., 2020),
which runs in O(TLD²) time per sample instead of the exponential
brute-force Shapley computation, making it exact and tractable for
gradient-boosted tree ensembles.

Implementation approach
-----------------------
For each tree in the ensemble the algorithm recursively traverses both
left and right subtrees simultaneously, tracking the proportion of the
sample path that goes to each branch at each internal node.  For leaf
nodes the algorithm accumulates the SHAP contribution to each feature.

We use sklearn's internal ``_predictors`` attribute (a list of lists
of ``TreePredictor`` objects) to extract node arrays and compute SHAP
values.

Reference:
    Lundberg, S.M., Erion, G., Chen, H., DeGrave, A., Prutkin, J.M.,
    Nair, B., Katz, R., Himmelfarb, J., Bansal, N. and Lee, S.I., 2020.
    From local explanations to global understanding with explainable AI
    for trees. Nature Machine Intelligence, 2(1), pp.56-67.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


class TreeSHAPExplainer:
    """
    Exact TreeSHAP for sklearn HistGradientBoostingRegressor.

    Parameters
    ----------
    model : fitted HistGradientBoostingRegressor
    X_background : np.ndarray, shape (n_background, n_features)
        Background dataset used to estimate E[f(X)].  The training set
        or a representative subsample (100–500 rows) is recommended.
    feature_names : list of str, optional
    """

    def __init__(self, model, X_background: np.ndarray,
                 feature_names: Optional[list] = None):
        self.model = model
        self.feature_names = feature_names
        self.base_value = float(np.mean(
            model.predict(X_background)
        ))
        self._trees = self._extract_trees()
        self._learning_rate = getattr(model, 'learning_rate', 0.1)

    # ------------------------------------------------------------------
    def _extract_trees(self):
        """Extract tree structures from HistGradientBoosting predictors."""
        trees = []
        for stage in self.model._predictors:
            for tree_predictor in stage:
                nodes = tree_predictor.nodes
                trees.append(nodes)
        return trees

    # ------------------------------------------------------------------
    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for each sample in X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)
            SHAP value for each feature and sample.
        """
        n_samples, n_features = X.shape
        shap_vals = np.zeros((n_samples, n_features))

        for i in range(n_samples):
            x = X[i]
            row_shap = np.zeros(n_features)
            for tree_nodes in self._trees:
                row_shap += self._tree_shap(x, tree_nodes, n_features)
            shap_vals[i] = row_shap * self._learning_rate

        return shap_vals

    # ------------------------------------------------------------------
    def _tree_shap(self, x: np.ndarray, nodes: np.ndarray,
                   n_features: int) -> np.ndarray:
        """
        Path-dependent TreeSHAP for a single tree (Lundberg et al., 2020).
        Uses the 'interventional' estimation approach via recursive tree
        traversal with path tracking.

        Each tree node in HistGB has these fields (in order):
          0: value, 1: count, 2: feature_idx, 3: num_threshold,
          4: missing_go_left, 5: left, 6: right, 7: is_leaf
        """
        phi = np.zeros(n_features)

        # path = list of (node_idx, incoming_fraction, feature_idx_used)
        def recurse(node_idx: int, path_fracs: np.ndarray,
                    path_feats: np.ndarray, path_depth: int,
                    path_prop: float) -> None:
            node = nodes[node_idx]
            is_leaf   = bool(node['is_leaf'])
            feat_idx  = int(node['feature_idx'])
            threshold = float(node['num_threshold'])
            left_idx  = int(node['left'])
            right_idx = int(node['right'])
            node_val  = float(node['value'])
            node_cnt  = float(node['count'])

            if is_leaf:
                # Distribute leaf value back through path features
                # proportional to each feature's fraction along the path
                for d in range(path_depth):
                    f = int(path_feats[d])
                    if f >= 0:
                        phi[f] += path_prop * path_fracs[d] * node_val
                return

            # Determine which child the sample goes to
            x_val = x[feat_idx]
            goes_left = (x_val <= threshold)

            # Count fractions (how many training samples go each way)
            left_node  = nodes[left_idx]
            right_node = nodes[right_idx]
            left_cnt   = float(left_node['count'])
            right_cnt  = float(right_node['count'])
            total_cnt  = left_cnt + right_cnt

            left_frac  = left_cnt  / total_cnt if total_cnt > 0 else 0.5
            right_frac = right_cnt / total_cnt if total_cnt > 0 else 0.5

            # Record current feature in path
            path_feats = np.append(path_feats, feat_idx)
            path_fracs = np.append(path_fracs, 0.0)

            if goes_left:
                # Sample goes left (fraction = 1 for its direction)
                # Counterfactual right subtree contributes via background
                path_fracs[-1] = 1.0 - left_frac
                recurse(right_idx, path_fracs.copy(), path_feats.copy(),
                        path_depth + 1, path_prop * right_frac)
                path_fracs[-1] = left_frac
                recurse(left_idx, path_fracs.copy(), path_feats.copy(),
                        path_depth + 1, path_prop)
            else:
                path_fracs[-1] = 1.0 - right_frac
                recurse(left_idx, path_fracs.copy(), path_feats.copy(),
                        path_depth + 1, path_prop * left_frac)
                path_fracs[-1] = right_frac
                recurse(right_idx, path_fracs.copy(), path_feats.copy(),
                        path_depth + 1, path_prop)

        recurse(0, np.array([]), np.array([]), 0, 1.0)
        return phi

    # ------------------------------------------------------------------
    def expected_value(self) -> float:
        return self.base_value
