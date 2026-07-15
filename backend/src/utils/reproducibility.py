"""
src/utils/reproducibility.py
==============================
Global seed-setting utility for full experiment reproducibility.

Sets seeds across:
- Python ``random`` module
- NumPy
- TensorFlow / Keras
- XGBoost (via ``seed`` parameter in configs)

Call ``set_global_seed(seed)`` once at application startup, before
any data splits, model initialisations, or augmentations.
"""

from __future__ import annotations

import os
import random

from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """
    Set random seeds across all relevant libraries for reproducibility.

    Parameters
    ----------
    seed : int
        The integer seed value. Default is 42 (matches ``configs/default.yaml``).

    Notes
    -----
    TensorFlow and NumPy imports are deferred so this module does not force
    those dependencies when only utils are needed (e.g., during config loading).

    Examples
    --------
    >>> from src.utils.reproducibility import set_global_seed
    >>> set_global_seed(42)
    """
    # Python stdlib
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.debug(f"Python random seed set to {seed}")

    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
        logger.debug(f"NumPy random seed set to {seed}")
    except ImportError:
        logger.warning("NumPy not installed; skipping NumPy seed.")

    # TensorFlow / Keras
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        logger.debug(f"TensorFlow random seed set to {seed}")
    except ImportError:
        logger.debug("TensorFlow not installed; skipping TF seed.")

    logger.info(f"Global random seed set to {seed} across all libraries.")
