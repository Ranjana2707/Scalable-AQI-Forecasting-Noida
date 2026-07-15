"""
src.deep_learning — Phase 6: Sequence models for multi-step AQI forecasting.

IMPLEMENTATION NOTE
--------------------
TensorFlow and PyTorch are not installable in this execution environment
(no network access to their package indices — only a small pre-mirrored
scientific stack: numpy/scipy/pandas/scikit-learn is reachable). To deliver
the requested LSTM, GRU, and CNN-LSTM architectures faithfully rather than
silently substituting a different model family, this module implements all
three architectures from scratch using NumPy:

- Vectorised forward and backward passes (BPTT) for LSTM and GRU cells
- 1D convolution + LSTM head for CNN-LSTM
- Adam optimiser, gradient clipping, Xavier/Glorot initialisation
- Early stopping and model checkpointing matching Keras semantics
- Identical train/val/test split and feature set as Phase 5 ML models

This keeps every modelling decision (architecture, hyperparameters,
training procedure) auditable line-by-line, which is otherwise hidden
inside a framework. For a production deployment with GPU access,
``src/deep_learning/keras_reference.py`` documents the equivalent
``tf.keras`` model definitions that should be used instead.
"""
