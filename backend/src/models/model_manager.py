import sys
import pickle
import numpy as np
import sklearn
import threading
from sklearn._loss import loss as loss_module
# Version aliasing workaround
sys.modules["_loss"] = loss_module

from src.config.app_config import BASE_DIR, MODEL_HASHES

class SecurityError(Exception):
    pass

class ModelManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.ml_models = {}
        self.dl_models = {}
        self.scalers = {}
        self.lock = threading.Lock()
        
        dl_seq_path = BASE_DIR / "data/processed/dl_sequences.npz"
        if dl_seq_path.exists():
            data = np.load(dl_seq_path)
            self.scalers["dl_mean"] = data["scaler_mean"]
            self.scalers["dl_std"] = data["scaler_std"]
            
        self._initialized = True
        
    def verify_hash(self, path, expected_hash):
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != expected_hash:
            raise SecurityError(f"Model integrity validation failed for {path.name}. SHA-256 mismatch!")

    def _load_ml_model(self, key, filename):
        import joblib
        path = BASE_DIR / "outputs/models" / filename
        if path.exists():
            expected = MODEL_HASHES.get(filename)
            self.verify_hash(path, expected)
            with open(path, "rb") as f:
                self.ml_models[key] = joblib.load(f)
            print(f"[ModelManager] Lazily loaded ML model: {key}")
        else:
            raise FileNotFoundError(f"Model file not found: {path}")

    def _load_dl_model(self, key, filename):
        import joblib
        path = BASE_DIR / "outputs/models/deep_learning" / filename
        if path.exists():
            expected = MODEL_HASHES.get(f"deep_learning/{filename}")
            self.verify_hash(path, expected)
            with open(path, "rb") as f:
                self.dl_models[key] = joblib.load(f)
            print(f"[ModelManager] Lazily loaded DL model: {key}")
        else:
            raise FileNotFoundError(f"Model file not found: {path}")

    def get_model(self, name):
        with self.lock:
            if "HistGradientBoosting" in name:
                key, filename, model_type = "HistGradientBoosting", "hist_gradient_boosting.joblib", "ML"
                if key not in self.ml_models:
                    self._load_ml_model(key, filename)
                return self.ml_models[key], model_type
            elif "Random" in name or "Decision" in name:
                key, filename, model_type = "DecisionTree", "decision_tree.joblib", "ML"
                if key not in self.ml_models:
                    self._load_ml_model(key, filename)
                return self.ml_models[key], model_type
            elif "XGBoost" in name or "Gradient" in name:
                key, filename, model_type = "GradientBoosting", "gradient_boosting.joblib", "ML"
                if key not in self.ml_models:
                    self._load_ml_model(key, filename)
                return self.ml_models[key], model_type
            elif "Linear" in name:
                key, filename, model_type = "LinearRegression", "linear_regression.joblib", "ML"
                if key not in self.ml_models:
                    self._load_ml_model(key, filename)
                return self.ml_models[key], model_type
            elif "Ridge" in name:
                key, filename, model_type = "RidgeRegression", "ridge_regression.joblib", "ML"
                if key not in self.ml_models:
                    self._load_ml_model(key, filename)
                return self.ml_models[key], model_type
            elif "LSTM" in name and "CNN" not in name:
                key, filename, model_type = "LSTM", "lstm_model.pkl", "DL"
                if key not in self.dl_models:
                    self._load_dl_model(key, filename)
                return self.dl_models[key], model_type
            elif "GRU" in name:
                key, filename, model_type = "GRU", "gru_model.pkl", "DL"
                if key not in self.dl_models:
                    self._load_dl_model(key, filename)
                return self.dl_models[key], model_type
            elif "CNN" in name:
                key, filename, model_type = "CNN-LSTM", "cnn_lstm_model.pkl", "DL"
                if key not in self.dl_models:
                    self._load_dl_model(key, filename)
                return self.dl_models[key], model_type
                
            key, filename, model_type = "HistGradientBoosting", "hist_gradient_boosting.joblib", "ML"
            if key not in self.ml_models:
                self._load_ml_model(key, filename)
            return self.ml_models[key], model_type
