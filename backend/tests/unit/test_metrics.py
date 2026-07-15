"""
tests/unit/test_metrics.py
===========================
Unit tests for src.utils.metrics — all pure functions.
"""
import numpy as np
import pytest

from src.utils.metrics import evaluate_all, mae, mape, rmse, r2_score, smape


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_predictions():
    y = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    return y, y.copy()


@pytest.fixture
def typical_predictions():
    y_true = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    y_pred = np.array([110.0, 140.0, 210.0, 240.0, 310.0])
    return y_true, y_pred


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------

class TestRMSE:
    def test_perfect(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        assert rmse(y_true, y_pred) == pytest.approx(0.0)

    def test_known_value(self, typical_predictions):
        y_true, y_pred = typical_predictions
        errors = y_true - y_pred
        expected = float(np.sqrt(np.mean(errors ** 2)))
        assert rmse(y_true, y_pred) == pytest.approx(expected, rel=1e-6)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            rmse(np.array([1.0, 2.0]), np.array([1.0]))

    def test_empty_array(self):
        with pytest.raises(ValueError, match="empty"):
            rmse(np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------

class TestMAE:
    def test_perfect(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        assert mae(y_true, y_pred) == pytest.approx(0.0)

    def test_known_value(self, typical_predictions):
        y_true, y_pred = typical_predictions
        expected = float(np.mean(np.abs(y_true - y_pred)))
        assert mae(y_true, y_pred) == pytest.approx(expected)

    def test_nonnegative(self):
        y_true = np.array([50.0, 100.0])
        y_pred = np.array([60.0, 90.0])
        assert mae(y_true, y_pred) >= 0.0


# ---------------------------------------------------------------------------
# MAPE
# ---------------------------------------------------------------------------

class TestMAPE:
    def test_perfect(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        assert mape(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)

    def test_near_zero_denominator(self):
        # Should not raise ZeroDivisionError
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([10.0, 110.0])
        result = mape(y_true, y_pred)
        assert np.isfinite(result)

    def test_returns_percentage(self, typical_predictions):
        y_true, y_pred = typical_predictions
        result = mape(y_true, y_pred)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# sMAPE
# ---------------------------------------------------------------------------

class TestSMAPE:
    def test_perfect(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        assert smape(y_true, y_pred) == pytest.approx(0.0, abs=1e-4)

    def test_bounded(self):
        y_true = np.array([1.0, 100.0, 50.0])
        y_pred = np.array([1000.0, 0.001, 25.0])
        result = smape(y_true, y_pred)
        assert 0.0 <= result <= 200.0

    def test_symmetric_property(self):
        # sMAPE is symmetric: smape(a,b) ≈ smape(b,a)
        a = np.array([100.0, 200.0, 150.0])
        b = np.array([110.0, 180.0, 170.0])
        assert smape(a, b) == pytest.approx(smape(b, a), rel=1e-6)


# ---------------------------------------------------------------------------
# R²
# ---------------------------------------------------------------------------

class TestR2:
    def test_perfect(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        assert r2_score(y_true, y_pred) == pytest.approx(1.0)

    def test_mean_baseline(self):
        y_true = np.array([100.0, 150.0, 200.0])
        y_pred = np.full(3, y_true.mean())
        assert r2_score(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)

    def test_can_be_negative(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([300.0, 100.0, 200.0])
        assert r2_score(y_true, y_pred) < 0.0

    def test_constant_true(self):
        y_true = np.array([100.0, 100.0, 100.0])
        y_pred = np.array([100.0, 100.0, 100.0])
        assert r2_score(y_true, y_pred) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# evaluate_all
# ---------------------------------------------------------------------------

class TestEvaluateAll:
    def test_returns_all_keys(self, typical_predictions):
        y_true, y_pred = typical_predictions
        result = evaluate_all(y_true, y_pred)
        assert set(result.keys()) == {"rmse", "mae", "mape", "smape", "r2"}

    def test_prefix(self, typical_predictions):
        y_true, y_pred = typical_predictions
        result = evaluate_all(y_true, y_pred, prefix="test_")
        assert all(k.startswith("test_") for k in result.keys())

    def test_values_are_floats(self, typical_predictions):
        y_true, y_pred = typical_predictions
        result = evaluate_all(y_true, y_pred)
        assert all(isinstance(v, float) for v in result.values())
