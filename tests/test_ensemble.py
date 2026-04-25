"""Tests for EnsembleConformalRisk."""
import numpy as np
import pytest
from conformal_risk.ensemble import EnsembleConformalRisk


RNG = np.random.default_rng(3)
RETURNS = RNG.normal(-0.0005, 0.01, 700)


def test_predict_finite():
    ens = EnsembleConformalRisk(alpha=0.05, n_folds=3)
    ens.fit(RETURNS[:600])
    v = ens.predict()
    assert np.isfinite(v) and v > 0


def test_base_predictions_keys():
    ens = EnsembleConformalRisk(alpha=0.05, base_models=["normal", "historical"])
    ens.fit(RETURNS[:600])
    preds = ens.base_predictions()
    assert set(preds.keys()) == {"normal", "historical"}
    assert all(np.isfinite(v) for v in preds.values())


def test_all_base_models():
    ens = EnsembleConformalRisk(alpha=0.05, base_models=["normal", "historical", "ewma"])
    ens.fit(RETURNS[:600])
    v = ens.predict()
    assert np.isfinite(v)


def test_coverage_ensemble():
    """Ensemble pooled coverage must be close to (1-alpha)."""
    alpha = 0.05
    all_errors = []
    rng = np.random.default_rng(99)
    for _ in range(30):
        r = rng.normal(-0.0005, 0.01, 700)
        ens = EnsembleConformalRisk(alpha=alpha, n_folds=3)
        ens.fit(r[:600])
        bound = ens.predict()
        test_losses = -r[600:]
        all_errors.extend((test_losses > bound).tolist())
    empirical_miscoverage = float(np.mean(all_errors))
    assert empirical_miscoverage <= alpha + 0.03, (
        f"Pooled miscoverage {empirical_miscoverage:.3f} > {alpha + 0.03:.3f}"
    )


def test_insufficient_data_raises():
    ens = EnsembleConformalRisk(n_folds=5)
    with pytest.raises(ValueError):
        ens.fit(np.random.randn(10))


def test_unknown_base_model_raises():
    ens = EnsembleConformalRisk(base_models=["nonexistent"])
    with pytest.raises(ValueError):
        ens.fit(RETURNS[:600])
