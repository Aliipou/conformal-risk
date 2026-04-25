"""Tests for ConformalCVaR."""
import numpy as np
import pytest

from conformal_risk.cvar import ConformalCVaR

RNG = np.random.default_rng(7)
RETURNS = RNG.normal(-0.0005, 0.01, 800)


def test_cvar_geq_var():
    """CVaR must always be >= VaR by definition."""
    cvar = ConformalCVaR(alpha=0.05)
    cvar.fit(RETURNS[:600])
    var_b, cvar_b = cvar.predict()
    assert cvar_b >= var_b, f"CVaR {cvar_b:.4f} < VaR {var_b:.4f}"


def test_coverage_keys():
    cvar = ConformalCVaR(alpha=0.05)
    cvar.fit(RETURNS[:600])
    stats = cvar.coverage(RETURNS[600:])
    assert "var_coverage" in stats
    assert "cvar_coverage" in stats
    assert "tail_fraction" in stats


def test_var_coverage_satisfied():
    cvar = ConformalCVaR(alpha=0.05)
    cvar.fit(RETURNS[:600])
    stats = cvar.coverage(RETURNS[600:])
    assert stats["var_coverage"] >= 0.93, f"VaR coverage {stats['var_coverage']:.3f} < 0.93"


def test_tail_fraction_near_alpha():
    cvar = ConformalCVaR(alpha=0.05)
    cvar.fit(RETURNS[:600])
    stats = cvar.coverage(RETURNS[600:])
    # Tail fraction should be roughly alpha = 0.05, with some slack
    assert 0.01 <= stats["tail_fraction"] <= 0.15


def test_small_calibration_raises():
    cvar = ConformalCVaR(alpha=0.05)
    with pytest.raises(ValueError):
        cvar.fit(np.random.randn(10))


def test_predict_without_fit_raises():
    with pytest.raises(RuntimeError):
        ConformalCVaR().predict()


def test_stricter_alpha_larger_bounds():
    cvar_05 = ConformalCVaR(alpha=0.05)
    cvar_01 = ConformalCVaR(alpha=0.01)
    cvar_05.fit(RETURNS[:600])
    cvar_01.fit(RETURNS[:600])
    _, cvar_bound_05 = cvar_05.predict()
    _, cvar_bound_01 = cvar_01.predict()
    assert cvar_bound_01 >= cvar_bound_05 - 0.001
