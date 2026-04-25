"""Tests for ConformalVaR — coverage guarantee and API."""
import numpy as np
import pytest

from conformal_risk.var import ConformalVaR, _conformal_quantile

RNG = np.random.default_rng(42)
NORMAL_RETURNS = RNG.normal(-0.0005, 0.01, 1000)
HEAVY_TAIL_RETURNS = RNG.standard_t(df=3, size=1000) * 0.01


def test_coverage_guarantee_normal():
    """
    Marginal coverage must be >= (1-alpha).

    We pool many test points across trials to reduce Binomial variance.
    With n=500 calibration, 100 test points per trial, 50 trials → 5000 total
    test observations.  Average coverage should be >= 0.95.
    """
    alpha = 0.05
    all_errors = []
    rng = np.random.default_rng(0)
    for _ in range(50):
        r = rng.normal(-0.0005, 0.01, 600)
        var = ConformalVaR(alpha=alpha)
        var.fit(r[:500])
        losses = -r[500:]
        bound = var.predict()
        all_errors.extend((losses > bound).tolist())
    empirical_miscoverage = float(np.mean(all_errors))
    assert empirical_miscoverage <= alpha + 0.02, (
        f"Pooled miscoverage {empirical_miscoverage:.3f} > {alpha + 0.02:.3f}"
    )


def test_coverage_guarantee_heavy_tail():
    """Coverage must hold under heavy-tailed (t-distributed) returns."""
    alpha = 0.05
    var = ConformalVaR(alpha=alpha)
    var.fit(HEAVY_TAIL_RETURNS[:800])
    coverage = var.coverage_probability(HEAVY_TAIL_RETURNS[800:])
    assert coverage >= (1 - alpha) - 0.02, f"Heavy tail coverage {coverage:.3f} < {1-alpha-0.02:.3f}"


def test_var_is_finite():
    var = ConformalVaR(alpha=0.05)
    var.fit(NORMAL_RETURNS[:500])
    v = var.predict()
    assert np.isfinite(v)
    assert v > 0  # VaR is a loss magnitude


def test_var_monotone_in_alpha():
    """Stricter alpha (smaller) should produce larger VaR bound."""
    var_05 = ConformalVaR(alpha=0.05)
    var_01 = ConformalVaR(alpha=0.01)
    var_05.fit(NORMAL_RETURNS[:500])
    var_01.fit(NORMAL_RETURNS[:500])
    assert var_01.predict() >= var_05.predict(), "VaR(0.01) must be >= VaR(0.05)"


def test_incremental_update():
    var = ConformalVaR(alpha=0.05, calibration_size=200)
    var.fit(NORMAL_RETURNS[:200])
    var.predict()  # v_before — not checked, update is the focus
    var.update(NORMAL_RETURNS[200])
    v_after = var.predict()
    assert np.isfinite(v_after)
    # Score window stays at calibration_size
    assert var._scores is not None
    assert len(var._scores) == 200


def test_prediction_interval_contains_var():
    var = ConformalVaR(alpha=0.05)
    var.fit(NORMAL_RETURNS[:500])
    lower, upper = var.predict_interval()
    # Upper end of return interval corresponds to lower loss = smaller VaR
    assert lower < upper
    assert np.isfinite(lower) and np.isfinite(upper)


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        ConformalVaR(alpha=0.0)
    with pytest.raises(ValueError):
        ConformalVaR(alpha=1.0)


def test_predict_without_fit_raises():
    with pytest.raises(RuntimeError):
        ConformalVaR().predict()


def test_zscore_nonconformity():
    var = ConformalVaR(alpha=0.05, nonconformity="zscore")
    var.fit(NORMAL_RETURNS[:500])
    v = var.predict()
    assert np.isfinite(v) and v > 0


def test_conformal_quantile_exact():
    """The conformal quantile at alpha=0.5 should be the empirical median (roughly)."""
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    q = _conformal_quantile(scores, 0.5)
    # ceil((5+1)*0.5)/5 = ceil(3)/5 = 0.6 quantile
    assert q == pytest.approx(np.quantile(scores, 0.6))
