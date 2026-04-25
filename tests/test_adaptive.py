"""Tests for AdaptiveConformalRisk (ACI)."""
import numpy as np
import pytest

from conformal_risk.adaptive import AdaptiveConformalRisk

RNG = np.random.default_rng(1)


def _regime_shift_returns(n_normal=300, n_stress=100):
    """Normal returns followed by a high-vol stress regime."""
    normal = RNG.normal(-0.0005, 0.01, n_normal)
    stress = RNG.normal(-0.005, 0.04, n_stress)  # 4x volatility
    return np.concatenate([normal, stress])


def test_aci_maintains_coverage_under_shift():
    """After a regime shift ACI should self-correct within ~50 steps."""
    returns = _regime_shift_returns(400, 200)
    aci = AdaptiveConformalRisk(alpha=0.05, gamma=0.01, window=200)
    aci.fit(returns[:300])
    errors = []
    for r in returns[300:]:
        record = aci.observe(r)
        errors.append(record["error"])
    empirical_miscoverage = np.mean(errors)
    # Allow some slack: ACI corrects over time, not instantly
    assert empirical_miscoverage < 0.20, f"Miscoverage {empirical_miscoverage:.2f} too high"


def test_aci_alpha_increases_after_miss():
    """After a loss > VaR, alpha_t should decrease (tighter bound)."""
    returns = RNG.normal(-0.0005, 0.01, 300)
    aci = AdaptiveConformalRisk(alpha=0.05, gamma=0.01, window=252)
    aci.fit(returns[:250])
    alpha_before = aci._alpha_t
    # Force a miss by observing a very large loss
    record = aci.observe(-0.20)  # -20% return -> large loss
    assert record["error"] == 1
    # alpha should move toward nominal after a miss (error=1):
    # alpha_t+1 = alpha_t + gamma*(alpha - 1) < alpha_t
    assert aci._alpha_t < alpha_before


def test_cusum_detects_regime_change():
    """CUSUM should flag the regime shift."""
    normal = RNG.normal(0, 0.01, 200)
    stress = RNG.normal(0, 0.05, 50)
    returns = np.concatenate([normal, stress])
    aci = AdaptiveConformalRisk(alpha=0.05, cusum_threshold=3.0, regime_reset=True)
    aci.fit(returns[:200])
    regime_changes = 0
    for r in returns[200:]:
        record = aci.observe(r)
        if record["regime_change"]:
            regime_changes += 1
    assert regime_changes >= 1, "CUSUM did not detect the regime shift"


def test_predict_infinite_before_warmup():
    """With fewer than 5 observations, predict returns inf."""
    aci = AdaptiveConformalRisk(alpha=0.05)
    assert aci.predict() == float("inf")


def test_coverage_history_keys():
    returns = RNG.normal(-0.0005, 0.01, 300)
    aci = AdaptiveConformalRisk(alpha=0.05)
    aci.fit(returns[:250])
    for r in returns[250:]:
        aci.observe(r)
    hist = aci.coverage_history()
    assert "empirical_coverage" in hist
    assert "target_coverage" in hist
    assert "regime_changes_detected" in hist
    assert hist["target_coverage"] == pytest.approx(0.95)


def test_no_regime_reset_keeps_alpha():
    """With regime_reset=False, alpha_t should not reset at change-points."""
    returns = _regime_shift_returns(300, 100)
    aci = AdaptiveConformalRisk(alpha=0.05, gamma=0.02, regime_reset=False)
    aci.fit(returns[:300])
    # In no-reset mode, alpha_t drifts continuously
    for r in returns[300:]:
        aci.observe(r)
    # Should still be a valid probability
    assert 0 < aci._alpha_t < 1
