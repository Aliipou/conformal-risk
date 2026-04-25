# conformal-risk

**Distribution-free financial risk metrics with finite-sample coverage guarantees.**

[![CI](https://github.com/Aliipou/conformal-risk/actions/workflows/ci.yml/badge.svg)](https://github.com/Aliipou/conformal-risk/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## The Problem with Standard VaR

Value-at-Risk is defined as the α-quantile of the loss distribution: the loss not exceeded with probability (1 − α). Every practitioner uses it. Almost no implementation of it has a *guaranteed* coverage probability.

**Parametric VaR** (normal, t, GARCH) assumes a distribution. When that assumption fails — during market crises, regime shifts, fat-tail events — the coverage guarantee disappears. In 2008, 99% VaR models were violated on 5–10% of days at major banks (Berkowitz & O'Brien 2002 documented this before the crisis; the crisis confirmed it).

**Historical simulation** is non-parametric but its coverage is asymptotically correct only as the calibration window → ∞, with no finite-sample statement.

**Conformal prediction** (Vovk, Gammerman & Shafer 2005) provides an exact finite-sample coverage guarantee under a single assumption: *exchangeability* of the return series. No distributional form is assumed. The guarantee is:

```
P(loss_{T+1} ≤ ConformalVaR_α) ≥ 1 − α
```

for *any* joint distribution, with any finite calibration set of size n.

For non-exchangeable (non-stationary) series, **Adaptive Conformal Inference** (Gibbs & Candes 2021) recovers long-run coverage guarantees via online alpha adjustment.

---

## What This Library Provides

| Estimator | Guarantee | Use case |
|-----------|-----------|----------|
| `ConformalVaR` | Exact finite-sample marginal coverage | Stationary return series; calibration window ≥ 50 |
| `ConformalCVaR` | Finite-sample marginal coverage on VaR + CVaR | Tail risk, expected shortfall |
| `AdaptiveConformalRisk` | Long-run coverage under distribution shift | Live systems, regime-changing markets |
| `EnsembleConformalRisk` | Coverage + reduced variance via model averaging | Production risk systems |

---

## Installation

```bash
pip install conformal-risk
```

Or from source:
```bash
git clone https://github.com/Aliipou/conformal-risk
cd conformal-risk
pip install -e ".[dev]"
```

---

## Quick Start

```python
import numpy as np
from conformal_risk import ConformalVaR, AdaptiveConformalRisk

# Historical returns (500 days)
returns = np.array([...])

# ── Split conformal VaR ──────────────────────────────────────────
var = ConformalVaR(alpha=0.05)    # 95% coverage
var.fit(returns[:400])
bound = var.predict()             # loss not exceeded >= 95% of the time
print(f"95% VaR: {bound:.4f}")

# Verify empirically
coverage = var.coverage_probability(returns[400:])
print(f"Empirical coverage: {coverage:.1%}")  # >= 95%

# Two-sided prediction interval for tomorrow's return
lo, hi = var.predict_interval()
print(f"95% return interval: [{lo:.4f}, {hi:.4f}]")

# Incremental update (streaming)
var.update(new_return=-0.012)
```

```python
# ── Adaptive conformal (ACI) for live systems ────────────────────
from conformal_risk import AdaptiveConformalRisk

aci = AdaptiveConformalRisk(
    alpha=0.05,
    gamma=0.005,        # ACI learning rate
    window=252,         # rolling calibration: 1 trading year
    regime_reset=True,  # reset alpha_t at detected change-points
)
aci.fit(historical_returns)

# Each trading day:
for ret in live_returns:
    record = aci.observe(ret)
    print(f"VaR: {record['var']:.4f}  |  regime change: {record['regime_change']}")

# Coverage statistics
hist = aci.coverage_history()
print(f"Empirical coverage: {hist['empirical_coverage']:.1%}")
print(f"Regime changes detected: {hist['regime_changes_detected']}")
```

---

## Coverage Guarantee — Mathematical Statement

**Theorem** (Vovk et al. 2005, Theorem 2.2). Let (X_1, ..., X_n, X_{n+1}) be exchangeable random variables with nonconformity scores s_i = h(X_i). Define the conformal quantile:

```
q̂ = Quantile({s_1, ..., s_n}, ⌈(n+1)(1−α)⌉/n)
```

Then:

```
P(s_{n+1} ≤ q̂) ≥ 1 − α
```

with no assumption on the distribution of X. The bound is also tight: P(s_{n+1} ≤ q̂) ≤ 1 − α + 1/(n+1).

**When does exchangeability fail?** Financial returns are not i.i.d. (volatility clusters, autocorrelation). The marginal coverage guarantee degrades proportionally to the deviation from exchangeability, quantifiable via the *weighted exchangeability* framework of Tibshirani et al. (2019). In practice, rolling windows of 252 days produce near-valid coverage in normal markets. ACI corrects for drift.

---

## Adaptive Conformal Inference (ACI)

For non-stationary series, the effective significance level α_t is updated online:

```
α_{t+1} = α_t + γ · (α_nominal − err_t)
```

where err_t = 1 if loss_t > VaR_t (coverage miss), else 0. This is a stochastic gradient step minimising the long-run pinball loss E[|loss − VaR|_α].

**Regime detection**: the library uses a CUSUM test on the nonconformity score sequence to detect structural breaks. At a detected break, α_t is reset to α_nominal, preventing the adaptive procedure from over-correcting based on stale distributional assumptions.

From Gibbs & Candes (2021, Theorem 3): ACI achieves

```
lim_{T→∞} (1/T) Σ_t err_t → α_nominal  a.s.
```

under mild conditions on the time-varying distribution, regardless of γ > 0.

---

## Ensemble Conformal Risk

`EnsembleConformalRisk` combines multiple base estimators (parametric normal, historical simulation, EWMA/RiskMetrics) via cross-conformal aggregation:

1. Split calibration into K folds.
2. For each fold: train base models on K−1 folds, compute residuals on held-out fold.
3. Pool residuals → single conformal quantile.
4. At prediction: aggregate base predictions + conformal shift.

This inherits the coverage guarantee of conformal prediction while reducing variance compared to any single base model.

---

## Comparison with Standard Methods

| Method | Coverage guarantee | Parametric assumption | Streaming |
|--------|-------------------|-----------------------|-----------|
| Parametric normal VaR | Asymptotic, if normal | Normal returns | No |
| Historical simulation | Asymptotic | None | No |
| GARCH VaR | Asymptotic, if GARCH | GARCH dynamics | Yes (with refit) |
| **ConformalVaR** | **Exact finite-sample** | **None** | **Yes (update)** |
| **AdaptiveConformalRisk** | **Long-run under shift** | **None** | **Yes (native)** |

---

## Empirical Validation

On S&P 500 daily returns (2000–2024, n=6,000 days), 95% VaR:

| Method | Empirical coverage | VaR violations > 3σ |
|--------|--------------------|---------------------|
| Normal parametric | 91.2% | 14 |
| Historical (252-day) | 93.8% | 6 |
| **ConformalVaR (252-day)** | **95.7%** | **3** |
| **ACI (γ=0.005)** | **95.1%** | **2** |

(Reproduced from `examples/sp500_validation.py`. Data via `yfinance`.)

---

## Research Context

This library implements and extends the following theoretical frameworks:

- **Split conformal prediction**: Papadopoulos et al. (2002), Vovk et al. (2005)
- **Adaptive conformal inference**: Gibbs & Candes, NeurIPS 2021
- **Conformal risk control**: Angelopoulos & Bates, ICLR 2023
- **Weighted conformal**: Tibshirani et al., JRSS-B 2019
- **Financial risk**: Berkowitz & O'Brien (2002), McNeil et al. *Quantitative Risk Management* (2005)

**Open research questions** motivating this library:

1. What is the optimal window size for ConformalVaR under volatility clustering (ARCH effects)?
2. Can CUSUM-integrated ACI achieve finite-time rather than asymptotic coverage bounds?
3. How does ensemble conformal risk compare to optimal transport-based robust risk measures under model misspecification?

---

## Citation

```bibtex
@software{conformal_risk_2025,
  author  = {Pourrahim, Ali},
  title   = {conformal-risk: Distribution-free financial risk metrics},
  year    = {2025},
  url     = {https://github.com/Aliipou/conformal-risk},
}
```

---

## License

MIT
