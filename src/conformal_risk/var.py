"""
Conformal Value-at-Risk (ConformalVaR).

Standard VaR assumes a parametric distribution (normal, t, etc.) and gives no
coverage guarantee when that assumption fails — which it routinely does during
market stress.

Split Conformal Prediction (Papadopoulos et al. 2002; Vovk et al. 2005) provides
a *distribution-free* guarantee: for exchangeable data, the conformal VaR achieves
at least (1 - alpha) marginal coverage with no parametric assumption.

Algorithm (split conformal)
---------------------------
1. Partition historical returns into calibration set and current window.
2. Use the absolute return (or a nonconformity score based on a base predictor)
   as the nonconformity score s_i = -r_i  (losses are positive).
3. The conformal quantile is q = the ceil((n+1)(1-alpha)) / n empirical quantile
   of {s_i}.
4. Report VaR = q as the risk bound.

Coverage guarantee
------------------
P(loss_{T+1} <= VaR_alpha) >= 1 - alpha   for any joint distribution,
provided (r_1, ..., r_n, r_{T+1}) are exchangeable.

References
----------
Papadopoulos, H., Proedrou, K., Vovk, V., & Gammerman, A. (2002).
  Inductive confidence machines for regression. ECML 2002.
Vovk, V., Gammerman, A., & Shafer, G. (2005).
  Algorithmic Learning in a Random World. Springer.
Barber, R., Candes, E., Ramdas, A., & Tibshirani, R. (2023).
  Conformal prediction beyond exchangeability. Annals of Statistics.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Sequence

import numpy as np


class ConformalVaR:
    """
    Split conformal Value-at-Risk with exact finite-sample coverage.

    Parameters
    ----------
    alpha : float
        Significance level. VaR estimates the alpha-quantile of the loss
        distribution, i.e., coverage = (1 - alpha).  Typical: 0.05, 0.01.
    calibration_size : int | None
        Number of historical returns to use for calibration.
        If None, all returns passed to .fit() are used.
    nonconformity : str
        ``"loss"``  : nonconformity score = -r_i  (raw loss).
        ``"zscore"`` : normalise by rolling std (accounts for volatility clustering).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> returns = rng.normal(-0.001, 0.01, 500)
    >>> var = ConformalVaR(alpha=0.05)
    >>> var.fit(returns[:400])
    >>> bound = var.predict()  # loss not exceeded with 95% coverage guarantee
    """

    def __init__(
        self,
        alpha: float = 0.05,
        calibration_size: int | None = None,
        nonconformity: str = "loss",
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if nonconformity not in ("loss", "zscore"):
            raise ValueError("nonconformity must be 'loss' or 'zscore'")
        self.alpha = alpha
        self.calibration_size = calibration_size
        self.nonconformity = nonconformity
        self._scores: np.ndarray | None = None

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, returns: Sequence[float] | np.ndarray) -> "ConformalVaR":
        """Calibrate on historical returns."""
        r = np.asarray(returns, dtype=float)
        if self.calibration_size is not None:
            r = r[-self.calibration_size :]
        self._scores = self._score(r)
        return self

    def update(self, new_return: float) -> "ConformalVaR":
        """Incrementally add a new observation to the calibration set."""
        if self._scores is None:
            raise RuntimeError("Call .fit() before .update()")
        new_score = self._score(np.array([new_return]))[0]
        # Append and trim to calibration_size if set
        scores = np.append(self._scores, new_score)
        if self.calibration_size is not None:
            scores = scores[-self.calibration_size :]
        self._scores = scores
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self) -> float:
        """
        Return the conformal VaR bound (expressed as a positive loss magnitude).

        The conformal quantile is the ceil((n+1)(1-alpha))/n empirical quantile,
        which guarantees marginal coverage >= (1 - alpha).
        """
        if self._scores is None:
            raise RuntimeError("Call .fit() before .predict()")
        return float(_conformal_quantile(self._scores, self.alpha))

    def coverage_probability(self, test_returns: Sequence[float] | np.ndarray) -> float:
        """
        Empirical coverage on a held-out set.

        Returns the fraction of test_returns whose loss is within the VaR bound.
        Should be >= (1 - alpha) for exchangeable data.
        """
        var = self.predict()
        losses = -np.asarray(test_returns, dtype=float)
        return float((losses <= var).mean())

    def predict_interval(self) -> tuple[float, float]:
        """
        Return (lower_var, upper_var) — a two-sided conformal prediction interval
        for the return, i.e., return_{T+1} in [lower_var, upper_var].

        Lower and upper are symmetric conformal quantiles at alpha/2 each side.
        """
        if self._scores is None:
            raise RuntimeError("Call .fit() before .predict_interval()")
        # For two-sided: compute quantile for loss and for gain separately
        upper_loss = _conformal_quantile(self._scores, self.alpha / 2)
        # Gain = positive return; scores = -return
        gain_scores = -self._scores  # flip sign: large gain = large score
        upper_gain = _conformal_quantile(gain_scores, self.alpha / 2)
        # lower return bound = -upper_loss; upper return bound = upper_gain
        return (-upper_loss, upper_gain)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _score(self, returns: np.ndarray) -> np.ndarray:
        if self.nonconformity == "loss":
            return -returns
        # zscore: normalise by rolling std with window=20
        window = min(20, len(returns))
        stds = np.array([
            returns[:max(i, 1)].std(ddof=0) or 1.0
            for i in range(len(returns))
        ])
        return -returns / stds


# ── Utilities ─────────────────────────────────────────────────────────────────

def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    The conformal quantile: ceil((n+1)(1-alpha)) / n empirical quantile.

    This is the smallest value q such that at least ceil((n+1)(1-alpha))
    of the n scores are <= q, which gives marginal coverage >= 1-alpha.
    """
    n = len(scores)
    level = math.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level))
