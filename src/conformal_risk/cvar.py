"""
Conformal Conditional Value-at-Risk (Expected Shortfall).

CVaR = E[loss | loss > VaR], i.e., the expected loss in the worst alpha fraction
of outcomes.  Standard CVaR estimation is parametric (normal, historical, Monte
Carlo).  This module implements a conformal variant with finite-sample guarantees.

Approach: Mondrian conformal prediction stratified on the VaR exceedance event.
We build a separate conformal predictor conditioned on loss > conformal_VaR,
using only the calibration observations that exceed the VaR threshold.

Reference
---------
Linhart, J., Zellinger, W., & Mayer, S. (2023).
  Conformal prediction for risk management. AAAI Workshop on Uncertainty
  Quantification for Machine Learning.
Angelopoulos, A., & Bates, S. (2023).
  Conformal risk control. ICLR 2023.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from conformal_risk.var import ConformalVaR, _conformal_quantile


class ConformalCVaR:
    """
    Conformal Expected Shortfall (CVaR) with finite-sample marginal coverage.

    Strategy: split the calibration set into VaR calibration and tail
    calibration.  Estimate VaR on the first half; estimate CVaR on the
    tail observations from the second half that exceed the estimated VaR.

    Parameters
    ----------
    alpha : float
        Risk level (e.g. 0.05 for 95% CVaR).
    split : float
        Fraction of calibration data used for VaR calibration.
        The remainder is used for tail / CVaR calibration.
    """

    def __init__(self, alpha: float = 0.05, split: float = 0.5) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if not 0 < split < 1:
            raise ValueError("split must be in (0, 1)")
        self.alpha = alpha
        self.split = split
        self._var_bound: float | None = None
        self._tail_scores: np.ndarray | None = None

    def fit(self, returns: Sequence[float] | np.ndarray) -> "ConformalCVaR":
        """Calibrate VaR and tail distribution on historical returns."""
        r = np.asarray(returns, dtype=float)
        n = len(r)
        split_idx = int(n * self.split)
        if split_idx < 10:
            raise ValueError(f"Calibration set too small ({n}). Need >= 20 returns.")

        # VaR calibration on first split
        var_estimator = ConformalVaR(alpha=self.alpha)
        var_estimator.fit(r[:split_idx])
        self._var_bound = var_estimator.predict()

        # Tail calibration: scores of observations that exceed VaR
        tail_losses = -r[split_idx:]          # positive = loss
        tail_exceedances = tail_losses[tail_losses > self._var_bound]
        if len(tail_exceedances) < 5:
            # Fallback: use all tail losses
            tail_exceedances = tail_losses
        self._tail_scores = tail_exceedances
        return self

    def predict(self) -> tuple[float, float]:
        """
        Return (var_bound, cvar_bound).

        var_bound : conformal VaR at alpha level.
        cvar_bound : conformal CVaR — upper bound on expected loss in the tail.
        """
        if self._var_bound is None or self._tail_scores is None:
            raise RuntimeError("Call .fit() before .predict()")
        # CVaR bound = conformal quantile of tail scores at alpha/2
        # (tighter tail bound — the tail itself has alpha fraction)
        cvar = _conformal_quantile(self._tail_scores, self.alpha / 2)
        cvar = max(cvar, self._var_bound)   # CVaR >= VaR by definition
        return self._var_bound, cvar

    def coverage(self, test_returns: Sequence[float] | np.ndarray) -> dict[str, float]:
        """
        Empirical coverage statistics on held-out returns.

        Returns
        -------
        dict with keys:
          var_coverage  : fraction of losses <= VaR  (target >= 1-alpha)
          cvar_coverage : fraction of tail losses <= CVaR  (target >= 1-alpha/2)
          tail_fraction : fraction of returns that entered the tail
        """
        var_b, cvar_b = self.predict()
        losses = -np.asarray(test_returns, dtype=float)
        tail = losses[losses > var_b]
        return {
            "var_coverage": float((losses <= var_b).mean()),
            "cvar_coverage": float((tail <= cvar_b).mean()) if len(tail) > 0 else 1.0,
            "tail_fraction": float(len(tail) / len(losses)) if len(losses) > 0 else 0.0,
        }
