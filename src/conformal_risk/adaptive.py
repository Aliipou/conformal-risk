"""
Adaptive Conformal Inference (ACI) for non-stationary return series.

The split conformal guarantee assumes exchangeability, which financial returns
violate: volatility clusters, regimes shift, distributions change.  ACI
(Gibbs & Candes 2021) adapts the significance level alpha_t online to maintain
marginal coverage despite distribution shift.

Algorithm (ACI)
---------------
At time t:
  1. Observe whether loss_{t} > VaR_{t-1} (a coverage failure, err_t = 1).
  2. Update: alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
     where gamma is the learning rate.
  3. Recompute the conformal quantile using the updated alpha_{t+1}.

This is a gradient step on the pinball loss: the effective alpha increases
(tightens the bound) after a miss, decreases after a hit.

For regime-aware adaptation, we maintain separate alpha streams per detected
regime and switch between them at change-points.

References
----------
Gibbs, I., & Candes, E. (2021).
  Adaptive Conformal Inference Under Distribution Shift. NeurIPS 2021.
Zaffran, M., Feron, O., Goude, Y., Josse, J., & Dieuleveut, A. (2022).
  Adaptive Conformal Predictions for Time Series. ICML 2022.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from conformal_risk.var import _conformal_quantile


class AdaptiveConformalRisk:
    """
    Online adaptive conformal VaR that maintains coverage under distribution shift.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate (e.g. 0.05 = 95% coverage).
    gamma : float
        ACI learning rate.  Larger gamma = faster adaptation, more variance.
        Gibbs & Candes recommend gamma in [0.001, 0.05].
    window : int
        Rolling calibration window size.
    regime_reset : bool
        If True, reset alpha_t to the nominal alpha at detected regime changes.
    cusum_threshold : float
        CUSUM threshold for regime change detection (in units of sigma).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        gamma: float = 0.005,
        window: int = 252,
        regime_reset: bool = True,
        cusum_threshold: float = 4.0,
    ) -> None:
        self.alpha_nominal = alpha
        self.gamma = gamma
        self.window = window
        self.regime_reset = regime_reset
        self.cusum_threshold = cusum_threshold

        self._alpha_t = alpha
        self._scores: deque[float] = deque(maxlen=window)
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._cusum_mu = 0.0
        self._cusum_sigma = 1.0
        self._n_updates = 0
        self._history: list[dict] = []

    # ── Streaming API ─────────────────────────────────────────────────────────

    def observe(self, ret: float) -> dict:
        """
        Process one new return observation.

        Returns a dict with:
          var         : current VaR bound (before this observation)
          loss        : realised loss (positive = loss)
          error       : 1 if loss > var else 0
          alpha_t     : effective alpha used this step
          regime_change : True if CUSUM detected a structural break
        """
        loss = -ret
        var = self.predict()
        error = int(loss > var)
        regime_change = False

        # Detect regime change via CUSUM on loss series
        if self._n_updates >= 20:
            regime_change = self._cusum_update(loss)
            if regime_change and self.regime_reset:
                self._alpha_t = self.alpha_nominal
                self._cusum_pos = 0.0
                self._cusum_neg = 0.0

        # ACI alpha update: gradient step on pinball loss
        alpha_used = self._alpha_t
        self._alpha_t = np.clip(
            self._alpha_t + self.gamma * (self.alpha_nominal - error),
            1e-4,
            0.5,
        )

        # Update calibration window
        self._scores.append(loss)
        self._n_updates += 1

        record = {
            "var": var,
            "loss": loss,
            "error": error,
            "alpha_t": alpha_used,
            "alpha_next": self._alpha_t,
            "regime_change": regime_change,
            "n": self._n_updates,
        }
        self._history.append(record)
        return record

    def predict(self) -> float:
        """Current conformal VaR estimate using adaptive alpha_t."""
        if len(self._scores) < 5:
            return float("inf")
        scores = np.array(self._scores)
        return float(_conformal_quantile(scores, self._alpha_t))

    def fit(self, historical_returns: np.ndarray) -> "AdaptiveConformalRisk":
        """
        Warm-start on historical data before going live.

        Runs .observe() on each historical return to calibrate the CUSUM
        parameters and fill the rolling window.
        """
        arr = np.asarray(historical_returns, dtype=float)
        # Estimate CUSUM baseline from first half
        half = max(arr.shape[0] // 2, 1)
        losses = -arr[:half]
        self._cusum_mu = float(losses.mean())
        self._cusum_sigma = float(losses.std(ddof=1)) or 1.0
        for ret in arr[half:]:
            self.observe(ret)
        return self

    def coverage_history(self) -> dict[str, float]:
        """Rolling statistics over observed history."""
        if not self._history:
            return {}
        errors = [h["error"] for h in self._history]
        regime_changes = sum(h["regime_change"] for h in self._history)
        return {
            "empirical_coverage": 1.0 - float(np.mean(errors)),
            "target_coverage": 1.0 - self.alpha_nominal,
            "coverage_gap": (1.0 - self.alpha_nominal) - (1.0 - float(np.mean(errors))),
            "regime_changes_detected": int(regime_changes),
            "n_observations": len(self._history),
            "current_alpha_t": self._alpha_t,
        }

    # ── CUSUM change-point detection ──────────────────────────────────────────

    def _cusum_update(self, loss: float) -> bool:
        """CUSUM test for shift in loss distribution. Returns True on detection."""
        z = (loss - self._cusum_mu) / (self._cusum_sigma or 1.0)
        self._cusum_pos = max(0.0, self._cusum_pos + z - 0.5)
        self._cusum_neg = max(0.0, self._cusum_neg - z - 0.5)
        if self._cusum_pos > self.cusum_threshold or self._cusum_neg > self.cusum_threshold:
            # Update baseline to current empirical moments
            recent = list(self._scores)
            if len(recent) >= 10:
                self._cusum_mu = float(np.mean(recent[-20:]))
                self._cusum_sigma = float(np.std(recent[-20:], ddof=1)) or self._cusum_sigma
            return True
        return False
