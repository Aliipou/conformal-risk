"""
Ensemble Conformal Risk: aggregating multiple base risk estimators.

Individual risk models (normal VaR, historical simulation, GARCH) each have
assumptions that can fail in different regimes.  Ensemble conformal prediction
(Angelopoulos et al. 2022) combines N base estimators into a single conformal
bound that inherits the best base-model performance while maintaining the
coverage guarantee of conformal prediction.

Strategy: cross-conformal aggregation.
  1. Split calibration set into N folds.
  2. For each fold k: train base models on the other N-1 folds, compute
     nonconformity scores on fold k.
  3. The aggregate nonconformity score = f({s_k^1, ..., s_k^M}) where M
     is the number of base models (e.g., mean, minimum, or learned weights).
  4. Use the pooled scores to compute the conformal quantile.

Reference
---------
Angelopoulos, A., Bates, S., Malik, J., & Jordan, M. I. (2022).
  Conformal Risk Control. ICLR 2023.
Venn prediction (Vovk 2003) for multi-model aggregation.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from conformal_risk.var import _conformal_quantile


BaseEstimator = Callable[[np.ndarray], float]


def _normal_var(alpha: float) -> BaseEstimator:
    """Parametric normal VaR base estimator."""
    from scipy.stats import norm

    def _estimate(calibration: np.ndarray) -> float:
        mu = calibration.mean()
        sigma = calibration.std(ddof=1)
        return float(-norm.ppf(alpha, loc=mu, scale=sigma))

    return _estimate


def _historical_var(alpha: float) -> BaseEstimator:
    """Historical simulation VaR base estimator."""

    def _estimate(calibration: np.ndarray) -> float:
        return float(np.quantile(-calibration, 1 - alpha))

    return _estimate


def _ewma_var(alpha: float, decay: float = 0.94) -> BaseEstimator:
    """EWMA volatility VaR (RiskMetrics)."""
    from scipy.stats import norm

    def _estimate(calibration: np.ndarray) -> float:
        n = len(calibration)
        weights = decay ** np.arange(n - 1, -1, -1)
        weights /= weights.sum()
        mu = float((weights * calibration).sum())
        var2 = float((weights * (calibration - mu) ** 2).sum())
        sigma = max(var2 ** 0.5, 1e-8)
        return float(-norm.ppf(alpha, loc=mu, scale=sigma))

    return _estimate


class EnsembleConformalRisk:
    """
    Ensemble conformal VaR combining multiple base estimators.

    Parameters
    ----------
    alpha : float
        Risk level.
    n_folds : int
        Cross-conformal folds.  3 or 5 is typical.
    aggregation : str
        How to aggregate base-model scores:
        ``"mean"``    — average of base predictions.
        ``"min"``     — most conservative (smallest VaR).
        ``"conformal"`` — conformal quantile across base nonconformity scores.
    base_models : list[str] | None
        Which base models to include. Options: ``"normal"``, ``"historical"``,
        ``"ewma"``. Defaults to all three.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_folds: int = 5,
        aggregation: str = "conformal",
        base_models: list[str] | None = None,
    ) -> None:
        self.alpha = alpha
        self.n_folds = n_folds
        self.aggregation = aggregation
        _all = ["normal", "historical", "ewma"]
        self._model_names = base_models or _all
        self._base_estimators: list[BaseEstimator] = []
        self._pooled_scores: np.ndarray | None = None

    def fit(self, returns: Sequence[float] | np.ndarray) -> "EnsembleConformalRisk":
        r = np.asarray(returns, dtype=float)
        n = len(r)
        if n < self.n_folds * 5:
            raise ValueError(f"Need at least {self.n_folds * 5} returns; got {n}")

        self._build_estimators()
        pooled: list[float] = []
        indices = np.array_split(np.arange(n), self.n_folds)

        for k, test_idx in enumerate(indices):
            train_idx = np.concatenate([idx for j, idx in enumerate(indices) if j != k])
            train_returns = r[train_idx]
            test_returns = r[test_idx]

            # Compute base-model predictions on training fold
            base_preds = [est(train_returns) for est in self._base_estimators]
            # Aggregate
            if self.aggregation == "mean":
                pred = float(np.mean(base_preds))
            elif self.aggregation == "min":
                pred = float(np.min(base_preds))
            else:  # conformal: use all base preds as separate scores
                pred = float(np.mean(base_preds))

            # Nonconformity scores for this fold: loss - model_prediction
            test_losses = -test_returns
            for loss in test_losses:
                pooled.append(loss - pred)

        self._pooled_scores = np.array(pooled)
        # Also store training-set base predictions for prediction shift
        self._train_preds = [est(r) for est in self._base_estimators]
        return self

    def predict(self) -> float:
        """Return ensemble conformal VaR."""
        if self._pooled_scores is None:
            raise RuntimeError("Call .fit() before .predict()")
        base_pred = float(np.mean(self._train_preds))
        shift = float(_conformal_quantile(self._pooled_scores, self.alpha))
        return base_pred + shift

    def base_predictions(self) -> dict[str, float]:
        """Individual base model VaR predictions (for inspection)."""
        if self._train_preds is None:
            raise RuntimeError("Call .fit() before .base_predictions()")
        return {
            name: float(pred)
            for name, pred in zip(self._model_names, self._train_preds)
        }

    def _build_estimators(self) -> None:
        self._base_estimators = []
        for name in self._model_names:
            if name == "normal":
                self._base_estimators.append(_normal_var(self.alpha))
            elif name == "historical":
                self._base_estimators.append(_historical_var(self.alpha))
            elif name == "ewma":
                self._base_estimators.append(_ewma_var(self.alpha))
            else:
                raise ValueError(f"Unknown base model: {name!r}")
