"""
conformal-risk quickstart

Shows all four estimators on simulated returns with a mid-series regime shift.
"""
import numpy as np
from conformal_risk import ConformalVaR, ConformalCVaR, AdaptiveConformalRisk, EnsembleConformalRisk

rng = np.random.default_rng(0)

# Simulate: 500 normal-market days, then 100 stress days
normal = rng.normal(-0.0005, 0.01, 500)
stress = rng.normal(-0.003, 0.035, 100)
returns = np.concatenate([normal, stress])

train, test = returns[:400], returns[400:]

print("=" * 60)
print("conformal-risk quickstart")
print("=" * 60)

# ── 1. ConformalVaR ───────────────────────────────────────────────
var = ConformalVaR(alpha=0.05)
var.fit(train)
print(f"\n[ConformalVaR]  95% VaR bound : {var.predict():.4f}")
cov = var.coverage_probability(test)
print(f"                Test coverage : {cov:.1%}  (target >= 95%)")
lo, hi = var.predict_interval()
print(f"                95% interval  : [{lo:.4f}, {hi:.4f}]")

# ── 2. ConformalCVaR ──────────────────────────────────────────────
cvar = ConformalCVaR(alpha=0.05)
cvar.fit(train)
var_b, cvar_b = cvar.predict()
print(f"\n[ConformalCVaR] VaR bound     : {var_b:.4f}")
print(f"                CVaR bound    : {cvar_b:.4f}")
stats = cvar.coverage(test)
print(f"                VaR coverage  : {stats['var_coverage']:.1%}")
print(f"                CVaR coverage : {stats['cvar_coverage']:.1%}")
print(f"                Tail fraction : {stats['tail_fraction']:.1%}")

# ── 3. AdaptiveConformalRisk (ACI) ───────────────────────────────
aci = AdaptiveConformalRisk(alpha=0.05, gamma=0.005, window=252, regime_reset=True)
aci.fit(train)
errors_pre  = []
errors_post = []
for i, r in enumerate(test):
    rec = aci.observe(r)
    if i < 100:    # normal regime
        errors_pre.append(rec["error"])
    else:          # stress regime
        errors_post.append(rec["error"])

pre_cov  = 1 - np.mean(errors_pre)  if errors_pre  else float("nan")
post_cov = 1 - np.mean(errors_post) if errors_post else float("nan")
hist = aci.coverage_history()
print(f"\n[ACI]           Pre-shift coverage  : {pre_cov:.1%}")
print(f"                Post-shift coverage : {post_cov:.1%}")
print(f"                Regime changes      : {hist['regime_changes_detected']}")
print(f"                Final alpha_t       : {aci._alpha_t:.4f}  (nominal: 0.05)")

# ── 4. EnsembleConformalRisk ─────────────────────────────────────
ens = EnsembleConformalRisk(alpha=0.05, n_folds=5)
ens.fit(train)
ens_bound = ens.predict()
base_preds = ens.base_predictions()
print(f"\n[Ensemble]      95% VaR bound : {ens_bound:.4f}")
for name, pred in base_preds.items():
    print(f"                  {name:<12}: {pred:.4f}")

test_coverage = (-test <= ens_bound).mean()
print(f"                Test coverage : {test_coverage:.1%}")

print("\n" + "=" * 60)
print("Coverage guarantee: under exchangeability, ConformalVaR")
print("and EnsembleConformalRisk guarantee >= 95% coverage")
print("regardless of the return distribution shape.")
print("ACI maintains coverage under distribution shift via")
print("adaptive alpha adjustment (Gibbs & Candes, NeurIPS 2021).")
