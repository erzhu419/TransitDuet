# Stage277: observable-state residual comparison

- Run: `native_transit_state_risk_stage277_20261017_r2`.
- Purpose: test whether true observable conditioning improves fleet-risk residual transfer, not only training fit.
- Same-capacity arms: 8 action slopes + 64 full context interactions + intercept, 73 coefficients each.
- Baseline context: eligible-action rank8 centroid; treatment: clock, own/other queue, onboard, own/other active fleet, remaining-time mean, LF forecast.
- Normalize context on each fit subset, apply tanh, then normalize lifted features on that subset; same physical-MSE ridge penalty 0.01.
- Reuse all 32 cached cells, both assigned16 banks, full60/prefix54 splits, raw prior, eligibility and original-action anchor.
- Pointwise clip radii unchanged from Stage272. Keep cached linear9 and RMS-bounded controls.
- Evaluate assigned/opposite16 on train, retained, excluded, outer held and identical prefix54 heads on outer held.
- Primary: mixed-pooled excluded/opposite and full60-held/opposite; report both banks and all views without selecting an arm or penalty.
- Budget: 1 scheduler CPU task, 1,408 lightweight residual fits; zero raw/prior fits, simulations, action selections or actor updates.
- Large caches, models and predictions remain remote; pull scalar report only.
- Completed: `t166087`, node004, done/0, no retries, 1,408 residual fits; normal-equation error 3.73e-17. Cached prediction/scalar-error reproduction both exactly zero. Pulled only 394,656 bytes. Queued r1 was cancelled before fitting; no extra fits.

## Results
Mixed-pooled clipped MSE x 1e6 (lower is better); these are repeated model/state exposures, not independent seeds.

| Population / label bank | Linear9 | Action-context73 | Observed-context73 |
| --- | ---: | ---: | ---: |
| Full60 training / assigned16 | 575.102 | 509.005 | 494.104 |
| Full60 training / opposite16 | 627.827 | 613.826 | 608.112 |
| Prefix54 retained / opposite16 | 628.047 | 612.547 | 608.032 |
| Prefix54 excluded / opposite16 | 670.899 | 705.709 | 694.170 |
| Full60 outer held / opposite16 | 868.048 | 922.265 | 895.584 |
| Same prefix54 heads, outer held / opposite16 | 871.484 | 924.451 | 901.443 |

Observed beats same-capacity action context by 1.64% excluded / 2.89% outer-held, but remains 3.47% / 3.17% worse than Linear9. It also fails to improve raw risk MSE (655.806 / 895.036).
This is not uniform: excluded bank0 reverses (+1.417e-6); root397 excluded +21.161e-6; root401 excluded/held +35.504/+47.502e-6 relative to action context. No stable conditioning superiority claim.
Without clip, observed/action MSE is 143.272/187.760 on training-assigned, but 728.006/694.795 on training-opposite; excluded observed MSE reaches 5,226.962. Observed clip fraction rises from 58.2% training to 72.1% excluded and 67.7% held. Better interpolation coexists with bank sensitivity and extrapolation error, not a restored fleet-risk expectation.
Next Stage278: feature-only effective-df matching to Linear9, preserving the maps, training subsets, labels, radii and evaluation populations; no label-based penalty search.
- Limits: reused development prefixes/futures and repeated models; equal coefficient count does not guarantee equal effective capacity. Not policy or end-to-end HRL validation. Stage253/248/169 remain closed.
