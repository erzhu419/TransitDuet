# Stage277: observable-state residual comparison

- Run: `native_transit_state_risk_stage277_20261017_r1`.
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
- Status: source/tests ready; scheduler submission pending.
- Limits: reused development prefixes/futures and repeated models; equal coefficient count does not guarantee equal effective capacity. Not policy or end-to-end HRL validation. Stage253/248/169 remain closed.
