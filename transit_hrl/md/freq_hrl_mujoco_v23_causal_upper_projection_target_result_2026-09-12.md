# MuJoCo v23 Causal Upper-Target Development Result

## Decision

All 48 frozen cells completed and passed provenance, heldout-path, capacity,
and training-activity checks. Neither causal candidate passed the registered
advancement rule. V23 therefore stops; no fresh-root confirmation is
authorized.

## Full Decision-Time Candidate

The full `decision_time` candidate reduced correction relative to the
zero-consistency reserve in all three environments, but did not preserve the
reward/correction tradeoff of the matched `macro_mean` control:

| Environment | Reward vs old control | Reward wins | Component vs zero / old | Total vs zero / old | Total correction RMS |
|---|---:|---:|---:|---:|---:|
| HalfCheetah-v5 | -11.18% | 2/4 | +28.21% / -18.87% | +44.24% / -9.27% | 0.1227 |
| Hopper-v5 | -9.57% | 0/4 | +26.98% / -5.23% | +43.17% / +9.75% | 0.2526 |
| Walker2d-v5 | +15.39% | 3/4 | +31.21% / +20.93% | +63.30% / +21.58% | 0.0419 |

It won reward in only 5/12 paired optimizer-root comparisons, improved mean
reward over the old control in only one environment, violated the per-domain
reward floor in HalfCheetah and Hopper, and narrowly exceeded the frozen Hopper
correction cap.

## Lower-Only Isolation Arm

The lower-only arm won 8/12 reward comparisons and improved mean reward in
Hopper and Walker2d. It nevertheless failed the HalfCheetah reward floor and
won only 1/4 HalfCheetah roots. More importantly, Hopper component and total
correction regressed 46.49% and 65.92% relative to the old control, producing
total correction RMS 0.4644. This arm does not advance.

## Validity And Diagnosis

Certificates, prefix budgets, fallback limits, and intended consistency
activity all passed. The strict all-cell convergence gate failed for every arm;
minimum convergence rates ranged from 0.9384 to 0.9497, below 0.95. This was
not the sole reason for rejection because both candidates independently failed
reward and correction gates.

The causal label removed future lower states but exposed the upper actor to one
stochastic lower-action draw. Mean upper consistency MSE increased relative to
`macro_mean` in every environment, including 0.663 versus 0.373 in Hopper.
Thus a first-sample target removes hindsight dependence but also removes the
variance reduction supplied by macro averaging.

## Consequence

The `decision_time` first-sample target and the lower-only parameterization are
rejected. V23 roots are retired and cannot be reused to tune coefficients,
thresholds, checkpoint rules, or target aggregation. A continuation must use a
new development panel and a new causal variance-reduction mechanism, such as a
same-state deterministic lower-policy mean target.
