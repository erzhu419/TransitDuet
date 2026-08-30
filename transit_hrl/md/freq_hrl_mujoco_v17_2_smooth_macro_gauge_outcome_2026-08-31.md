# MuJoCo v17.2 Smooth Macro Gauge Outcome

## Decision

`smooth_macro_gauge_preflight_not_supported`

This is valid development evidence, not confirmatory evidence. All nine
scheduleurm tasks (`t85416`--`t85424`) completed on `node003` with dynamic
placement and no node binding. The local result bundle is 1.8 MB and contains
only summaries, evaluation CSV files, and server artifact locations; checkpoints
and training histories remain on the worker.

## Paired Mechanics

Each of the nine environment-by-alpha cells trained one strength-zero policy and
then evaluated that frozen checkpoint at strengths zero and one on the same 40
paths. Across all 360 pairs:

- reward trace hashes matched exactly;
- executed-action trace hashes matched exactly;
- latent-policy trace hashes matched exactly;
- numeric reward and latent frequency metrics matched;
- router and responsibility reconstruction passed the frozen `1e-7` RMS gate;
- upper and lower transition counts satisfied the asynchronous hierarchy
  contract.

The function-preserving implementation therefore works as intended. The
frequency target does not.

## Frozen Frequency Result

| Alpha | Upper-HPF8 pass | Lower-LPF32 pass | Joint-merit pass | Median joint reduction | Worst joint reduction |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 2/3 | 0/3 | 0/3 | -22.87% | -191.97% |
| 0.10 | 1/3 | 0/3 | 0/3 | -151.29% | -202.03% |
| 0.20 | 2/3 | 0/3 | 0/3 | -153.54% | -505.39% |

The naive smooth gauge often reduced upper-HPF8 power in HalfCheetah and
Hopper, but the delayed low-pass target moved slow residual structure into the
lower complement. HalfCheetah lower-LF power worsened by 30.6% at alpha 0.05
and by more than 700% at the other alphas. Hopper worsened by 157.8%--213.9%.
Walker2d did not repair the mechanism: alpha 0.05 improved lower LF by only
2.68% while increasing upper-HF power by a factor of roughly 68 relative to the
small control baseline; the other alphas worsened both registered endpoints.

## Analysis Repair

The first frozen-analysis invocation rejected one cell before reading its
frequency outcome. The analyzer had added two checks that were not in the
preregistered gate:

1. it required the legacy aggregate `protocol_valid` bit, whose internal
   float32 maximum-error tolerance is stricter than the registered RMS gate;
2. it classified additive clip excess as a latent-policy metric.

Two of 360 candidate rows had the legacy bit unset, while their maximum reported
responsibility reconstruction RMS was `1.75e-8` and router reconstruction was
zero. Revision `8e52f0fc57` replaced the extra bit with direct transition-count,
finiteness, trace, and RMS checks, and restricted latent numeric equality to the
actual latent metrics. Regression tests cover both cases. No frequency
threshold, alpha, seed, row, or selection rule changed. The final decision is
still not supported because lower-LF and joint merit failed in every cell.

## Diagnosis And Next Step

Smoothing a sampled EMA target addresses macro-boundary jumps but does not
address causal lag. The lower complement must absorb the difference between the
current total action and a delayed upper plan, so slow changes reappear as lower
drift. Alpha tuning cannot fix this pattern.

The next mechanism must optimize the registered objective directly. At each
upper boundary it should build a causal finite-horizon forecast from total-action
history, choose a bounded frozen curve that minimizes predicted upper HPF8 plus
lower LPF32 merit, and preserve exact additive execution. Before any
leakage-active training, the policy and cost critic must observe the plan's
low-pass state, current value, target, and phase. New development seeds are
required; v17.2 paths cannot be reused for selection.

## Claim Boundary

Allowed: v17.2 validates pathwise function preservation and rejects the naive
EMA-target smooth macro gauge on a frozen three-environment development panel.

Forbidden: v17.2 validates frequency separation, leakage no-tradeoff, reward
improvement, learned constraint improvement, optimizer-seed robustness, or a
final Freq-HRL algorithm.
