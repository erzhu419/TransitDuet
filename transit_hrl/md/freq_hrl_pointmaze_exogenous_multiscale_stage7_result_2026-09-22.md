# PointMaze Exogenous Multiscale Stage-7 Confirmation Result

Date: 2026-09-22  
Run: `pointmaze_exogenous_multiscale_stage7_v1_confirmation_20260922_r1`  
Experiment protocol: `pointmaze_exogenous_multiscale_stage7_v1_confirmation`  
Frozen algorithm revision: `4ed9e8e131235f0f99844e8ea8bfeb737a68276f`

## Execution Audit

All 64 registered cells completed: four methods at sixteen fresh optimizer
roots. Two initial attempts failed and were replaced by tasks `t100316` and
`t100317` for the same registered cells. No root, environment seed, method, or
gate was added. The final evidence contains 1,024 held-out episodes and 1,024
paired untrained episodes.

The independent audit passed exact method/root completeness, preregistration,
role-seed and exogenous-path pairing, 134-dimensional states, architecture-
matched capacity, 769-entry training histories, finite optimizer updates,
0-based checkpoint schedules, fixed 300-step episodes, twelve HRL upper
decisions per episode, disabled legacy mechanisms, runtime identity, and
independent reproduction of every primary Student-t interval. Only compact
`result.json` files were synced.

## Absolute Results

| Method | Tracking success, mean [95% CI] | Return, mean [95% CI] | RMSE |
|---|---:|---:|---:|
| Flat history | 0.872 [0.836, 0.907] | 225.816 [221.736, 229.896] | 0.319 |
| Flat all-band | 0.850 [0.778, 0.921] | 223.100 [213.781, 232.418] | 0.343 |
| HRL history | 0.906 [0.887, 0.926] | 235.242 [231.808, 238.676] | 0.285 |
| HRL all-band | 0.908 [0.874, 0.942] | 235.141 [230.909, 239.373] | 0.284 |

HRL all-band learned reliably relative to its paired untrained state: success
improved by 0.730 [0.695, 0.764] and return by 120.965 [115.527, 126.403]. Its
absolute-success condition passed.

## Frozen Primary Contrasts

| Contrast | Success improvement [95% CI] | Decision |
|---|---:|---|
| HRL all-band - HRL history | +0.001823 [-0.028916, +0.032562] | inconclusive |
| HRL all-band - flat all-band | +0.058359 [-0.000041, +0.116760] | inconclusive |
| Hierarchy x multiscale interaction | +0.023607 [-0.048801, +0.096015] | inconclusive |

The lower bound for HRL all-band versus flat all-band is
`-0.000041436945`. It remains below zero and must not be rounded into a
positive interval.

All three frequency-specific confirmation conditions failed. Therefore the
full confirmation conjunction is **not supported**.

## Secondary Boundary

Ordinary HRL history improved return over flat history by 9.426 [4.463,
14.389], improved RMSE by 0.034 [0.011, 0.057], and improved final distance by
0.045 [0.004, 0.087]. Its success contrast remained inconclusive. HRL
all-band likewise improved return and RMSE over flat all-band, but did not pass
the preregistered success contrast.

These secondary outcomes support the ordinary goal-conditioned hierarchy on
this task more strongly than they support any frequency representation.

## Decision

The Stage-6 development signal did not replicate. Fixed hard routing was
already rejected, and Stage 7 now rejects the narrower claim that the complete
Haar representation provides a confirmed hierarchy-specific success benefit.
No sequential root extension is authorized.

The current PointMaze evidence supports reliable goal-conditioned HRL and a
development-only multiscale observation, not a confirmed Freq-HRL algorithm.
A subsequent mainline must introduce a genuinely new decision-relevant
temporal mechanism, such as learned regime/innovation inference with
event-triggered replanning, and test it first as new development work. Merely
changing wavelet bases, adding roots, or selecting a secondary endpoint cannot
repair this confirmation.
