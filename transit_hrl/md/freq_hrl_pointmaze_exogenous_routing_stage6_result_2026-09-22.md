# PointMaze Exogenous Frequency-Routing Stage-6 Result

Date: 2026-09-22  
Run: `pointmaze_exogenous_routing_stage6_v1_development_20260922_r1`  
Protocol: `pointmaze_exogenous_frequency_routing_stage6_v1`  
Frozen algorithm revision: `4ed9e8e131235f0f99844e8ea8bfeb737a68276f`

## Execution Audit

The frozen 64-cell matrix completed all eight methods at all eight optimizer
roots. Four initial scheduler attempts failed, and tasks `t100114`-`t100117`
successfully reran the same registered cells; no seed, method, or protocol was
added. The final evidence contains 1,024 held-out episodes and 1,024 paired
untrained episodes.

The independent audit passed method/root completeness, exact role-seed and
external-path pairing, 134-dimensional states, identical capacity within each
architecture, 769-entry training histories, finite PPO updates, causal stream
visibility, 300-step fixed horizons, twelve HRL upper decisions per episode,
disabled legacy mechanisms, runtime identity, and independent reproduction of
the root-level Student-t intervals. Only compact `result.json` artifacts were
synced.

## Absolute Results

| Method | Tracking success, mean [95% CI] | Return | RMSE |
|---|---:|---:|---:|
| Flat history | 0.933 [0.877, 0.990] | 229.547 | 0.298 |
| Flat causal filter | 0.729 [0.635, 0.822] | 213.576 | 0.404 |
| Flat all-band multiscale | 0.875 [0.791, 0.959] | 230.545 | 0.301 |
| HRL history | 0.880 [0.837, 0.922] | 233.520 | 0.296 |
| HRL causal filter | 0.859 [0.826, 0.891] | 228.508 | 0.320 |
| HRL all-band multiscale | 0.929 [0.906, 0.953] | 236.712 | 0.275 |
| HRL intended routing | 0.935 [0.899, 0.971] | 239.599 | 0.263 |
| HRL swapped routing | 0.945 [0.927, 0.963] | 239.921 | 0.260 |

The routed policy learned reliably relative to its paired untrained state:
success improved by 0.749 [0.716, 0.781] and return by 124.816 [120.484,
129.148]. Its absolute-success gate also passed.

## Registered Attribution Contrasts

| Root-paired contrast | Success improvement [95% CI] | Decision |
|---|---:|---|
| Flat filter - flat history | -0.205 [-0.326, -0.083] | contradicted |
| Flat all-band - flat history | -0.058 [-0.166, 0.049] | inconclusive |
| HRL filter - HRL history | -0.021 [-0.060, 0.017] | inconclusive |
| HRL all-band - HRL history | +0.050 [+0.003, +0.096] | supported |
| Routed - HRL history | +0.055 [+0.033, +0.077] | supported |
| Routed - HRL filter | +0.076 [+0.040, +0.113] | supported |
| Routed - HRL all-band | +0.005 [-0.041, +0.052] | inconclusive |
| Routed - swapped | -0.010 [-0.050, +0.029] | inconclusive |

The hierarchy-by-multiscale success interaction was +0.108 [-0.017, 0.233],
so it was inconclusive. Return, RMSE, and final-distance interactions were also
inconclusive.

## Decision

The strict Freq-HRL selective-routing conjunction is **not supported**. Three
registered conditions failed: intended routing did not beat all-band input,
did not beat swapped routing, and did not establish a positive
hierarchy-by-multiscale interaction.

The defensible development finding is narrower: all-band causal multiscale
features improved HRL success over raw history, and intended routing improved
over history and causal filtering. The experiment does not identify the fixed
slow-upper/high-lower assignment as the source of that gain. Swapped routing
must not be relabeled as the intended mechanism.

Together with the equal-shape Stage-4 result, this closes fixed hard masking as
the mainline algorithm. The next independent test should confirm the simpler
two-by-two history-versus-all-band, flat-versus-HRL effect on fresh roots. It
should not add more roots to this failed routing gate or tune masks on these
held-out paths.
