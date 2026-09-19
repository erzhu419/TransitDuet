# PointMaze Multiscale Stage-3 V2 Result

Date: 2026-09-20

## Evidence Status

The frozen development campaign
`pointmaze_multiscale_stage3_v2_development_20260920_r1` completed all 160
registered cells: five methods, four scenarios, and eight independent
optimizer roots. The analysis contains 2,560 held-out fixed-horizon episodes.
All result files passed protocol, runtime, seed-role, capacity, finite-value,
option-boundary, stress-channel, and exogenous-pairing audits.

The registered cross-stress Freq-HRL gate is **not supported**. The
multiscale-routing increment over raw-history HRL was positive under both
primary stresses, and clean success noninferiority passed. However, neither
primary hierarchy-by-multiscale success interaction was supported, and the
observation-noise comparison against the causal-filter control was
inconclusive.

## Registered Success Results

| Scenario | Flat history | Flat multiscale | HRL history | HRL multiscale | Flat causal filter |
|---|---:|---:|---:|---:|---:|
| clean | 0.461 | 0.680 | 0.367 | 0.594 | 0.656 |
| fast observation noise | 0.609 | 0.766 | 0.273 | 0.594 | 0.461 |
| slow drift + fast action | 0.594 | 0.648 | 0.414 | 0.672 | 0.664 |
| persistent action shift | 0.555 | 0.680 | 0.297 | 0.523 | 0.625 |

Root-paired registered success contrasts were:

| Scenario and contrast | Mean | 95% CI | Status |
|---|---:|---:|---|
| observation noise: HRL multiscale - HRL history | +0.320 | [+0.132, +0.509] | supported |
| observation noise: factorial interaction | +0.164 | [-0.185, +0.513] | inconclusive |
| observation noise: HRL multiscale - causal filter | +0.133 | [-0.058, +0.323] | inconclusive |
| observation noise: HRL multiscale - flat multiscale | -0.172 | [-0.334, -0.010] | contradicted |
| action stress: HRL multiscale - HRL history | +0.258 | [+0.106, +0.409] | supported |
| action stress: factorial interaction | +0.203 | [-0.043, +0.449] | inconclusive |
| clean: HRL multiscale - HRL history | +0.227 | [-0.018, +0.472] | noninferiority supported at -0.10 margin |
| persistent shift: HRL multiscale - HRL history | +0.227 | [+0.074, +0.379] | supported secondary result |

Return and final-distance endpoints support a representation benefit in
several scenarios, but they do not repair the failed primary success gate.

## Interpretation

The result supports the narrower statement that routed multiscale features can
substantially improve this raw-history HRL implementation under causal stress.
It does not identify a benefit unique to combining hierarchy with multiscale
information. In particular, flat multiscale PPO was significantly better than
Freq-HRL under observation noise and statistically tied under action stress.
This is the boundary anticipated by the GPT6 diagnosis: the current evidence
supports multiscale representation learning more strongly than it supports
Freq-HRL-specific control.

Adding post-hoc roots to seek significance is not authorized. The next
mechanism experiment must distinguish selective frequency routing from generic
Haar conditioning and compression inside the same functioning hierarchy,
rather than rerun this failed gate unchanged.

## Scheduler Accounting

The scheduler archive contains 160 successful terminal tasks. Five initial
attempts were classified as failed despite exit code zero and were retried;
each affected signature has one successful replacement and one valid final
result. Result synchronization now explicitly prefers a successful retry over
an earlier failed record. Only `result.json` artifacts were synchronized.

## Claim Boundary

Allowed: under the frozen PointMaze V2 protocol, multiscale routing improved
the HRL raw-history baseline under observation noise, continuous action stress,
and the secondary persistent shift, while clean success met the registered
noninferiority margin.

Forbidden: V2 establishes a positive hierarchy-by-frequency interaction,
Freq-HRL superiority over flat multiscale PPO or ordinary causal filtering, a
general Freq-HRL advantage, or a confirmatory result.

Machine-readable analysis is under
`results/pointmaze_multiscale_stage3_v2_development_20260920_r1/analysis/`.

