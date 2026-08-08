# Freq-HRL MuJoCo v5 Leakage Pareto And v6 Design

Date: 2026-08-08

## Evidence Boundary

All results in this note are development-only. The optimizer replicates and
held-out environment seeds have already been inspected and cannot be reused as
confirmatory evidence. The purpose of the matrix is to reject or refine the
protocol before committing a larger compute budget.

## v5 Completion And Integrity

- The v5 preflight completed 4/4 source-bound cells and the audit reported zero
  issues and zero warnings.
- The v5 pilot completed 36/36 cells: three MuJoCo environments, four methods,
  and three independent optimizer replicates.
- Its audit covered 900 held-out episode rows and reported zero issues and zero
  warnings.
- Standard zero-disturbance checkpoints for `freq_hrl_no_leakage` and
  `generic_hrl` were identical under common random numbers. This confirms that
  the new frequency route has no hidden behavior difference when its explicit
  exogenous input is zero.

The standard-condition return means were:

| environment | flat PPO | generic HRL | Freq-HRL no leakage | Freq-HRL |
|---|---:|---:|---:|---:|
| HalfCheetah-v5 | 830.05 | 838.72 | 965.97 | 921.32 |
| Hopper-v5 | 247.50 | 184.72 | 243.34 | 208.78 |
| Walker2d-v5 | 251.02 | 264.85 | 259.38 | 256.96 |

The structural v5 repair is therefore real, but the fixed leakage constraint
still introduces a reward tradeoff.

## Fixed-Budget Sweep

Four additional source-bound sweeps used the same development seeds and only
changed the lower-controller LF RMS budget. Every sweep completed 9/9 cells,
was result-synced through scheduleurm, and passed the integrity audit. The
following values average the five evaluation disturbance conditions after
first averaging paths within each independent optimizer replicate.

### HalfCheetah-v5

| budget | return | delta vs no leakage | drift ratio vs no leakage |
|---:|---:|---:|---:|
| 0.050 | 765.82 | -17.12 | 0.53 |
| 0.075 | 535.52 | -247.42 | 1.55 |
| 0.100 | 801.43 | +18.49 | 1.14 |
| 0.150 | 942.68 | +159.74 | 2.08 |
| 0.200 | 782.94 | +0.00 | 1.00 |

### Hopper-v5

| budget | return | delta vs no leakage | drift ratio vs no leakage |
|---:|---:|---:|---:|
| 0.050 | 204.26 | -41.29 | 0.04 |
| 0.075 | 235.43 | -10.13 | 0.10 |
| 0.100 | 242.47 | -3.09 | 0.15 |
| 0.150 | 221.03 | -24.53 | 0.64 |
| 0.200 | 295.12 | +49.57 | 0.83 |

### Walker2d-v5

| budget | return | delta vs no leakage | drift ratio vs no leakage |
|---:|---:|---:|---:|
| 0.050 | 264.42 | -4.02 | 0.05 |
| 0.075 | 255.42 | -13.02 | 0.11 |
| 0.100 | 264.35 | -4.09 | 0.24 |
| 0.150 | 265.89 | -2.55 | 0.74 |
| 0.200 | 264.78 | -3.66 | 0.53 |

No fixed budget gives a scale-invariant no-tradeoff solution. Budget 0.10 is
a useful operating point for Hopper and Walker2d, but it does not reduce
HalfCheetah drift. Selecting an environment-specific budget from held-out
returns would be post-hoc and is not an acceptable repair.

## v6 Algorithmic Repair

Protocol `freq_hrl_mujoco_shared_core_v6_reward_guarded_projection` replaces
the lower actor's scalarized reward-plus-cost update with two stages:

1. apply the ordinary reward PPO update through its existing Adam optimizer;
2. compute the leakage-cost gradient at the resulting policy;
3. remove any cost-gradient component that opposes reward descent;
4. apply the remaining correction with a geometric backtracking line search;
5. accept the correction only if the same minibatch's reward surrogate does
   not regress and its leakage surrogate does not increase; otherwise restore
   the reward-only parameters exactly.

The implementation reports gradient conflict, projected-gradient norm,
acceptance, backtracking count, reward-surrogate delta, and cost-surrogate
delta. The old `scalarized` behavior remains available for backward
compatibility.

This mechanism gives a local, sampled-surrogate guard. It is not a theorem that
episode return cannot decrease: finite-sample estimation, state-distribution
shift, and optimizer dynamics remain. Any manuscript statement must preserve
that boundary unless fresh confirmatory experiments support a stronger claim.

## Compute And Scheduler Incident

The old `jtl110cpu` records showing approximately 110 hours of runtime are not
valid long-running experiments. Live probes repeatedly fail during SSH key
exchange, remote PIDs cannot be confirmed, and result synchronization cannot
be verified. These are stale lost-contact scheduler records and are excluded
from all evidence. New MuJoCo work uses only dynamic, unpinned placement over
`node001` through `node006`, where PID, CPU, RAM, log, exit marker, and result
artifact are all independently observable.

## Next Gate

Freeze v6 source, run a four-cell preflight, then compare v6 against the frozen
v5 constrained and no-leakage policies under the same development CRN matrix.
Only a consistent Pareto improvement may advance to fresh-seed confirmatory
evaluation.
