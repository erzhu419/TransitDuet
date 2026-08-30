# MuJoCo v17.6 Full-Horizon Oracle Outcome

## Decision

`mixed_router_recoverability_and_actor_floor`

The remaining v17.4 frequency-budget failures are not caused by one mechanism.
Most are responsibility-router failures, while a smaller Hopper subset cannot
meet both budgets without changing the learned total-action trajectory.

## Integrity

All 120 scheduleurm tasks (`t85559`--`t85678`) completed on `node003` with no
failure or cancellation. The placement was data-local because the three source
checkpoints remained server-only on that node. Only one small JSON result per
path was synchronized; checkpoints and trajectory arrays were not downloaded.

Every frozen v17.4 path replayed exactly. The maximum responsibility
reconstruction error was `5.56e-17`, the maximum KKT residual was `2.11e-13`,
the maximum BVLS optimality residual was `1.05e-13`, and no box constraint was
violated. The oracle therefore satisfies its registered numerical contract.

## Path Classification

| Environment | Paths | Baseline feasible | Oracle feasible | Router-recoverable | Actor-floor failures |
|---|---:|---:|---:|---:|---:|
| HalfCheetah-v5 | 40 | 0 | 40 | 40 | 0 |
| Hopper-v5 | 40 | 0 | 33 | 33 | 7 |
| Walker2d-v5 | 40 | 32 | 40 | 8 | 0 |
| **Total** | **120** | **32** | **113** | **81** | **7** |

The full-horizon split lowered mean lower-LPF32 power from `0.0091420` to
`0.0008652` while satisfying the upper budget on 113 paths. Eighty-one paths
that failed under the online v17.4 split became jointly feasible without
changing any executed action.

The seven remaining failures are all Hopper paths. They cover low-frequency,
standard, high-frequency, OOD-chirp, and mixed disturbances and concentrate on
evaluation seeds `294864529` and `2802248628`. The upper budget itself is
physically feasible on every path. However, after imposing that upper budget,
the minimum attainable lower power on these seven traces remains above the
registered lower budget. This is a total-action spectral limitation, not a
router numerical failure.

## Consequence

The next method needs two changes:

1. A causal finite-horizon responsibility optimizer or a policy distilled from
   the full-horizon oracle, because the current online router leaves 81
   feasible paths unresolved.
2. An actor-level trajectory feasibility cost, because no responsibility split
   can repair the seven Hopper total-action traces.

This experiment reuses rejected v17.4 paths and solves an acausal oracle. It is
a development mechanism diagnosis only. It does not support a reward,
generalization, or final-algorithm claim.
