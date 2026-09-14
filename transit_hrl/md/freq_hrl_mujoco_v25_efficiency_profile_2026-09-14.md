# Freq-HRL MuJoCo v25 Efficiency Profile

## Decision

Keep the stable terminal-reserve solver at revision `a047161cc8`. On one
matched HalfCheetah v25 action-sample diagnostic cell, wall time fell from
2,078.14 s to 802.90 s, a 2.59x end-to-end speedup. This is an efficiency
result for one representative cell, not policy-performance evidence.

## Profile

Each run used one CPU and the same roots and executed 53,384 projector steps:
16,384 training, 32,000 crossed checkpoint selection, and 5,000 final
evaluation steps. Raw `.prof` files stayed on the assigned compute node.

| Run | Task / revision | Wall (s) | Calls | Projector (s) | Dykstra (s) | Decision |
|---|---|---:|---:|---:|---:|---|
| r1 | `t93922` / `a066456502` | 2,078.14 | 3.048B | 1,960.10 | 1,872.28 | baseline |
| r2 | `t93930` / `1372fa7ead` | 862.77 | 564.4M | 746.38 | 633.05 | retained |
| r3 | `t93937` / `78220f3ca9` | 788.68 | 519.6M | 675.64 | 574.94 | rejected |
| r4 | `t93942` / `a047161cc8` | 802.90 | 595.1M | 685.74 | 622.41 | retained |

In r4 the projector still consumed 85.5% of profiled time and Dykstra 77.6%.
MuJoCo `mj_step` consumed only 2.97 s. The bottleneck is the responsibility
projection, not environment simulation or PPO tensor operations.

## Retained Changes

- Replace closure-based generic Dykstra with a balls-plus-box implementation.
- Inline ball norms and reuse preallocated residual and work buffers.
- Remove exactly containing balls while preserving active-constraint order.
- Return immediately for already feasible starts.
- Vectorize prefix-sum terminal certificates and cumulative ball construction.
- Preserve the projection tolerance and the 512-iteration cap.

The frozen-reference trajectory benchmark improved from 14.382 s to 4.609 s
(3.12x). Mean iterations remained 39.431, the maximum remained 512, and the
convergence rate remained 98.4375%. A separate 10,000-certificate benchmark
improved from 4.286 s to 1.498 s (2.86x). The focused suite passes 29 tests,
including differential trajectories in dimensions 1, 3, and 6 at `rtol=1e-9`
and `atol=2e-8`.

## Rejected Changes

The r3 exact single-active-ball shortcut is not adopted. In a 30-trajectory
audit, one trajectory differed by 9.6e-4 when the frozen 512-iteration
reference had not converged. Its closed-loop diagnostic also moved the lower
LF behavior to the budget boundary. Mathematical feasibility alone did not
justify changing the implemented finite-iteration solver.

A violated-ball active-subset solve is also rejected: its local benchmark
regressed from 4.37 s to 6.07 s. Both negative branches remain documented;
neither is present in the final tree.

## Scientific Boundary

Small floating-point differences are amplified by PPO. Between r2 and r4,
218 numeric summary fields and checkpoint hashes changed. Overall diagnostic
reward moved from -0.01078 to -0.01008, while lower LF power moved from
0.001244 to 0.001622; neither contrast is an algorithm claim. Therefore r4 is
a new implementation revision and must not be mixed with the already running
v25 cells, which were staged from the old solver and retain their provenance.

## Next Optimization

A full v25 cell performs 1,616,576 rollout steps: 1,048,576 training, 528,000
selection, and 40,000 final evaluation. The 4 training roots, 16 crossed
selection roots, and 40 final evaluation paths are independent at each frozen
policy state. An ordered process map can use about 20 workers per cell; across
48 cells this uses about 960 of the 1,152 physical cores. The ideal stage
model is about 5.4x faster per cell before serialization and PPO-update costs.

Implement this only with deterministic seed isolation, ordered aggregation,
a one-worker versus multi-worker equivalence test, and a fresh source
manifest. Do not reduce tolerance or iteration limits merely for speed.

## Reproduction

Use `scripts/submit_mujoco_v25_cprofile_scheduleurm.py` to dispatch the compact
diagnostic and `scripts/run_mujoco_cprofile_small_export.py` on the worker.
Only `profile_summary.json`, `profile_top.txt`, and `cell_summary.json` are
retrieved; checkpoints, histories, and raw profiles remain server-only.
