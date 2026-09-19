# PointMaze Frequency-Routing Stage-4 Preflight

Date: 2026-09-20

## Outcome

The 15-cell preflight
`pointmaze_routing_stage4_v1_preflight_20260920_r1` completed as scheduler
tasks `t97442` through `t97456`. All tasks reached `done` and wrote one complete
`result.json`. The run used frozen algorithm revision
`95b31570541af16e99f6bddbb5208e1ce0940f09`; only compact JSON results were
synced locally.

The software and protocol contract passed:

- all five routing controls and three scenarios were present exactly once;
- all episodes used the fixed 64-step preflight horizon;
- every cell performed 32 finite PPO gradient updates;
- every rollout made three upper decisions and three lower option boundaries;
- both hierarchy levels retained full current actor-visible physical state;
- trainable-parameter ratios ranged from 0.993071 to 1.006635;
- optimizer, training, selection, and evaluation seed roles matched the frozen
  specification;
- registered runtime versions matched on every node;
- clean cells reported no injected stress;
- observation-noise cells reported measurement noise only;
- action-stress cells reported slow and fast action stress only;
- stress diagnostics were exactly paired across the five methods, and stress
  truth remained hidden from both actors.

The scheduler dynamically distributed the cells across node001-node006 with
counts 3, 2, 4, 2, 2, and 2 respectively. No task used a hard node binding.

## Boundary

This preflight is software and protocol evidence only. Two training iterations,
one optimizer root, and one held-out episode per cell cannot estimate a routing
effect. It authorizes the frozen 120-cell development matrix without changing
methods, scenarios, seeds, stress amplitudes, capacity bounds, endpoints, or
claim gates.

Allowed: the Stage-4 implementation executes all registered routing controls
under matched capacity, seed, timing, state-feedback, and stress contracts.

Forbidden: the preflight supports selective routing, an all-band or swapped-
routing disadvantage, a causal-filter comparison, or any performance claim.
