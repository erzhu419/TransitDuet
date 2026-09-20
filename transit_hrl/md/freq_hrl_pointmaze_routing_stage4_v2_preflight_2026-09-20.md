# PointMaze Frequency-Routing Stage-4 V2 Preflight

Date: 2026-09-20

## Outcome

The 15-cell equal-shape preflight
`pointmaze_routing_stage4_v2_preflight_20260920_r1` completed as scheduler
tasks `t99438` through `t99452`. All tasks reached `done`; only compact
`result.json` files were synchronized. The run used frozen algorithm revision
`f9ab0b4a532d1bc0c31466b24d4dcbc70634e585`.

The registered software contract passed:

- all five methods and three scenarios were present exactly once;
- upper and lower states were 134-dimensional in every cell;
- every model had hidden size 79 and exactly 68,424 trainable parameters;
- unit-level parameter comparison verified identical initial weights across
  methods for a matched optimizer root;
- routed and swapped masks selected the registered coefficient slots while
  retaining all fixed-shape slots;
- every cell performed 32 finite PPO gradient updates;
- all episodes used 64 steps, three upper decisions, and three lower option
  boundaries;
- fresh optimizer, training, selection, and evaluation seeds matched the
  frozen specification;
- runtime, current-physical-feedback, and disabled-mechanism contracts held;
- clean, observation-noise, and action-stress channels were independently
  injected and exactly paired across methods;
- disturbance truth remained hidden from both actors.

The scheduler dynamically placed 3 cells on node001 and 4 each on node004,
node005, and node006. No task was hard-bound to a node.

## Boundary

This is software and protocol evidence only. Two iterations, one optimizer
root, and one held-out episode per cell cannot estimate a performance effect.
It authorizes the unchanged 120-cell V2 development matrix.

Allowed: V2 removes the V1 per-level shape/capacity confound in executable
training and evaluation paths.

Forbidden: this preflight supports a routing direction, performance benefit,
Freq-HRL claim, or any conclusion from the observed smoke rewards.
