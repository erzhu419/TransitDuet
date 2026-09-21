# PointMaze Exogenous-Control Stage-5 V1 Protocol

Date: 2026-09-21

## Frozen Code

- Protocol: `pointmaze_exogenous_control_stage5_v1`
- Algorithm revision: `df516684f8ea2fbb89fb65fa038e11a44a005200`
- Environment: `PointMaze_UMaze-v3`
- Methods: `flat_exogenous_history`, `hrl_exogenous_history`

This protocol freezes the separate external-stream substrate described in
`freq_hrl_pointmaze_exogenous_stage5_design_2026-09-21.md`. Frequency routing,
promotion, leakage loss, responsibility gauges, and action-spectrum projection
are disabled.

## Training Matrix

The development matrix contains 16 independent cells: two methods crossed with
eight optimizer roots. Each cell uses 768 PPO iterations, four training rollout
roots, 16 disjoint checkpoint-selection seeds, and 16 disjoint held-out
evaluation seeds. Each evaluation seed is run both before training and after
frozen checkpoint selection. Episodes use the fixed 300-step horizon.

The flat reference hidden size is 128. The hierarchical hidden size is selected
by the existing parameter matcher. Both methods receive the same 32-step causal
external history and current physical feedback.

## Physical-Time Contract

- environment control interval: 0.01 seconds;
- upper waypoint period: 0.25 seconds;
- external-history duration: 0.32 seconds;
- registered fast period: 0.04 seconds;
- target speed: 1.0 world unit per second;
- target round-trip period: 12 seconds;
- measured force RMS: 0.12 per axis;
- measured force period: 0.04 seconds per axis.

The target and force stream is generated only from the evaluation seed and is
identical under different action sequences. Current values are visible before
the action; future values are not visible.

## Frozen Gate

The statistical unit is the optimizer root. Held-out episodes are averaged
within root before root-level 95% t intervals are formed. The gate is supported
only if all conditions hold:

1. hierarchical tracking-success CI lower endpoint is at least 0.50;
2. root-paired final-minus-untrained hierarchical tracking success has a
   strictly positive 95% interval; and
3. root-paired final-minus-untrained hierarchical episode return has a
   strictly positive 95% interval.

HRL-versus-flat tracking and return effects are reported but do not gate
admission. Passing admits a separately frozen frequency-routing experiment; it
does not itself establish any frequency benefit.

## Execution

Each cell requests one CPU core and 2,560 MB RAM. Scheduler placement is
dynamic across `node001` through `node006`; no cell is pinned to a node. Only
compact `result.json` artifacts are synchronized to the local worktree.

