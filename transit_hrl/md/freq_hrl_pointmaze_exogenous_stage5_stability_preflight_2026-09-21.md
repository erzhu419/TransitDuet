# PointMaze Stage-5 Stability-Screen Preflight

Date: 2026-09-21

Run: `pointmaze_exogenous_stage5_stability_v1_preflight_20260921_r1`

Tasks `t99981` through `t99984` completed on node003, node004, node006,
and node005. The four cells covered both checkpoint-rank modes and both
training-rollout counts.

The execution audit passed:

- resolved rank modes and 4/8 training-seed counts matched the frozen arms;
- train, selection, and evaluation seeds matched the preflight registry;
- all cells used one HRL method, 134-dimensional upper/lower states, and
  68,424 trainable parameters;
- PPO updates were finite and nonzero;
- runtime versions were identical;
- each result contained one final and one paired untrained episode; and
- only compact `result.json` artifacts were synchronized.

This is a software and execution-path check. Its two-iteration checkpoint
choices and episode outcomes are not performance evidence. It authorizes the
eight-cell development screen registered in
`freq_hrl_pointmaze_exogenous_stage5_stability_protocol_2026-09-21.md`.

