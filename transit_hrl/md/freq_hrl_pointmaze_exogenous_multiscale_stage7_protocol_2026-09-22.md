# PointMaze Exogenous Multiscale Stage-7 Confirmation Protocol

Date: 2026-09-22  
Evidence stage: confirmation  
Experiment protocol: `pointmaze_exogenous_multiscale_stage7_v1_confirmation`  
Runtime protocol: `pointmaze_exogenous_frequency_routing_stage6_v1`  
Frozen algorithm revision: `4ed9e8e131235f0f99844e8ea8bfeb737a68276f`

## Question

Stage 6 did not support fixed slow-upper/high-lower routing, but it produced a
registered development signal for all-band multiscale HRL versus HRL history.
Stage 7 tests that narrower result on entirely fresh optimizer and environment
seeds. It asks whether a complete causal multiscale representation has a
hierarchy-specific control benefit; it does not reopen selective routing.

## Frozen Factorial

The four methods are:

1. `flat_exogenous_history`;
2. `flat_exogenous_multiscale_all`;
3. `hrl_exogenous_history`;
4. `hrl_exogenous_multiscale_all`.

Raw history and Haar coefficients are transformations of the same 32 causal
external samples. All states remain 134-dimensional. Methods within each
architecture have identical capacity and initial parameters for a given root.
Both hierarchy levels retain current physical feedback. Projection, promotion,
leakage loss, responsibility gauge, and hard frequency masks are disabled.

## Fixed Sample And Runtime

- Formal optimizer roots: 16, fixed before any confirmation run.
- Cells: 64; no interim analysis or sequential root extension is allowed.
- Per cell: 8 training, 16 selection, and 16 held-out evaluation seeds.
- Training: 768 iterations; horizon: 300 primitive steps.
- Checkpoint selection: success first, then dense return, every 96 iterations.
- Statistical unit: optimizer root; episode outcomes are averaged within root.
- Interval: two-sided 95% Student-t CI over 16 independent root means.
- Scheduler: one CPU and 2560 MB per cell, dynamic node001-node006 placement.
- Synced artifact: compact `result.json` only.

The 16-root size was fixed from Stage-6 development dispersion. At that
dispersion it gives projected success-effect half-widths of approximately
0.030 for HRL all-band versus HRL history and 0.080 for the factorial
interaction. These are planning values, not guaranteed confirmation effects.

## Confirmation Gate

The multiscale-HRL claim is supported only if all conditions hold:

1. HRL all-band absolute success CI lower bound is at least 0.50;
2. HRL all-band improves success and return over its paired untrained policy;
3. HRL all-band improves success over HRL history;
4. HRL all-band improves success over flat all-band;
5. the hierarchy-by-multiscale success interaction has a positive CI.

All secondary return, RMSE, final-distance, flat representation, and ordinary
hierarchy contrasts are reported without replacing a failed primary gate.

Even a fully supported result would establish only a fresh PointMaze external-
stream multiscale-HRL effect. It would not validate fixed band routing,
promotion, leakage control, action-spectrum constraints, Transit deployment,
or domain-general superiority.
