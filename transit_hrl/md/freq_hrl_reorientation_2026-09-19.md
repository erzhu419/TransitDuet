# Freq-HRL Research Reorientation

Date: 2026-09-19

## Decision

The main research object is now:

> Multiscale information for goal-conditioned hierarchical control: long-term
> planning, short-term execution, and change detection use causal
> representations matched to their control responsibilities.

It is no longer assumed that every actuator command must be decomposable into
an LF upper torque and an HF lower torque. Persistent lower-level actuation can
be necessary for balance, drag compensation, and goal tracking.

## Two Algorithm Paths

### Mainline: multiscale goal-conditioned HRL

The upper policy produces a state-space goal:

```text
goal_k = upper(task state, slow task information, regime information)
```

The lower policy alone produces physical action:

```text
action_t = lower(full physical state, goal_k, mid/fast task information)
```

The implementation is in:

- `freq_hrl/core/multiscale.py`
- `freq_hrl/rl/goal_conditioned_actor_critic.py`
- `freq_hrl/experiments/multiscale_tracking_validation.py`
- `freq_hrl/domains/mujoco/goal_adapter.py`

The first protocol disables action-spectrum projection, promotion, leakage
loss, responsibility gauge, and projection-consistency fitting. Upper/lower
credit uses the existing asynchronous SMDP trajectory accounting.

### Side branch: constrained actuator spectrum

`freq_hrl/experiments/mujoco/control_validation.py` remains available for tasks
that actually impose actuator bandwidth or spectral budgets. Its result rows
are labeled `spectral_action_constraint_side_branch`. Results from this branch
cannot by themselves support the mainline multiscale-HRL claim.

## Stage-1 Identification Protocol

The protocol has four capacity-matched learned-policy cells:

| Control structure | Raw causal history | Causal multiscale representation |
|---|---|---|
| Flat | `flat_history` | `flat_multiscale` |
| Goal-conditioned HRL | `hrl_history` | `hrl_multiscale` |

`flat_causal_filter` is an auxiliary, capacity-matched low-pass baseline for
the noisy conditions. It distinguishes a general denoising benefit from a
multiscale representation benefit; it is not a fifth cell in the two-by-two
factorial claim.

History and multiscale arms use the exact same fixed trailing samples. The
multiscale arm applies an orthonormal causal-window Haar transform, so it does
not receive a longer hidden filter history. Band labels use physical support in
seconds, not a universal number of simulator steps. A slow-band energy
envelope is available to the upper policy.

The four frozen mechanism scenarios are:

1. `clean`
2. `slow_target_fast_force`
3. `slow_signal_fast_observation_noise`
4. `band_swap`

The target, dynamics force, and observation noise are generated independently.
Their empirical RMS is recorded. Dynamics forces enter the state transition;
measurement noise enters only the task observation. Hidden truth is retained
for evaluation and is not passed to the actor.

The primary model-selection objective is episode return. Supporting metrics
are tracking RMSE, goal-tracking RMSE, persistent goal-error rate, action RMS,
saturation rate, physical window durations, and response-normalized periods.
No action LF/HF responsibility metric is a mainline success criterion.

## Interpretation Gates

- `flat_multiscale > flat_history` supports a representation benefit.
- `hrl_history > flat_history` supports a hierarchy benefit.
- `hrl_multiscale` exceeding both matched controls supports an interaction
  between hierarchy and multiscale representation.
- If `flat_multiscale` explains all gains, the result is multiscale RL, not
  evidence that hierarchy is required.
- Harm under `band_swap` is a claim boundary, not a gate to tune away.
- Software smoke tests do not count as performance evidence.

## Stage-2 Boundary

The new MuJoCo adapter parses Gymnasium-Robotics dictionary observations into
physical state, achieved goal, and desired goal; resolves control `dt` for
PointMaze and AntMaze; and decodes upper actions into relative state-space
subgoals. PointMaze must first show that ordinary goal-conditioned HRL learns
the task. Multiscale enhancement is evaluated only after that baseline works.
AntMaze follows PointMaze.

The frozen Stage-2 V1 development gate uses `PointMaze_UMaze-v3`, a 300-step
horizon, eight independent optimizer roots, and matched primitive interaction
budgets. Checkpoints are selected on disjoint validation seeds by mean success
rate first and dense return second. The ordinary-HRL gate is supported only if
the lower bound of the root-level 95% success-rate interval is at least 0.50.
Flat-versus-HRL paired effects are reported but do not replace that absolute
learning requirement. See
`freq_hrl_pointmaze_stage2_protocol_2026-09-19.md`.

## Frozen Negative Result

MuJoCo v25 is a failed development result. Its 48 cells and 1,920 evaluation
episodes did not pass the preregistered validity, reward, correction, or Hopper
correction gates. It used the old finite-iteration projector and cannot be
retroactively repaired by the later optimized solver. The action-sample arm
does not advance to confirmation.

## Current Evidence Status

The new mainline has passed interface, causal-prefix, equal-information,
physical-time, parameter-budget, PointMaze-adapter, and learned PPO/SMDP smoke
tests. The frozen Stage-1 V1 development matrix then completed 160 unique cells,
eight independent optimizer roots per method-scenario cell, and 1,280 held-out
evaluation episodes. Its mainline HRL increment is **not supported**:
`hrl_multiscale` was contradicted against `flat_multiscale` in clean and
fast-force conditions, while the flat representation contrast was
inconclusive throughout. See
`freq_hrl_multiscale_goal_stage1_v1_result_2026-09-19.md`.

This negative result closes the identifiable point-mass task as a source of a
positive mainline claim. The next gate is ordinary goal-conditioned HRL on
PointMaze. Multiscale enhancement is not admitted there until the hierarchy
baseline itself learns the task.

The frozen PointMaze Stage-2 V1 development campaign subsequently completed
all 16 cells and 128 held-out episodes. The ordinary-HRL gate was **not
supported**: hierarchical success was 0.359 with root-level 95% CI [0.197,
0.522], whose lower endpoint missed the registered 0.50 threshold. The paired
success contrast against flat PPO was inconclusive. Post-hoc code inspection
identified lower-level credit crossing waypoint boundaries and a mismatch with
the existing progress-reward contract; those defects must be repaired and
tested in a new development protocol before multiscale mechanisms are admitted.
See `freq_hrl_pointmaze_stage2_v1_result_2026-09-19.md`.
