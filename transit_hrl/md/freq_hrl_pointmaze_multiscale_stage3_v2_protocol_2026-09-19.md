# PointMaze Multiscale Goal-Control Stage-3 V2 Protocol

Date: 2026-09-19

## Reason for V2

Stage-3 V1 passed software preflight but failed design review before producing
performance evidence. Its multiscale upper policy saw the desired goal and
slow history coefficients but not the current physical observation. That
reintroduced the exact failure mode identified by the GPT6 diagnosis: frequency
routing had replaced necessary endogenous feedback instead of augmenting it.

V2 restores full current actor-visible physical feedback to both hierarchy
levels, separates observation, execution, and persistent-mode stresses, and
adds an ordinary causal-filter baseline. V1 tasks `t95604`-`t95667` were all
cancelled; V2 uses fresh seed roles and is not a continuation of V1 training.

## Methods

The registered two-by-two factorial remains:

| Control structure | Raw causal history | Causal Haar representation |
|---|---|---|
| Flat | `flat_history` | `flat_multiscale` |
| Goal-conditioned HRL | `hrl_history` | `hrl_multiscale` |

`flat_causal_filter` is an auxiliary fifth method. It receives the same fixed
trailing samples after a causal EMA and is parameter matched to the same flat
reference. It rules out the explanation that any observation-noise gain is
only ordinary smoothing.

All methods use a 0.32-second, 32-step trailing window of the four-dimensional
actor-visible PointMaze physical observation. Raw, filtered, and orthonormal
Haar flat states each contain 128 transformed history scalars plus the same
current physical state and task-goal error.

The HRL states are:

- raw upper: current physical state, final-goal error, and raw history;
- multiscale upper: current physical state, final-goal error, and slow+mid
  Haar coefficients;
- raw lower: current physical state, waypoint error, and raw history;
- multiscale lower: current physical state, waypoint error, and mid+high Haar
  coefficients.

The mid band is shared. No hierarchy level loses current physical feedback.
The lower level never observes the final task goal. The slow-energy envelope is
recorded by the encoder but is not an actor input in this factorial.

## Independent Causal Stresses

The four registered scenarios are:

1. `clean`: no exogenous stress.
2. `fast_observation_noise`: position noise standard deviation 0.06 and
   velocity noise standard deviation 0.08; no action disturbance.
3. `slow_drift_fast_action`: stationary AR action drift with standard deviation
   0.12 and 1.0-second time constant, plus independent fast action noise with
   standard deviation 0.08; no observation noise.
4. `persistent_action_shift`: a hidden action bias of magnitude 0.18 begins at
   35% of the episode and persists; no observation or other action noise.

The observation, action-noise, and persistent-mode channels are separate by
construction and separately reported. Current disturbance truth, measurement
error, and mode truth never enter actor state or learning reward. Environment
reward, success, and goal distance use the true simulator state. Exogenous
sequences are deterministic by episode seed and paired across methods.

## Task and Training Contract

- Environment: `PointMaze_UMaze-v3`.
- Official dense `exp(-goal_distance)` reward.
- Fixed 300-step horizon, `continuing_task=True`, unchanged target.
- Upper period: 0.25 seconds (25 primitive steps).
- Upper action: relative XY waypoint, maximum delta 0.75 per coordinate.
- Lower action: physical acceleration only.
- Lower reward: actor-visible waypoint progress minus 0.005 requested-action
  cost.
- Lower GAE boundary: waypoint change or episode end.
- Upper task credit: true environment reward with SMDP duration accounting.
- 768 PPO updates, four training paths per update, and eight checkpoint
  evaluations.
- 16 disjoint selection and 16 held-out evaluation paths per root.
- Eight independent optimizer roots.
- 160 formal cells and 2,560 held-out evaluation episodes.

Every method is matched to the trainable parameter count of a 128-hidden-unit
`flat_history` actor-critic. Projector, promotion, leakage loss, responsibility
gauge, and projection consistency remain disabled.

## Analysis and Claim Gate

The optimizer root is the statistical unit. Held-out episodes are averaged
within root before two-sided 95% t intervals. Success is primary; dense return
and final distance are supportive.

For every scenario report:

- `flat_multiscale - flat_history`;
- `hrl_history - flat_history`;
- `hrl_multiscale - hrl_history`;
- `hrl_multiscale - flat_multiscale`;
- the hierarchy-by-multiscale factorial interaction;
- `flat_multiscale - flat_causal_filter`;
- `hrl_multiscale - flat_causal_filter`.

The cross-stress Freq-HRL gate is supported only if:

1. `hrl_multiscale - hrl_history` has a success CI lower endpoint above zero
   in both `fast_observation_noise` and `slow_drift_fast_action`;
2. the success factorial interaction has a CI lower endpoint above zero in
   both primary stress scenarios;
3. `hrl_multiscale - flat_causal_filter` has a success CI lower endpoint above
   zero under `fast_observation_noise`;
4. the clean `hrl_multiscale - hrl_history` success CI lower endpoint is at
   least -0.10.

`persistent_action_shift` is a registered secondary boundary and is always
reported, but it is not silently substituted for a failed primary gate. A
valid V2 preflight is software evidence only.
