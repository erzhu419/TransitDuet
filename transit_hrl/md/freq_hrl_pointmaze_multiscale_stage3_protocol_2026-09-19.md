# PointMaze Multiscale Goal-Control Stage-3 Protocol

Date: 2026-09-19

## Purpose

PointMaze Stage-2 V3 established that the ordinary goal-conditioned hierarchy
learns under aligned fixed-horizon reward semantics. Stage-3 now tests the
actual Freq-HRL hypothesis. It separates a generic multiscale representation
effect from a hierarchy-specific frequency-routing effect instead of comparing
only the proposed method with flat PPO.

This is a development factorial. Its outcome cannot be described as a positive
Freq-HRL result until the registered interaction gate passes.

## Frozen Factorial

The four capacity-matched methods are:

| Control structure | Raw causal history | Causal Haar representation |
|---|---|---|
| Flat | `flat_history` | `flat_multiscale` |
| Goal-conditioned HRL | `hrl_history` | `hrl_multiscale` |

All methods use the same 0.32-second trailing window of actor-visible physical
observations. At PointMaze's 0.01-second control interval this is 32 samples
and 128 scalar history values. The Haar transform is orthonormal and therefore
contains exactly the same samples as raw history.

`flat_multiscale` receives every Haar coefficient. In `hrl_multiscale`, the
upper policy receives the desired goal, slow coefficients, and slow energy;
the lower policy receives current physical feedback, waypoint error, and the
mid/high coefficients. `hrl_history` gives both levels the corresponding raw
history. The lower policy never receives the final task goal.

Each method is matched to the parameter count of a 128-hidden-unit
`flat_history` actor-critic. Primitive interaction counts, optimizer roots,
training paths, checkpoint schedule, selection paths, and held-out paths are
paired across all four methods and both scenarios.

## Causal Scenarios

The registered scenarios are:

1. `clean`: no observation or action disturbance.
2. `mixed_causal_stress`: hidden slow action drift plus hidden fast action and
   observation noise.

The stress parameters are frozen as:

- position observation noise standard deviation: 0.06;
- velocity observation noise standard deviation: 0.08;
- stationary slow action-drift standard deviation: 0.12;
- slow action-drift time constant: 1.0 second;
- fast action-noise standard deviation: 0.08.

The policy acts from the current noisy observation. Its requested action is
then perturbed and clipped before the environment transition. Environment
reward, success, and reported goal distance use the true simulator state.
Neither measurement-error truth nor action-disturbance truth enters any policy
state or learning reward. Independent noise streams are deterministic by
episode seed and paired across methods.

## Unchanged HRL Semantics

- Environment: `PointMaze_UMaze-v3`.
- Reward: official dense `exp(-goal_distance)` reward.
- Episode: fixed 300 steps with `continuing_task=True` and unchanged target.
- Upper period: 0.25 seconds (25 primitive steps).
- Upper action: relative XY waypoint, maximum delta 0.75 per coordinate.
- Lower action: physical acceleration only.
- Lower reward: actor-visible waypoint progress minus `0.005` requested-action
  cost.
- Lower GAE boundary: waypoint change or episode end.
- Upper credit: true environment reward with SMDP duration accounting.

Action-spectrum projection, promotion, leakage loss, responsibility gauge, and
projection consistency remain disabled.

## Training and Statistical Units

- 768 PPO updates per cell.
- Four training paths per update.
- Eight checkpoint evaluations per cell.
- 16 disjoint validation paths per checkpoint.
- 16 held-out evaluation paths per cell.
- Eight independent optimizer roots.
- 64 formal cells and 1,024 held-out evaluation episodes.

The independent optimizer root is the statistical unit. Evaluation episodes
are averaged within root before two-sided 95% t intervals are computed.

## Registered Contrasts and Gate

The primary endpoint is held-out success. Dense return and final goal distance
are supportive endpoints. Report all of:

- `flat_multiscale - flat_history`: generic representation effect;
- `hrl_history - flat_history`: hierarchy effect without frequency routing;
- `hrl_multiscale - hrl_history`: frequency-routing increment;
- `hrl_multiscale - flat_multiscale`: proposed method versus matched flat
  multiscale control;
- `(hrl_multiscale - hrl_history) - (flat_multiscale - flat_history)`:
  hierarchy-by-multiscale interaction.

The Freq-HRL mainline gate is supported only when all three conditions hold:

1. Under `mixed_causal_stress`, the 95% CI lower endpoint for the success
   increment of `hrl_multiscale` over `hrl_history` is above zero.
2. Under `mixed_causal_stress`, the 95% CI lower endpoint for the success
   factorial interaction is above zero.
3. Under `clean`, the 95% CI lower endpoint for the success increment of
   `hrl_multiscale` over `hrl_history` is at least -0.10.

If flat multiscale gains explain the result or the interaction is not
supported, the evidence is for representation learning rather than Freq-HRL.
The preflight validates software, causality, fixed horizons, parameter matching,
and serialization only; it is not performance evidence.
