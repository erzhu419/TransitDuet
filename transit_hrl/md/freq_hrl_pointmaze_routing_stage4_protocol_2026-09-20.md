# PointMaze Stage-4 Frequency-Routing Attribution Protocol

Date: 2026-09-20

## Motivation

Stage-3 V2 found positive `hrl_multiscale - hrl_history` success effects under
both primary stresses, but did not support either hierarchy-by-multiscale
interaction or superiority to the causal-filter control. Flat multiscale PPO
was significantly better under observation noise. The result therefore does
not show that assigning different frequency bands to different hierarchy
levels caused the HRL gain.

Stage 4 isolates that assignment inside one goal-conditioned HRL architecture.
It is an attribution experiment, not an attempt to add seeds to the failed V2
gate.

## Methods

All five methods use the same SMDP-PPO hierarchy, upper waypoint semantics,
lower actuator controller, 32-step causal history, current physical feedback,
training budget, and trainable-parameter budget.

1. `hrl_history`: raw causal history goes to both levels.
2. `hrl_causal_filter`: causally filtered history goes to both levels.
3. `hrl_multiscale_all`: every causal Haar band goes to both levels.
4. `hrl_multiscale_routed`: slow+mid goes to the upper level and mid+high to
   the lower level.
5. `hrl_multiscale_swapped`: mid+high goes to the upper level and slow+mid to
   the lower level.

The routed and swapped arms have identical state dimensions and model
capacity. The all-band arm controls for Haar representation without selective
routing. The filter arm controls for ordinary causal smoothing. Both hierarchy
levels retain the full current actor-visible physical state in every arm, and
the lower level never receives the final task goal.

## Scenarios

- `clean`: no exogenous stress.
- `fast_observation_noise`: position standard deviation 0.06 and velocity
  standard deviation 0.08, with no action disturbance.
- `slow_drift_fast_action`: AR action drift with standard deviation 0.12 and
  1.0-second time constant plus independent fast action noise with standard
  deviation 0.08, with no observation noise.

The two stress families remain separate. Disturbance truth is evaluation-only,
and every exogenous sequence is paired by held-out episode seed across methods.

## Training Contract

- Environment: `PointMaze_UMaze-v3`.
- Fixed 300-step continuing-task horizon and official dense reward.
- Upper period: 0.25 seconds (25 primitive steps).
- History: 0.32 seconds (32 steps); fast boundary: 0.04 seconds.
- 768 PPO updates with four training paths per update.
- Eight checkpoint evaluations at 96-update intervals.
- 16 disjoint selection and 16 held-out evaluation paths per root.
- Eight fresh optimizer roots and fresh role seeds.
- 120 formal cells and 1,920 held-out evaluation episodes.
- Projector, promotion, leakage loss, responsibility gauge, and projection
  consistency disabled.

## Analysis and Gate

The optimizer root is the statistical unit. Held-out episodes are averaged
within root before two-sided 95% t intervals. Success is primary; dense return
and final distance are supportive.

Selective routing is supported only if all conditions hold:

1. `routed - all` success has a CI lower endpoint above zero in both primary
   stresses;
2. `routed - swapped` success has a CI lower endpoint above zero in both
   primary stresses;
3. `routed - causal filter` success has a CI lower endpoint above zero under
   observation noise;
4. clean `routed - all` success has a CI lower endpoint of at least -0.10.

This gate separates correct assignment from generic Haar conditioning,
information compression, and smoothing. A successful Stage 4 would still not
overturn the Stage-3 finding that flat multiscale PPO was better under
observation noise; it would establish only a selective-routing mechanism
inside HRL. A valid preflight is software evidence only.

