# PointMaze Goal-Control Stage-2 V2 Protocol

Date: 2026-09-19

## Purpose

V1 produced a valid negative ordinary-HRL gate. Post-hoc inspection identified
a concrete lower-level credit defect: GAE crossed waypoint changes, and the
rollout used cumulative negative waypoint distance instead of the existing
progress-based intrinsic-reward contract. V2 tests the corrected ordinary HRL
substrate before any multiscale mechanism is admitted.

This remains development evidence. Passing V2 permits a later multiscale
factorial experiment; it is not itself a positive Freq-HRL claim.

## Frozen Algorithm Change

The task, hierarchy, model capacity, primitive interaction budget, and primary
endpoint remain as in V1. The only learning-path changes are:

1. lower reward is waypoint-distance progress minus `0.005` mean squared
   physical action;
2. lower GAE terminates at each scheduled waypoint change or episode end;
3. upper SMDP transitions remain open across lower option boundaries and still
   use environment task reward with `gamma^duration` bootstrap.

Projector, promotion, leakage loss, responsibility gauge, multiscale features,
and projection consistency remain disabled.

## Task and Budget

- Environment: `PointMaze_UMaze-v3`, dense reward.
- Horizon: 300 primitive steps.
- Upper period: 0.25 seconds, resolved as 25 primitive steps.
- Maximum relative waypoint delta: 0.75 world units per coordinate.
- Training: 768 updates, four rollout paths per update.
- Methods: capacity-matched `flat_goal_ppo` and `hrl_goal_ppo`.
- Independent optimizer roots: 8.
- Held-out evaluation paths per method/root: 16.
- Evidence unit: optimizer-root mean, not episode.

All optimizer, training, selection, and evaluation seeds are new and disjoint
from V1. Methods remain paired within optimizer root.

## Checkpoint Selection

V1 evaluated 48 checkpoints on four validation paths, which allowed excessive
reuse of a noisy selection set. V2 evaluates eight checkpoints at 96-update
intervals on 16 disjoint validation paths. Selection remains lexicographic:

1. maximize mean validation success;
2. break ties by mean dense episode return.

The untrained checkpoint remains ineligible. Held-out evaluation seeds are
never used for checkpoint selection.

## Registered Decision

The optimizer root is the statistical unit. Report two-sided 95% t intervals
for success, episode return, and final distance, plus paired HRL-minus-flat
intervals.

The ordinary-HRL gate is supported only if the lower endpoint of the HRL
root-level 95% success interval is at least 0.50. Otherwise it is not supported
and multiscale enhancement remains blocked. The paired flat comparison remains
diagnostic rather than a substitute for the absolute learning gate.

The preflight validates execution, runtime provenance, option boundaries,
finite metrics, and compact result serialization. It is not performance
evidence.
