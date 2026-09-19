# PointMaze Goal-Control Stage-2 V3 Protocol

Date: 2026-09-19

## Purpose

V2 repaired lower option credit but failed the ordinary-HRL learning gate.
Post-hoc analysis then exposed a common objective defect: Gymnasium-Robotics
dense reward is `exp(-goal_distance)`, while episodic success immediately
terminated the trajectory and removed future positive reward. Successful V2
episodes therefore accumulated substantially less training return than failed
full-horizon episodes.

V3 corrects this common task objective before any further hierarchy change.
It remains development evidence and does not admit multiscale mechanisms unless
the unchanged ordinary-HRL gate passes.

## Frozen Reward and Termination Semantics

Both methods use the official PointMaze dense reward with:

- `continuing_task=True`;
- `reset_target=False`;
- a fixed 300-step horizon;
- success defined as reaching the unchanged target at any time.

After reaching the target, the episode continues and the agent can retain high
reward by remaining near it. Earlier success is therefore aligned with larger
return instead of being penalized by removal of future positive rewards. No
method-specific reward or terminal bonus is introduced.

## Unchanged V2 Structure

- Environment: `PointMaze_UMaze-v3`.
- Methods: capacity-matched `flat_goal_ppo` and `hrl_goal_ppo`.
- Upper waypoint period: 0.25 seconds (25 primitive steps).
- Maximum relative waypoint delta: 0.75 per coordinate.
- Lower reward: waypoint-distance progress minus `0.005` mean squared action.
- Lower GAE boundary: waypoint change or episode end.
- Upper credit: environment task reward with `gamma^duration` bootstrap.
- Training: 768 updates and four rollout paths per update.
- Selection: eight checkpoints, 16 validation paths per root.
- Evaluation: 16 held-out paths per method/root.
- Independent optimizer roots: 8.

Projector, promotion, leakage loss, responsibility gauge, multiscale features,
and projection consistency remain disabled. V3 uses optimizer, train,
selection, and evaluation seeds disjoint from V1 and V2.

## Registered Decision

The optimizer root remains the statistical unit. Report two-sided 95% t
intervals for success, dense return, and final distance, plus paired
HRL-minus-flat intervals.

The ordinary-HRL learning gate is supported only if the lower endpoint of the
HRL root-level 95% success interval is at least 0.50. Otherwise it is not
supported and multiscale enhancement remains blocked. Flat comparison and the
success-return association are diagnostics, not substitutes for this gate.

The preflight validates fixed-horizon semantics, runtime provenance, option
boundaries, compact serialization, and finite metrics. It is not performance
evidence.
