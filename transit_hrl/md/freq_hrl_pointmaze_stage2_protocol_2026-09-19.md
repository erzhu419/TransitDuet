# PointMaze Goal-Control Stage-2 V1 Protocol

Date: 2026-09-19

## Evidence Role

This is a development gate for the reoriented Freq-HRL mainline. It asks one
question before multiscale information is introduced:

> Can the ordinary goal-conditioned hierarchy learn a genuine waypoint task?

Passing this gate permits a later multiscale factorial experiment. It does not
by itself support a positive Freq-HRL or frequency-separation claim.

## Task and Control Semantics

- Environment: `PointMaze_UMaze-v3` from Gymnasium-Robotics.
- Reward: environment dense reward.
- Horizon: 300 primitive steps.
- Physical control interval: read from the environment and expected to be
  0.01 seconds.
- Episode semantics: `continuing_task=False`, `reset_target=False`.
- Primary endpoint: environment success rate.
- Supporting endpoints: episode return, final/minimum goal distance, path
  length, action RMS, saturation, and subgoal tracking.

The hierarchical upper policy observes the physical state and final-goal error
and emits a relative XY waypoint every 0.25 seconds (25 primitive steps). The
waypoint displacement is bounded to 0.75 world units per coordinate and then
clipped to the exposed maze bounds. The lower policy observes the physical
state and current waypoint error, but not the final task goal, and is the only
policy that emits a physical acceleration action.

This is a real goal/action hierarchy. No upper torque anchor or additive action
decomposition is used.

## Compared Methods

1. `flat_goal_ppo`: joint PPO observes physical state plus final-goal error and
   emits the primitive action every environment step.
2. `hrl_goal_ppo`: SMDP PPO learns an upper waypoint policy and a lower
   goal-conditioned primitive controller.

Both methods receive the same primitive interaction budget. Their combined
trainable actor-critic parameter counts are matched to the flat network with
reference hidden width 128. The parameter ratio is recorded in every result.

Projector, promotion, leakage loss, responsibility gauge, multiscale features,
and projection-consistency fitting are disabled. They cannot be credited for
this gate.

## Optimization and Selection

- Independent optimizer roots: 8.
- Training rollouts per update: 4.
- Training iterations: 768.
- Primitive training transitions per method/root: at most 921,600.
- Validation paths per checkpoint: 4 disjoint seeds.
- Held-out evaluation paths per method/root: 8 disjoint seeds.
- Checkpoint evaluation interval: 16 updates, plus the final update.
- Learning rate: `3e-4`.
- PPO epochs: 4.
- Evidence stage: development, not confirmation.

Train, validation, and held-out evaluation seed roles are disjoint within every
optimizer root and are not reused across optimizer roots. The two methods use
paired role seeds and optimizer roots.

Checkpoint selection is state-aligned and lexicographic:

1. maximize validation mean success rate;
2. break ties by validation mean dense episode return.

The initial untrained checkpoint is ineligible. This aligns selection with the
primary endpoint while retaining dense reward as a learning-sensitive
tie-breaker before success becomes common.

## Registered Analysis

Evaluation episodes are averaged within optimizer root. The optimizer root,
not an episode, is the statistical unit. Two-sided 95% t intervals are reported
for each method's root-level success rate, return, and final distance. Paired
root-level HRL-minus-flat intervals are reported for the same endpoints, with
distance sign-adjusted so positive always means improvement.

The ordinary-HRL learning gate is **supported** only when the lower endpoint of
the hierarchical method's 95% success-rate interval is at least 0.50. Otherwise
it is **not supported**, and multiscale enhancement remains blocked.

The paired flat comparison is diagnostic. Ordinary HRL can pass the learning
gate without outperforming flat PPO; that outcome would establish a working
hierarchical substrate, not a hierarchy advantage.

## Execution Boundary

The formal campaign runs as one independent scheduler task per
`(method, optimizer_root)` cell on `node001` through `node006`, with dynamic
placement and one CPU core per cell. The preflight uses one root, both methods,
two updates, and one seed per role. Only `result.json` and logs are required;
checkpoints are not synchronized.

Software smoke and scheduler preflight establish only execution validity. The
formal development matrix is required for the registered learning decision.
