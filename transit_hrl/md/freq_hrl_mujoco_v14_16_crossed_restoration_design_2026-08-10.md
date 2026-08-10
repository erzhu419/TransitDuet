# Freq-HRL MuJoCo v14.16 Crossed Restoration Design

Date: 2026-08-10

## Evidence boundary

This is a development mechanism screen, not confirmation evidence. The v14.16
design was created only after the complete v14.15 r2 multiseed primary family
failed. Any mechanism selected here must be frozen again, evaluated with more
optimizer seeds, and then tested on a fresh confirmation namespace.

## Why v14.15 was rejected

The valid v14.15 r2 screen completed all 450 cells, but only 8 of 45
environment-by-optimizer candidate cells passed the complete gate. The
one-sided Wilson lower bound was 0.103187, below the registered 0.70 gate.
HalfCheetah passed 0/15 cells, Hopper 7/15, and Walker2d 1/15.

The failure was not simply insufficient projection strength:

1. Several training trajectories reduced guard violations to zero, but the
   disjoint checkpoint selector restored the initial anchor.
2. The differentiable correction optimized only the maximum normalized group
   excess, so nonmaximal violating groups received no direct correction.
3. Frozen-state replay reused iteration-zero training paths and therefore did
   not test independent state coverage.
4. Closed-loop guard and checkpoint constraints averaged guard roots within a
   disturbance mode, allowing a harmed path to be hidden by another path.
5. Reward-PPO actor updates continued during infeasible restoration and could
   undo or redirect frequency corrections before the outer guard transaction.

## Algorithm changes

v14.16 adds four independently testable mechanisms while retaining the same
shared upper/lower SMDP PPO core:

1. `violation_l2`: a group-count-normalized L2 norm over all positive
   normalized frequency violations. Every active violating group contributes a
   differentiable correction; `worst_group` remains available as the causal
   control.
2. Pathwise robustness: checkpoint selection and closed-loop restoration apply
   reward floors and five frequency endpoints to each independent rollout path,
   rather than to disturbance-mode means.
3. Restoration actor freeze: while the current closed-loop snapshot is
   frequency-infeasible, upper/lower reward-PPO actor updates are disabled.
   Critics continue training and the differentiable frequency projection can
   still update actors. Ordinary actor training resumes in feasible
   maintenance.
4. Crossed frozen-state replay: independent replay roots are crossed with all
   training disturbance modes. These paths are disjoint from training,
   selection, closed-loop guard, and held-out evaluation roles.

The MuJoCo core now also requires an explicit v14.16 protocol selection for
this screen. This allows controls and candidates to share the same checkpoint
format and source contract without forcing new mechanisms on the controls.

## Frozen execution identity

- Algorithm revision:
  `ffc60268ece9f4acd459bb61726fadcbf4a36d8b`
- Source manifest SHA-256:
  `b79dcf44898b7128a29d9f647f9e8577de74a45add1e9945159655652435cca4`
- Core protocol:
  `freq_hrl_mujoco_shared_core_v14_16_crossed_pathwise_restoration`
- Development protocol:
  `mujoco_v14_16_crossed_restoration_mechanism_screen_v1`

## Cumulative causal matrix

All learned arms retain the v14.15 selected budgets, relative 5% targets,
groupwise constraints, PPO trust region, closed-loop restoration filter,
reward tolerance 0.005, eight inner projection steps, eight outer backtracks,
and funnel multiplier 3.

| Arm | L2 | pathwise | actor freeze | crossed replay |
|---|---:|---:|---:|---:|
| `worst_mode_trainreplay` | no | no | no | no |
| `l2_mode_trainreplay` | yes | no | no | no |
| `l2_path_trainreplay` | yes | yes | no | no |
| `l2_path_freeze_trainreplay` | yes | yes | yes | no |
| `l2_path_freeze_crossreplay` | yes | yes | yes | yes |

Three additional controls are retained:

- `mean_s000_control`: zero router strength and no frequency objective.
- `mean_s050_projection_calibration`: router strength 0.5 without learning.
- `paired_s050_d000_control`: matched router and paired selector without a
  learned frequency objective.

The primary arm is pre-registered as
`l2_path_freeze_crossreplay`; ranking the other arms is diagnostic and cannot
retroactively redefine a confirmation candidate.

## Experiment scale

- Environments: HalfCheetah-v5, Hopper-v5, Walker2d-v5.
- Optimizer seeds: 3 fresh development replicates.
- Per replicate/environment: 1 shared anchor plus 8 continuation arms.
- Total tasks: 81.
- Each task requests one physical CPU core and 768 MB RAM.
- Eligible nodes: `node001` through `node006`.
- `require_node=None`; tasks are dynamically placed and may reroute after node
  failure.
- No Slurm path and no `jtl110cpu` execution.

Training uses four conditions (`standard`, `low_frequency`, `high_frequency`,
`mixed`); held-out evaluation adds `ood_chirp`. Pathwise arms use 16 guard
paths and therefore 96 closed-loop constraints per snapshot. The crossed replay
arm uses 16 independent frozen-state paths. Continuations run 32 iterations;
anchors retain 64 iterations.

## Analysis contract

The optimizer seed is the statistical unit. Held-out paths are paired
observations and are never counted as independent replicates.

Each learned arm is compared with the matched comparator on identical
environment, disturbance, and held-out seed paths. Reported effects are:

- normalized episode-return difference, with a -2% noninferiority threshold;
- log baseline/candidate ratio for each of five frequency endpoints, with a
  threshold of `-log(0.95)`.

The development analyzer reports every path, environment-by-optimizer cell,
and cumulative arm. Three seeds are intentionally insufficient for a paper
claim or a reliable confidence interval. A positive result only authorizes a
larger frozen multiseed development screen; it does not authorize manuscript
language claiming effectiveness.

## Verification completed before dispatch

- New and legacy mechanism tests: 76 passed.
- Shared RL/MuJoCo/launcher regression: 153 passed and 6 subtests passed.
- End-to-end v14.16 checkpoint write/load, crossed replay, pathwise guard, and
  restoration freeze smoke: passed.
- Legacy v14.15 launcher/spec tests remain passing.
