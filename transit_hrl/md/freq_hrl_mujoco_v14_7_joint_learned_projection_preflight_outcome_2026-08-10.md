# MuJoCo v14.7 Joint Learned-Projection Preflight Outcome

Date: 2026-08-10

## Frozen preflight

- Run: `mujoco_v14_7_joint_learned_projection_preflight_20260809_r1`
- Scheduler tasks: `t78869`, `t78870`, `t78871`, and `t78876..t78879`
- Frozen algorithm revision: `f31811c662411c0cb58db890f950a75d660470f2`
- Frozen source manifest: `a253feb248549b8c4f1a20c858ffc8c2bd3088a7a0cfc9645a1a630cc0691459`
- Environment: `HalfCheetah-v5`
- Optimizer replicates: one development seed (`2341827529`)
- Held-out paths per continuation: 5 disturbance modes x 8 evaluation seeds
- Evidence role: development preflight only

All seven registered continuations completed naturally through scheduleurm. The
run is too small for optimizer-replicate confidence intervals and was accessed
before any full cross-environment screen.

## Mechanism checks

The zero-dual `s=0.5` projection calibration and zero-strength control selected
identical upper and lower actors (combined actor RMS difference `0.0`). Their
episode returns and reward, latent-policy, and executed-action trace hashes were
identical on all 40 held-out paths. This is pathwise evidence for the
function-preserving coordinate transform, not learned frequency separation.

Every primal-dual arm changed the actor (combined RMS difference
`0.01208..0.01361`) and changed all 40 executed-action traces relative to the
zero-strength control. The learned branch therefore executed rather than
silently reproducing the anchor.

## Invalid learned comparison

The apparently strongest arm, `joint_s050_pd_u010_l010`, selected iteration
`19`, changed all held-out action traces, and had higher path-mean return than
the zero-strength control in each of the five disturbance modes. It also had
lower path-mean routed, raw, and latent lower-frequency diagnostics and lower
upper high-frequency power in each mode. These are descriptive single-seed
outcomes only.

The comparison is not selection-valid. The zero-strength comparator selected
iteration `31` with `checkpoint_score_mode=mean_reward`, whereas every learned
arm used `checkpoint_score_mode=behavior_robust`. Thus optimization and
checkpoint selection changed together. A reward or separation contrast cannot
be attributed to the learned projection objective. In addition, hashes of the
full checkpoint include cost-critic state and cannot establish actor identity;
actor-only tensor comparisons are required.

The v14.8 successor consequently froze a projection- and selector-matched
zero-dual comparator and added pre-routing latent endpoints before reading its
own outcomes.

## Decision and claim boundary

Decision: `comparator_confounded`. No v14.7 arm is eligible for algorithm
selection or expansion.

Allowed: v14.7 validates the pathwise projection calibration, confirms that the
learned branch changes actor behavior, and identifies checkpoint-selector and
full-checkpoint-hash confounds that must be removed.

Forbidden: v14.7 supports reward improvement, learned frequency separation,
no-tradeoff behavior, cross-task generality, statistical evidence,
confirmatory evidence, or a selected submission algorithm.
