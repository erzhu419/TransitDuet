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

Stage-2 V2 is frozen as a clean repair test. It restores progress-based lower
reward, terminates lower GAE at waypoint changes without terminating the upper
SMDP transition, uses fresh seed roles, and reduces noisy checkpoint reuse.
The task, interaction budget, capacity match, and absolute success gate remain
unchanged. See `freq_hrl_pointmaze_stage2_v2_protocol_2026-09-19.md`.

Stage-2 V2 then completed 16/16 cells and 256 held-out episodes but again did
not support the ordinary-HRL gate. Both methods had 0.297 mean success, and the
HRL 95% CI was [0.152, 0.441]. The option-credit repair sharply reduced lower
value loss, but post-hoc analysis found that successful episodes accumulated
about half the positive dense return of failed full-horizon episodes because
success terminated the episode. The next protocol must align the common PPO
training reward with success before changing the hierarchy again. See
`freq_hrl_pointmaze_stage2_v2_result_2026-09-19.md`.

Stage-2 V3 is frozen as the common reward-semantics repair. It uses the
official fixed-horizon PointMaze mode (`continuing_task=True`, unchanged target)
so reaching the target no longer removes future positive dense reward. V2
option credit, budgets, capacity matching, selection schedule, and the absolute
success gate remain unchanged; all seed roles are fresh. See
`freq_hrl_pointmaze_stage2_v3_protocol_2026-09-19.md`.

Stage-2 V3 completed 16/16 cells and 256 fixed-horizon held-out episodes. The
ordinary-HRL gate is **supported**: hierarchical success was 0.789 with
root-level 95% CI [0.710, 0.868]. Paired dense return and final-distance
improvements over flat PPO were supported, while the paired success difference
was inconclusive. Reward-success association became positive and lower critic
stability held. This admits a separately registered PointMaze multiscale
factorial experiment; it does not itself support a Freq-HRL claim. See
`freq_hrl_pointmaze_stage2_v3_result_2026-09-19.md`.

## Stage-3 V2 Registered Factorial

The admitted PointMaze experiment uses the same four-grid design used for
mechanism identification: flat/HRL crossed with raw history/multiscale, plus a
flat causal-filter control. Both hierarchy levels retain full current physical
feedback. Multiscale history augments that feedback: slow+mid is routed upward
and mid+high downward. Observation noise, continuous slow/fast action stress,
and persistent action-mode shift are separate causal scenarios.

A cross-stress Freq-HRL claim requires positive success increments and
factorial interactions in both primary stress families, superiority to causal
filtering under observation noise, and clean noninferiority. See
`freq_hrl_pointmaze_multiscale_stage3_v2_protocol_2026-09-19.md`.

The eight-cell V1 preflight completed on scheduler tasks t95327-t95334, but a
subsequent design review found that its upper multiscale policy had lost
current physical feedback. All 64 V1 development tasks t95604-t95667 were
cancelled. V1 supplies no performance evidence and does not authorize V2;
V2 requires fresh preflight and seeds. See
`freq_hrl_pointmaze_multiscale_stage3_preflight_2026-09-19.md`.

The corrected 20-cell V2 preflight then completed as tasks t95931-t95950. It
passed the registered state, capacity, seed, option-boundary, runtime, and
independent-stress checks and authorizes the frozen 160-cell development
matrix. It remains software evidence only. See
`freq_hrl_pointmaze_multiscale_stage3_v2_preflight_2026-09-19.md`.

The full V2 development matrix subsequently completed all 160 cells and 2,560
held-out episodes. Frequency routing improved the HRL history baseline under
both primary stresses, but the Freq-HRL-specific gate was not supported:
factorial interactions and the causal-filter comparison were inconclusive,
and flat multiscale PPO significantly exceeded HRL multiscale under
observation noise. The result is a representation-positive but
hierarchy-specific-negative boundary. See
`freq_hrl_pointmaze_multiscale_stage3_v2_result_2026-09-20.md`.

Stage 4 therefore moves to within-HRL attribution rather than adding roots to
the failed V2 gate. It compares intended routing against all-band,
frequency-swapped, causal-filter, and raw-history controls with fresh role
seeds. The gate requires correct routing to beat both all-band and swapped
routing under each primary stress, which distinguishes assignment from generic
multiscale representation or compression. See
`freq_hrl_pointmaze_routing_stage4_protocol_2026-09-20.md`.

The 15-cell Stage-4 preflight completed as tasks t97442-t97456 and passed the
registered software, update, capacity, seed, state-feedback, timing, runtime,
and stress-channel checks. It is software evidence only and authorizes the
unchanged 120-cell development matrix. See
`freq_hrl_pointmaze_routing_stage4_preflight_2026-09-20.md`.

The full 120-cell Stage-4 matrix then rejected the fixed intended assignment.
Routed improved over raw history in clean and observation-noise conditions,
but did not significantly beat all-band HRL and was significantly worse than
swapped routing in clean and both stresses. A post-hoc design audit found that
the unequal slow/high coefficient counts also reversed upper/lower input-layer
parameter allocation: both arms had 67,502 total parameters, but their
25,479/42,023 per-level split was exchanged. V1 therefore does not isolate
band semantics. The immediate repair is a fresh equal-shape masked-routing
protocol; exogenous/endogenous separation remains the next environment-level
test. See
`freq_hrl_pointmaze_routing_stage4_result_2026-09-20.md`.

Stage-4 V2 freezes that repair at algorithm revision
`f9ab0b4a532d1bc0c31466b24d4dcbc70634e585`. Both levels always receive fixed
134-dimensional states; routed and swapped arms differ only in which Haar
coefficient slots are zeroed. Network shapes, per-level parameters, and initial
weights are identical within each fresh optimizer root. The V1 scenarios,
training budget, endpoints, and conjunctive gate remain unchanged. See
`freq_hrl_pointmaze_routing_stage4_v2_protocol_2026-09-20.md`.

The 15-cell V2 preflight completed as tasks t99438-t99452. It passed exact-mask,
equal-shape, identical-initialization, learned-update, fresh-seed, timing,
runtime, and stress-channel checks. This remains software evidence only and
authorizes the unchanged 120-cell V2 matrix. See
`freq_hrl_pointmaze_routing_stage4_v2_preflight_2026-09-20.md`.

The 120-cell V2 matrix completed with 1,920 held-out episodes and passed every
execution audit. Routed multiscale history improved over raw history in clean
and both stresses, but the selective assignment gate was not supported:
routed-versus-all was inconclusive in both stresses and swapped routing was
better under action stress. Because V2 cleanly removes the V1 capacity
confound, this closes endogenous PointMaze history as evidence for the central
Freq-HRL routing claim. The next mainline benchmark must provide a separate
actor-visible exogenous stream while leaving endogenous physical feedback
unchanged at both levels. See
`freq_hrl_pointmaze_routing_stage4_v2_result_2026-09-21.md`.

Stage 5 implements that required environment-level separation. Both hierarchy
levels retain current physical state, while a separately buffered external
stream supplies a slow moving target and a measured 0.04-second execution
force whose path is independent of actions. Frequency routing remains disabled
for the substrate gate. The two-cell preflight completed as tasks t99960 and
t99961 on node004 and node006 and passed state, capacity, causal, update,
option-boundary, runtime, and artifact checks. It authorizes the frozen 16-cell
development matrix but is not performance evidence. See
`freq_hrl_pointmaze_exogenous_stage5_protocol_2026-09-21.md` and
`freq_hrl_pointmaze_exogenous_stage5_preflight_2026-09-21.md`.

The full 16-cell Stage-5 V1 matrix then completed with 256 final held-out and
256 paired untrained episodes. HRL final-minus-untrained tracking success and
return were supported, but the absolute success interval was 0.616 [0.424,
0.808], missing the frozen 0.50 lower-bound requirement. One optimizer root did
not learn and one remained on the boundary. The substrate gate is therefore not
supported and external-stream frequency routing remains blocked pending a
fresh optimization-stability repair. See
`freq_hrl_pointmaze_exogenous_stage5_result_2026-09-21.md`.

The registered Stage-5 stability screen reused the two difficult optimizer
initializations but replaced all environment-path seeds. Increasing the
training rollout batch from four to eight raised worst-root held-out success
from 0.442 to 0.955; changing checkpoint rank alone had no effect. This selects
an optimization recipe rather than supplying evidence. A Stage-5 V2 may now
use fresh optimizer and role seeds, with equal eight-rollout budgets for flat
and HRL. See
`freq_hrl_pointmaze_exogenous_stage5_stability_result_2026-09-22.md`.
