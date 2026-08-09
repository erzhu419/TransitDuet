# MuJoCo v14.10 Deployment-Aligned Preflight Outcome

Date: 2026-08-09

## Audited run

- Run: `mujoco_v14_10_deployment_aligned_preflight_20260809_r1`
- Analysis: `mujoco_v14_10_deployment_aligned_preflight_analysis_20260809_r1`
- Scheduler cells: `t79296..t79306`
- Environment: `HalfCheetah-v5`
- Optimizer seed: `2706173366`
- Frozen algorithm revision: `1704ce3f5f867ab493899d424f1398557cc4a625`
- Frozen source manifest: `dbc8cc11399a4baad5cb7c1231860c3911293398f553cd2426670fd08636ea80`
- Analysis input SHA-256: `ccca3d669b1e8915206d4eb666e903bb504deb05be9e7e06c868642aa91422b0`
- Evidence role: one-optimizer-seed mechanism preflight, no confidence interval

All 11 cells completed on `node004`. The source-bound merge validated every
checkpoint hash, source identity, anchor dependency, paired selector payload,
training schedule, and 40-row held-out evaluation grid. The anchor used 428 MB
peak RAM and 455.6 seconds. Nine continuations used approximately 233 seconds;
the paired zero-dual comparator used 332.2 seconds. Continuation peak RAM was
406--425 MB. No `jtl110cpu` record or artifact contributed to this run.

## Calibration result

The function-preserving projection calibration passed. Its paired upper,
lower, and combined actor RMS differences were exactly zero. Reward and all
three trace hashes matched pathwise in all five disturbance modes.

Across those modes, the minimum registered lower-frequency reduction was
68.5--71.5%, and effective upper-HF power fell by 39.0--39.6%. This validates
the coordinate transform, not learned separation.

## Learned result

The preflight decision was `do_not_expand`. All seven learned arms selected
iteration `-1`, the registered initial-checkpoint fallback. Their held-out
rewards and frequency metrics therefore exactly matched the paired comparator;
none changed actor tensors or executed actions, and none supplied a learned
candidate.

This fallback was not caused by an inactive mechanism. Every enabled lower
projection was attempted on 31 or 32 updates and accepted on 31 or 32. Enabled
upper projections were accepted on 21--32 updates. Their mean accepted
within-update power reductions were:

- upper: approximately 0.30--0.87%;
- lower: approximately 0.25--0.77%.

The corrections were real but too small relative to the registered 5% or 10%
paired target and to the following PPO update. On training batches, median
post-correction lower normalized violations remained approximately 1.16--2.46
for representative joint arms, with maxima of 1.65--4.13. On independent
checkpoint-selection paths, the best trained states still had worst normalized
violations of approximately 2.26--13.87, compared with only 0.0526 for the
safe initial fallback.

## Diagnosis

v14.10 fixed the v14.9 sampled-action versus deployment-action mismatch, but
it exposed a second optimization defect: one small reward-guarded correction
after each PPO update is not a projection onto the paired frequency-feasible
set. It slightly reduces current-batch power but cannot prevent the next reward
update from leaving that set, and it does not generalize to the registered
selection paths.

The next core revision must implement an iterative projection with:

1. multiple constraint corrections per PPO iteration;
2. a cumulative reward-surrogate budget relative to the pre-projection policy,
   rather than a fresh tolerance for every correction;
3. explicit stopping when normalized deployment-frequency excess reaches a
   registered tolerance;
4. diagnostics for requested, attempted, accepted, and target-reaching steps;
5. a new preflight before any multi-seed expansion.

## Claim boundary

Allowed: v14.10 demonstrates that deterministic deployment-frequency gradients
are active, reward-guarded steps can reduce same-batch power, unsafe trained
checkpoints are correctly rejected, and projection-only responsibility routing
is function preserving.

Forbidden: v14.10 supports learned frequency separation, reward improvement,
no-tradeoff behavior, cross-task generality, or any confirmatory manuscript
claim.
