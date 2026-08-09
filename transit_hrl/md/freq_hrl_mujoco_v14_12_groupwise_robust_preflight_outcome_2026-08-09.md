# MuJoCo v14.12 Groupwise-Robust Preflight Outcome

Date: 2026-08-09

## Audited run

- Run: `mujoco_v14_12_groupwise_robust_preflight_20260809_r1`
- Analysis: `mujoco_v14_12_groupwise_robust_preflight_analysis_20260809_r1`
- Scheduler cells: `t79556..t79566`
- Environment: `HalfCheetah-v5`
- Optimizer seed: `102499482`
- Frozen algorithm revision: `de068387e15018762a130b97447dd06af5baeda5`
- Frozen source manifest: `10063fb6b3f9b125ee73ae090b56b9b39ca1a9e400a7750ac3cb218f088dc155`
- Analysis input SHA-256: `6c2bb1655924ed34a646baa70cd09049fd2d7a2ab84c3936631a679ff52675e9`
- Evidence role: one-optimizer-seed mechanism preflight, no confidence interval

All 11 cells completed naturally through scheduleurm on `node004`. Every cell
was dynamically eligible for `node001..node006`, requested one physical core,
and was synchronized through the run-scoped manifest. The small preflight did
not require artificial spreading over six 192-core nodes. No stale
`jtl110cpu` record or artifact contributed.

The first analysis attempt exposed a version-isolation defect: the v14.12
analyzer imported a v14.11-private CSV registry reader. The audited analysis
uses a repaired v14.12-owned reader that requires exactly all five disturbance
modes crossed with all eight registered held-out seeds. The repair changes no
cell output, gate, threshold, or rank.

## Mechanism result

The function-preserving calibration passed with exactly zero upper, lower, and
combined actor RMS difference. Group identities and cumulative reward guards
were also active: every groupwise arm exposed four source rollouts at each
policy level and recorded zero per-group PPO surrogate-budget violations.

However, robust projection was materially weaker than pooled projection. The
groupwise arms reduced the maximum within-update normalized excess by only
1.99--3.62% on average; the pooled 10% comparator reduced its pooled excess by
8.30%. Increasing the projection budget from 8 to 16 steps raised the
groupwise mean reduction only from 1.99% to 3.20%. Actor-anchor coefficients
0.01 and 0.05 made both policy levels pass the narrow projection audit for the
5% target, but neither produced an accepted learned checkpoint.

## Learned-policy result

The preregistered decision was `do_not_expand`. Five of six groupwise learned
arms selected iteration `-1`, the initial-checkpoint fallback. The remaining
10% groupwise arm selected iteration 3, below the registered minimum iteration
7. The pooled comparator also selected iteration 3 and was ineligible to
authorize expansion by design.

The changed 10% groupwise checkpoint supplied a real reward signal and passed
three of five held-out disturbance gates, but failed the standard condition's
reward floor and worsened one mixed-condition frequency endpoint. Its selected
checkpoint's worst selection-path constraint was low-frequency
`RawLowerLFDriftAbs`, with normalized violation 0.1062. The pooled comparator's
worst selection-path violation was 0.1005 on the same endpoint. No arm passed
all reward-floor and five-endpoint conditions.

## Root cause

v14.12 fixes pooled averaging but leaves two deeper mismatches.

First, frequency correction is applied only after a complete PPO actor update.
For the 10% groupwise arm, lower-policy maximum normalized excess increased
from 0.180 in iteration 0 to 2.052 by iteration 31. Eight guarded correction
steps often reduced the excess, but could not offset the next unconstrained PPO
update. This is an update-order defect, not a missing dual-rate grid point.

Second, each relative target is evaluated on states visited by the current
candidate while the frozen anchor supplies only counterfactual actions on
those states. The paper gate compares candidate and anchor under their own
closed-loop state distributions. Candidate-only training states therefore do
not cover anchor-visited regions that can determine the held-out worst
endpoint.

The next revision must preserve the groupwise reward guard while adding a
frequency trust region around every PPO actor update and a frozen anchor-state
replay batch. It must keep selection paths validation-only and must not relax
the reward floor, five-endpoint targets, minimum learned iteration, or
worst-condition rank.

## Claim boundary

Allowed: v14.12 demonstrates that source-rollout identity and per-group
cumulative reward guards can be implemented and audited without a budget
violation, and it rejects post-PPO groupwise projection as insufficient under
the registered closed-loop gates.

Forbidden: v14.12 supports an accepted learned checkpoint, held-out frequency
separation, reward improvement, no-tradeoff behavior, cross-task generality,
confirmatory evidence, or a submission-ready selected algorithm.
