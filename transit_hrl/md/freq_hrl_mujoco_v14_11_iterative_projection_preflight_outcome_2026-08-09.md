# MuJoCo v14.11 Iterative-Projection Preflight Outcome

Date: 2026-08-09

## Audited run

- Run: `mujoco_v14_11_iterative_projection_preflight_20260809_r1`
- Analysis: `mujoco_v14_11_iterative_projection_preflight_analysis_20260809_r1`
- Scheduler cells: `t79506..t79515`
- Environment: `HalfCheetah-v5`
- Optimizer seed: `2619431165`
- Frozen algorithm revision: `ed87b9a0d3e2b78a9c7c10fd76291c64af246564`
- Frozen source manifest: `97f2cbfe15bc5d5ae061a2c23a9159202ba9f20eec85d94e2bb2cf52ee475863`
- Analysis input SHA-256: `af488b22500a7d859419b370492144d63d1175e10e4319bc0df589329a2d8209`
- Evidence role: one-optimizer-seed mechanism preflight, no confidence interval

All ten cells completed naturally through scheduleurm on `node004`. The jobs
were dynamically eligible for `node001..node006`; colocating nine small
continuations with the staged anchor used only a small fraction of the node's
192 physical cores and avoided redundant staging. No stale `jtl110cpu` record
or artifact contributed to this run.

## Calibration

The function-preserving projection calibration passed. Actor RMS differences
and paired reward/trace differences were exactly zero. Across the five held-out
disturbance modes, the minimum registered lower-frequency reduction was
66.2--68.8% and effective upper-HF power fell by 37.6--38.3%. This remains a
coordinate-transform check, not learned-policy evidence.

## Iterative mechanism

The iterative correction fixed the narrow v14.10 step-size defect. Relative to
the matched one-step arm's 0.94% mean within-update power reduction:

| arm | accepted steps | multistep updates | mean reduction | gain over k=1 |
|---|---:|---:|---:|---:|
| `k=4`, scale `1/3`, target 5% | 226 | 57 | 3.87% | 2.94 pp |
| `k=8`, scale `1/3`, target 5% | 416 | 56 | 6.97% | 6.03 pp |
| `k=16`, scale `1/3`, target 5% | 609 | 52 | 9.79% | 8.85 pp |
| `k=8`, scale `3/10`, target 5% | 214 | 39 | 7.98% | 7.04 pp |
| `k=8`, scale `3/10`, target 10% | 178 | 38 | 10.18% | 9.25 pp |

There were zero cumulative PPO reward-surrogate budget violations. In the
aggressive 10% arm, lower and upper same-batch targets were reached on 28/32
and 29/32 updates, respectively.

## Learned-policy result

The decision was `do_not_expand`. Every learned arm selected iteration `-1`,
the registered initial-checkpoint fallback. Therefore no learned actor or
action change entered held-out evaluation and no performance claim is
available.

The best near-candidate occurred in the aggressive 10% arm at iteration 7. Its
mean selection-path return increased from 878.1 to 995.0 and all five aggregate
frequency endpoints decreased. Nevertheless, the selector is deliberately
worst-condition first: its largest paired normalized violation was 0.177,
versus 0.111 for the initial fallback, and its worst reward-floor slack was
-0.120. It was correctly rejected.

## Root cause

The deployment constraint pools four training rollouts before computing one
mean frequency power and one mean PPO reward guard. Checkpoint selection instead
computes a separate paired target for each disturbance mode and ranks the worst
of six constraints per mode: reward floor plus five effective/raw/latent
frequency endpoints. The v14.11 update can therefore improve every pooled mean
while allowing one disturbance group to regress. More iteration count or dual
rate tuning does not repair this objective mismatch.

The next revision must retain rollout identity through concatenation, project
the worst group-relative frequency excess, and reject a projection step if any
training group exceeds its cumulative pre-projection reward-loss allowance.
Selection seeds remain validation-only and must not enter this correction.

## Claim boundary

Allowed: v14.11 demonstrates that iterative deterministic deployment-frequency
projection materially outperforms one correction on the sampled training
batches while respecting its registered cumulative PPO surrogate budget.

Forbidden: v14.11 supports an accepted learned checkpoint, held-out frequency
separation, reward improvement, no-tradeoff behavior, cross-task generality,
confirmatory evidence, or a submission-ready selected algorithm.
