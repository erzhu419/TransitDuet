# MuJoCo v14.5 Paired-Anchor Development Outcome

Date: 2026-08-09

## Frozen protocol

- Run: `mujoco_v14_5_paired_anchor_screen_20260809_r1`
- Scheduler tasks: `t77700..t78131`
- Frozen algorithm revision: `d3201ad9b12a558fbcec10887d3c9217c6c2585c`
- Frozen source manifest: `dfa329af4eb2aa9dc6206cd2e2ae712936e835c7731ef8b6e6fa958629831fa0`
- Design: 48 direct anchors plus 384 compute-matched continuation cells
- Held-out evaluation: 5 disturbance modes x 8 seeds per cell
- Evidence role: development screen, not confirmatory

Every routed continuation and its direct comparator loaded the exact matching
environment-by-optimizer checkpoint, including optimizer state. Strict merging
verified all 432 checkpoint file hashes, parameter hashes, source identities,
seed grids, router schedules, anchor contracts, and 17,280 held-out evaluation
rows. All tasks exited naturally on `node001..node006`; median cell runtime was
282.2 seconds and the maximum was 448.6 seconds.

## Decision

No behavior-safe candidate was selected.

| Arm | Complete | Reward NI | Responsibility | Raw drift | Upper HF | Minimum trained fraction |
|---|---:|---:|---:|---:|---:|---:|
| `s=.10, upper=0, lower=0` | 0/15 | 5/15 | 2/15 | 0/15 | 10/15 | 0.0625 |
| `s=.10, upper=0, lower=.01` | 5/15 | 5/15 | 10/15 | 10/15 | 10/15 | 0 |
| `s=.10, upper=0, lower=.10` | 5/15 | 5/15 | 15/15 | 10/15 | 10/15 | 0 |
| `s=.10, upper=0, lower=1.0` | 5/15 | 5/15 | 15/15 | 11/15 | 10/15 | 0 |
| `s=.10, upper=.05, lower=.10` | 5/15 | 5/15 | 15/15 | 10/15 | 10/15 | 0.0625 |
| `s=.10, upper=.20, lower=1.0` | 5/15 | 5/15 | 15/15 | 15/15 | 10/15 | 0 |
| `s=.15, upper=.05, lower=.10` | 5/15 | 5/15 | 15/15 | 15/15 | 10/15 | 0.0625 |

The two strongest drift arms passed both physical raw-drift and responsibility
gates in every registered condition. They nevertheless met reward
noninferiority only in the five Walker2d conditions. Every routed arm failed
return noninferiority in all HalfCheetah and Hopper conditions, and Hopper also
missed the upper-HF budget. No arm met the minimum trained-checkpoint fraction.

## Root cause and next design

Paired initialization removed the optimizer-trajectory confound, and the
analytic Gaussian anchor constrained policy-space movement. It did not make the
router intervention function preserving. At nonzero strength, the current
router subtracts a causal EMA baseline from the lower action without adding
that removed low-frequency component to the upper execution channel. The
closed-loop action therefore changes even when the actor parameters are
identical to the anchor. Actor KL cannot bound this deterministic action jump.

The next design must transfer the removed causal low-frequency component into
the upper executed responsibility so that upper plus lower reconstructs the
pre-router total action exactly at takeover. Candidate and direct continuation
must also use the same minimum eligible checkpoint iteration so that a selected
result represents learned continuation rather than the untouched anchor.

## Claim boundary

Allowed: v14.5 shows that paired proximal continuation can stabilize latent
policy movement and obtain uniform drift reduction, but the non-function-
preserving router causes task-dependent return loss.

Forbidden: v14.5 supports physical no-tradeoff, learned behavior-safe routing,
or a confirmatory Freq-HRL claim.
