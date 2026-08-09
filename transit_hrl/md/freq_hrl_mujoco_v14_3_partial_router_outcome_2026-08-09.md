# MuJoCo v14.3 Partial-Router Development Outcome

Date: 2026-08-09

## Frozen protocol

- Run: `mujoco_v14_3_partial_router_screen_20260809_r1`
- Scheduler tasks: `t76832..t77215`
- Frozen algorithm revision: `8d3aafd9c9447de3fd18664fe311cda79ad811ab`
- Frozen source manifest: `e683b7212e19943d89ffc716d5f08594760e3401f9ad22d6ededf6c56e597d25`
- Design: 3 environments x 8 arms x 16 optimizer replicates = 384 cells
- Held-out evaluation: 5 disturbance modes x 8 seeds per cell
- Evidence role: development screen, not confirmatory

All 384 cells finished naturally on `node001..node006`. The local audit verified
384 checkpoint hashes, 384 source identities, 384 trained checkpoint selections,
and 15,360 evaluation rows with the frozen per-arm router strength.

## Decision

No behavior-safe candidate was selected.

| Arm | Complete gates | Reward NI | Responsibility drift | Raw drift | Upper HF |
|---|---:|---:|---:|---:|---:|
| `a=.04, s=.06` | 0/15 | 2/15 | 15/15 | 0/15 | 15/15 |
| `a=.04, s=.08` | 0/15 | 0/15 | 15/15 | 5/15 | 10/15 |
| `a=.04, s=.10` | 4/15 | 4/15 | 15/15 | 10/15 | 10/15 |
| `a=.04, s=.15` | 1/15 | 1/15 | 15/15 | 5/15 | 10/15 |
| `a=.10, s=.06` | 0/15 | 0/15 | 15/15 | 7/15 | 10/15 |
| `a=.10, s=.10` | 0/15 | 2/15 | 15/15 | 5/15 | 15/15 |
| `a=.10, s=.15` | 0/15 | 0/15 | 15/15 | 5/15 | 10/15 |

The result separates two claims. Partial routing reliably changes the assigned
responsibility signal, but a fixed intervention strength is not a behavior-safe
physical controller across tasks. The strongest arm, `a=.04, s=.10`, reduced
raw lower-LF drift in 10 of 15 conditions but met the joint gate in only 4.

## Root cause and next design

The fixed router changes physical actions from the first policy update. A
plausible mechanism, not established by v14.3, is that this early intervention
separates optimizer trajectories before the policy has learned the unrouted
control task. Its observed effect is environment dependent: Hopper lacks
raw-drift and upper-HF support at the best fixed strength, while Walker2d
primarily misses return noninferiority.

The next development protocol must therefore test a causal homotopy: train the
same action parameterization without routing, then ramp to a frozen target
strength while exposing the current strength in the policy state. Held-out
evaluation must always use the target strength. This is an algorithm change and
requires fresh development seeds; v14.3 outcomes cannot be reused as confirmation.

## Claim boundary

Allowed: v14.3 shows that fixed partial routing improves responsibility-space
drift everywhere and raw drift in a subset of registered conditions, but does
not provide a behavior-safe cross-task candidate.

Forbidden: v14.3 supports physical no-tradeoff, universal behavioral frequency
separation, or a confirmatory Freq-HRL result.
