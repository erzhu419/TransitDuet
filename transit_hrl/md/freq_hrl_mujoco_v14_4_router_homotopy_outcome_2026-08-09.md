# MuJoCo v14.4 Router-Homotopy Development Outcome

Date: 2026-08-09

## Frozen protocol

- Run: `mujoco_v14_4_router_homotopy_screen_20260809_r1`
- Scheduler tasks: `t77260..t77643`
- Frozen algorithm revision: `03482136f155bdb86f1cf421cd320555bcb42c81`
- Frozen source manifest: `ed3f1db546d7171ceb11b05938890caead3b1f685a0fbcb4618fd4d2b4fef974`
- Design: 3 environments x 8 arms x 16 optimizer replicates = 384 cells
- Held-out evaluation: 5 disturbance modes x 8 seeds per cell
- Evidence role: development screen, not confirmatory

All 384 cells exited naturally on `node001..node006`. Strict merging verified
the frozen source identity, per-iteration routing schedule, target-strength
checkpoint selection, checkpoint hashes, and all 15,360 held-out evaluation
rows. Median cell runtime was 316.2 seconds; the slowest was 427.6 seconds.

## Decision

No behavior-safe candidate was selected.

| Arm | Complete | Reward NI | Responsibility | Raw drift | Upper HF |
|---|---:|---:|---:|---:|---:|
| `s=.10, constant` | 0/15 | 1/15 | 9/15 | 5/15 | 10/15 |
| `s=.10, linear 0-.25` | 0/15 | 10/15 | 5/15 | 5/15 | 10/15 |
| `s=.10, linear .125-.375` | 3/15 | 3/15 | 6/15 | 5/15 | 11/15 |
| `s=.10, linear .25-.50` | 0/15 | 0/15 | 5/15 | 5/15 | 10/15 |
| `s=.10, cosine .25-.50` | 0/15 | 3/15 | 5/15 | 5/15 | 12/15 |
| `s=.15, linear .125-.375` | 0/15 | 2/15 | 10/15 | 6/15 | 11/15 |
| `s=.15, cosine .25-.50` | 0/15 | 5/15 | 10/15 | 7/15 | 10/15 |

The fastest ramp had a positive mean return difference of `+60.76`, but this
did not transfer to the physical separation gates: responsibility and raw
lower-LF drift each passed only 5 of 15 conditions. The schedule with the most
joint passes reached 3 of 15 and had a mean return difference of only `+0.99`.

## Root cause and next design

Exposing the routing strength removed the hidden-state defect in v14.3, but it
did not remove optimizer-trajectory confounding. Every candidate still trained
independently from the direct baseline. The policy can therefore compensate for
the router by changing its latent action, recovering reward while reintroducing
low-frequency power into the physical action. Conversely, stronger routing can
reduce physical drift while moving the policy onto a lower-return trajectory.

The next protocol must start every candidate from the exact matching direct
baseline checkpoint. Router continuation will be paired within optimizer seed,
and policy updates will be constrained relative to the anchor policy. This
tests the incremental effect of frequency routing without treating unrelated
optimization trajectories as if they were a causal intervention.

## Claim boundary

Allowed: v14.4 shows that observed router homotopy can improve return in some
conditions, but it does not produce stable physical separation across tasks.

Forbidden: v14.4 supports physical no-tradeoff, a universal homotopy schedule,
or a confirmatory Freq-HRL claim.
