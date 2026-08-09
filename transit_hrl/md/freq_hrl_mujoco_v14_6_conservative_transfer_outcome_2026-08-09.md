# MuJoCo v14.6 Conservative-Transfer Development Outcome

Date: 2026-08-09

## Frozen protocol

- Run: `mujoco_v14_6_conservative_transfer_screen_20260809_r1`
- Scheduler tasks: `t78213..t78644`
- Frozen algorithm revision: `a20ddbcb28aa0244e5fb337c7b8261cdf93e2f8a`
- Frozen source manifest: `d29b42c228dbdffdfe1f5591c8c641f4491426da85727c2d1273708a04e427a1`
- Design: 48 anchors plus 384 paired continuations
- Held-out evaluation: 5 disturbance modes x 8 seeds per cell
- Optimizer replicates: 16 fresh development seeds
- Evidence role: development screen, not confirmatory

All 432 scheduler cells exited naturally on `node001..node005`; no task was
failed or cancelled. Cell runtimes were 155.0 seconds minimum, 245.7 seconds
median, 386.4 seconds at the 95th percentile, and 435.1 seconds maximum. Peak
observed cell memory was 424 MB. Run-scoped synchronization verified all 432
result directories before strict merging.

The strict merge audited checkpoint files, source identities, seed namespaces,
minimum checkpoint iterations, selected parameter hashes, reconstruction
contracts, and all held-out rows. The analysis input digest was
`5fa247f4e03c63cc214fae824eb004dbfd97b63c70f8361afc8feef3e9416c4b`.

## Decision

No behavior-safe candidate was selected.

| Arm | Complete conditions | Strict lower drift | Exact params | Exact action/reward traces | Trained checkpoints |
|---|---:|---:|---:|---:|---:|
| `s=.025` | 0/15 | 0/15 | yes | 15/15 | yes |
| `s=.050` | 10/15 | 10/15 | yes | 15/15 | yes |
| `s=.075` | 10/15 | 15/15 | yes | 15/15 | yes |
| `s=.100` | 10/15 | 15/15 | yes | 15/15 | yes |
| `s=.125` | 10/15 | 15/15 | yes | 15/15 | yes |
| `s=.150` | 10/15 | 15/15 | yes | 15/15 | yes |
| `s=.200` | 10/15 | 15/15 | yes | 15/15 | yes |

Every strength reproduced its paired control exactly in selected parameters,
latent-policy traces, executed actions, rewards, and returns. Strengths at or
above 0.075 achieved the registered 5% lower-LF reduction in every environment
by disturbance condition. They nevertheless failed the absolute upper-HF
budget in all five Hopper conditions; HalfCheetah and Walker2d passed all gates.

## Scientific diagnosis

v14.6 fixes the v14.5 action discontinuity, but it also exposes an
identifiability problem. The conservative transfer changes the reported upper
and lower responsibilities while preserving the total action. With no active
frequency-sensitive policy objective, candidate and control optimizer
trajectories are identical. The observed lower-LF reduction is therefore a
causal responsibility-coordinate transformation, not evidence that the
hierarchical policies learned better control.

The Hopper failure is not a return tradeoff. Its canonical upper action is
already close to the 0.10 reporting budget, and transferring the EMA component
of the lower action raises the familywise upper-HF bound by roughly 12%. A
valid successor must jointly remove upper high-frequency responsibility and
lower low-frequency responsibility while retaining a reward floor. Merely
relaxing the Hopper threshold or relabeling the transfer channel is forbidden.

## Next design boundary

The next development protocol must separate two questions:

1. A joint causal responsibility projection can provide a function-preserving
   initialization and control both upper-HF and lower-LF leakage.
2. A learned variant must activate frequency-sensitive policy objectives and
   demonstrate non-identical policy parameters plus a behavioral advantage
   such as sample efficiency, robustness, or recovery, not only a different
   decomposition of the same action trace.

Any projection-only result is calibration or ablation evidence. It is not a
positive learned Freq-HRL result and cannot be promoted directly to a
confirmatory protocol.

## Claim boundary

Allowed: v14.6 establishes exact pathwise function preservation and uniform
lower-LF responsibility reduction for strengths at or above 0.075, while
showing that one-sided transfer violates Hopper's upper-HF budget and does not
alter learned behavior.

Forbidden: v14.6 validates learned behavior-safe Freq-HRL, improves control
performance, or supplies a selected algorithm for confirmatory testing.
