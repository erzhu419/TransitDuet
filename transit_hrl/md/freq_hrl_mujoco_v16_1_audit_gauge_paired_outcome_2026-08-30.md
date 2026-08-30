# MuJoCo v16.1 Adaptive-Gauge Paired Continuation Outcome

## Decision

`audit_gauge_paired_preflight_not_supported`

This result is valid development evidence, not confirmatory evidence. All 27
scheduleurm tasks completed, and the analysis used nine paired
environment-by-optimizer-seed cells with 40 held-out paths per arm.

## Frozen Result

- Exact additive-action reconstruction: 9/9 cells.
- Adaptive cutoff active: 9/9 cells.
- Held-out reward noninferiority versus the compute-matched control: 9/9 cells.
- Latent-frequency noninferiority versus control: 9/9 cells.
- A trained candidate checkpoint selected: 1/9 cells.
- At least 10% canonical-frequency reduction versus control: 1/9 cells.
- Per-environment two-of-three support gate: 0/3 environments.

The one positive cell was Hopper-v5 with optimizer seed 2868862553. Its selected
iteration 11 improved reward from 231.59 to 288.81 and reduced canonical
frequency merit by 12.65%. The other eight cells used the preregistered anchor
fallback, so their candidate and control outcomes were identical.

## Diagnosis

The adaptive gauge repaired the exact reconstruction and reward-preservation
failures of the v16 EMA gauge, but the continuation optimizer did not reliably
improve raw responsibility separation. Training-batch deployment-frequency
projection was active only while a batch violated its target. The actor update
itself had no closed-loop acceptance test on frozen anchor paths. Consequently,
reward PPO updates continued even when HalfCheetah or Walker2d already met the
absolute frequency budgets, then degraded held-out reward or frequency ranks.

This is an update-acceptance failure, not evidence that more repetitions of the
same continuation will help. The next algorithm revision must make an actor
update transactional against paired anchor paths, freeze the reward actor during
frequency restoration, and abstain when the anchor is already feasible.

## Claim Boundary

Allowed: the adaptive gauge is exactly function preserving in the evaluated
additive controller, and paired anchor fallback preserved reward and latent
noninferiority in this development panel.

Forbidden: v16.1 establishes cross-environment raw responsibility separation,
reward improvement, leakage no-tradeoff, fresh-seed confirmation, or the final
Freq-HRL algorithm.
