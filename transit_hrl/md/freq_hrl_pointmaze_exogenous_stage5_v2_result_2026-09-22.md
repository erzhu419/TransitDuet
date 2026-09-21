# PointMaze Exogenous-Control Stage-5 V2 Result

Date: 2026-09-22

## Outcome

The frozen 16-cell V2 confirmation completed with eight fresh optimizer roots,
256 final held-out episodes, and 256 paired untrained episodes. The ordinary
HRL substrate gate is **supported**, so a fresh external-stream frequency
routing attribution experiment is admitted.

| Method | Tracking success mean [95% CI] | Return | Tracking RMSE |
|---|---:|---:|---:|
| flat external history | 0.874 [0.780, 0.968] | 226.558 | 0.319 |
| HRL external history | 0.887 [0.860, 0.915] | 232.308 | 0.299 |

The three registered conjuncts all passed:

- the HRL absolute tracking-success CI lower endpoint was 0.860, above 0.50;
- HRL final-minus-untrained tracking success was +0.707
  [0.676, 0.738]; and
- HRL final-minus-untrained return was +116.632
  [109.639, 123.625].

HRL-versus-flat differences were not significant. Tracking-success improvement
was +0.013 [-0.088, 0.114], and return improvement was +5.750
[-6.651, 18.152]. This comparison was descriptive and non-gating.

## Execution Audit

All 16 method/root cells were unique and complete. Formal optimizer and role
seeds were disjoint from V1, the stability screen, and V2 preflight. Every cell
used eight training rollout roots, 16 selection paths, 16 held-out paths, 768
iterations, and 300 transitions per episode. External paths were exactly
paired across methods within root and seed.

State dimensions, parameter counts, causal observability, force RMS/period,
option boundaries, finite outputs, runtime versions, and disabled legacy
mechanisms matched the protocol. Flat and HRL each received 6,144 training
episodes per cell. Their different gradient-update totals reflect one joint
PPO versus separate upper/lower PPO optimizers, not different environment
interaction budgets. Node003 through node006 each ran four cells. Only compact
result JSON files were synchronized.

## Claim Boundary

Allowed: ordinary learned HRL reliably learns the separate-exogenous PointMaze
substrate under the frozen fresh-seed V2 protocol. This satisfies the registered
prerequisite for a new frequency-routing attribution experiment.

Forbidden: V2 proves HRL superiority over flat PPO, validates selective
frequency assignment, proves leakage control or promotion, or establishes
domain-general Freq-HRL.

