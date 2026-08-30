# MuJoCo v17.4 Streaming Audit Projection Outcome

## Decision

`streaming_audit_projection_preflight_not_supported`

This is valid development evidence, not confirmatory evidence. All three
scheduleurm tasks (`t85521`--`t85523`) completed on `node003` with dynamic
placement and no node binding. The synchronized bundle contains only cell
summaries, evaluation CSV files, and server artifact locations; checkpoints and
training histories remain on the worker.

## Paired Mechanics

Each environment trained one strength-zero policy with the complete HPF8/LPF32
FIR state, then evaluated the same frozen checkpoint at strengths zero and one
on the same 40 paths. Across all 120 pairs:

- reward, executed-action, and latent-policy trace hashes matched exactly;
- numeric reward and latent-policy metrics matched exactly;
- router and responsibility reconstruction passed the frozen `1e-7` RMS gate;
- transition counts satisfied the asynchronous hierarchy contract;
- all three candidate cells met the absolute upper-HPF8 power budget.

The causal projection is therefore function-preserving on the frozen panel and
materially improves the measured split. It does not satisfy the full absolute
frequency contract.

## Frozen Frequency Result

The registered power budgets were `0.075^2 = 0.005625` for upper HPF8 and
`0.0475^2 = 0.00225625` for lower LPF32.

| Environment | Candidate upper power | Candidate lower power | Lower reduction | Joint reduction | Upper feasible rate | Decision |
|---|---:|---:|---:|---:|---:|---|
| HalfCheetah-v5 | 0.004977 | 0.003556 | 60.25% | 41.24% | 97.55% | not supported |
| Hopper-v5 | 0.004543 | 0.022179 | 63.52% | 62.91% | 100.00% | not supported |
| Walker2d-v5 | 0.001077 | 0.001691 | 90.44% | 88.05% | 100.00% | supported |

Unlike v17.3, v17.4 reduced lower-LPF32 and normalized joint merit in every
environment. It also kept candidate upper-HPF8 power below the absolute budget
in every environment. The all-environment expansion rule still failed: only
Walker2d met the absolute lower budget, and HalfCheetah missed the registered
`0.99` upper-feasibility threshold.

The failure was not isolated to one disturbance label. HalfCheetah candidate
lower power remained between 0.003505 and 0.003601 across the five modes;
Hopper remained between 0.020129 and 0.024939. The issue is therefore the
responsibility feasible envelope or projection objective, not a single OOD
condition.

## Diagnosis

The streaming state and receding update corrected the macro-boundary mismatch
identified in v17.3. The remaining contract treats fixed upper and lower RMS
budgets as jointly attainable even when component bounds restrict the split of
the realized total action. v17.4 reports upper infeasibility directly, exposing
this problem in HalfCheetah. It does not yet report the corresponding joint
upper/lower feasible set or an unavoidable lower residual floor.

The next mechanism must distinguish two cases at every step:

1. If the physical interval intersects both current audit-budget intervals,
   enforce both budgets as hard constraints.
2. If the intersection is empty, compute and report the minimum normalized
   violation allowed by the physical interval, then optimize excess above that
   floor instead of treating an impossible absolute target as algorithmic
   leakage.

This is a mechanism change and requires a new frozen protocol and fresh roots.
The v17.4 held-out paths cannot be reused for selection.

## Claim Boundary

Allowed: v17.4 validates exact pathwise function preservation, absolute upper
budget compliance, and large paired lower-LPF32 and joint-merit reductions in
all three environments. It also identifies where the fixed absolute budget is
not attained under component constraints.

Forbidden: v17.4 validates the frozen all-environment absolute frequency
contract, leakage no-tradeoff, reward improvement, optimizer-seed robustness,
fresh-seed confirmation, or a final Freq-HRL algorithm.
