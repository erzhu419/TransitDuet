# MuJoCo v17.5 Feasibility Diagnostic Outcome

## Decision

`greedy_feasibility_projection_not_advanced`

The v17.5 current-step feasibility projection is not eligible for a fresh
preflight. It eliminates its local normalized regret by construction, but that
local quantity does not reliably improve the full-trajectory frequency
endpoints.

## Integrity

All three scheduleurm tasks (`t85547`--`t85549`) completed on `node003`. The
refactored v17.4 implementation exactly reproduced the frozen v17.4 reward,
executed-action, and latent-policy hashes on all 120 reused paths. Every audited
numeric endpoint also matched exactly. The checkpoint and training history
remained server-only; only JSON and CSV diagnostics were synchronized.

The v17.5 and v17.4 closed loops diverged on all 120 paths. This is expected:
the router preserves the instantaneous summed action, but its responsibility
history is part of the learned policy state and therefore changes later latent
actions. Reward differences are consequently descriptive development results,
not a paired function-preservation claim.

## Endpoint Results

| Environment | v17.4 lower LF | v17.5 lower LF | v17.4 upper HF | v17.5 upper HF | v17.4 return | v17.5 return |
|---|---:|---:|---:|---:|---:|---:|
| HalfCheetah-v5 | 0.00355650 | 0.00353278 | 0.00497676 | 0.00502688 | 739.8032 | 755.2836 |
| Hopper-v5 | 0.02217873 | 0.03215537 | 0.00454285 | 0.00494646 | 179.5530 | 179.8229 |
| Walker2d-v5 | 0.00169081 | 0.00221401 | 0.00107651 | 0.00130771 | 282.2791 | 282.2860 |

The local budget-excess regret fell to numerical zero in all three
environments. Nevertheless, lower-LPF32 improved in only one environment,
lower-budget violation improved in only one, joint-budget feasible rate
improved in none, and upper-HPF8 improved in none. Hopper lower drift increased
by 45.0% and Walker2d lower drift increased by 30.9%.

## Interpretation

The failure is temporal, not a numerical implementation defect. When the
current lower budget is infeasible, v17.5 greedily chooses the split with the
smallest current violation. That choice changes the FIR history and can raise
future unavoidable floors. The v17.4 constant-tail objective sometimes accepts
a small current excess to obtain a better trajectory-level decomposition.

The next experiment must freeze the total-action trace and solve the complete
bounded HPF8/LPF32 responsibility allocation as a convex full-horizon oracle.
Only that oracle can determine whether the remaining budget gap is attributable
to the online router or to the learned total action. No v17.5 result is eligible
for a positive manuscript claim.
