# MuJoCo v17.6 Full-Horizon Oracle Outcome

Status: `mixed_online_router_and_total_action_limits`

| Environment | Paths | Baseline feasible | Oracle feasible | Recoverable | Oracle infeasible |
|---|---:|---:|---:|---:|---:|
| HalfCheetah-v5 | 40 | 0 | 40 | 40 | 0 |
| Hopper-v5 | 40 | 0 | 33 | 33 | 7 |
| Walker2d-v5 | 40 | 32 | 40 | 8 | 0 |

Across 120 reused paths, the oracle recovered 81 paths that the online v17.4 split did not make jointly feasible. 7 paths remained infeasible even for the acausal bounded full-horizon split.

This is development-only mechanism diagnosis. The acausal oracle is not an online policy and these accessed paths cannot support a fresh performance or generalization claim.
