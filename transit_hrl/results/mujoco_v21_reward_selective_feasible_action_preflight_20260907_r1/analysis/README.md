# MuJoCo v21 Reward-Selective Feasible-Action Preflight

- Status: `v21_reward_selective_feasible_action_preflight_stops`
- Cells: 48
- Advance candidate: `None`
- Evidence role: development preflight only

| Environment | Raw reward | Reserve reward | Uniform reward | Selective reward | Reward delta | Component vs reserve | Component vs uniform |
|---|---:|---:|---:|---:|---:|---:|---:|
| HalfCheetah-v5 | 2182.916 | 1536.375 | 1705.939 | 1612.543 | -0.0604 | 0.1518 | -0.1519 |
| Hopper-v5 | 260.922 | 217.540 | 235.539 | 237.347 | 0.0030 | 0.3013 | -0.1283 |
| Walker2d-v5 | 272.653 | 150.345 | 257.349 | 196.998 | 0.0985 | 0.2711 | -0.6093 |

## Gates

- projected_controls_valid: `true`
- reward_improves_in_two_environments: `true`
- pooled_normalized_reward_positive: `true`
- reward_floor_in_every_environment: `false`
- component_reduction_vs_reserve: `true`
- total_reduction_vs_reserve: `true`
- component_noninferior_to_uniform: `false`
- total_noninferior_to_uniform: `false`
- physical_burden_supported: `false`
- weight_audit_supported: `true`

A failed gate stops this weighting mechanism. These roots cannot be reused for temperature, clip, coefficient, or schedule tuning.
