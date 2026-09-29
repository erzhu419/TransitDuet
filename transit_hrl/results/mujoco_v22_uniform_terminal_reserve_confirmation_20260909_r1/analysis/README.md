# MuJoCo v22 Uniform Terminal-Reserve Confirmation

- Status: `v22_uniform_terminal_reserve_confirmation_not_supported`
- Cells: 288
- Selected candidate: `None`
- Evidence role: `fresh_root_confirmation_of_unchanged_v21_uniform_control`

| Environment | Reserve reward | Uniform reward | Reward delta | Reward wins | Component reduction | Total reduction | Total correction | Change rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HalfCheetah-v5 | 1744.708 | 1686.392 | -0.0334 [-0.1280, 0.0692] | 18/32 | 0.2726 [0.1244, 0.3880] | 0.3754 [0.2109, 0.4980] | 0.0930 | 0.3462 |
| Hopper-v5 | 187.520 | 182.169 | -0.0285 [-0.0918, 0.0333] | 17/32 | 0.3115 [0.2709, 0.3506] | 0.3683 [0.3060, 0.4277] | 0.2831 | 0.6320 |
| Walker2d-v5 | 236.178 | 247.290 | 0.0470 [-0.0616, 0.1683] | 19/32 | 0.2725 [0.1605, 0.3656] | 0.5657 [0.4118, 0.6991] | 0.0578 | 0.3177 |

## Registered Gates

- projected_controls_valid: `false`
- uniform_training_audit_supported: `true`
- reward_noninferior_in_every_environment: `false`
- pooled_reward_noninferior: `true`
- pooled_component_reduction_supported: `true`
- pooled_total_reduction_supported: `true`
- component_improves_in_two_environments: `true`
- total_improves_in_two_environments: `true`
- component_noninferior_in_every_environment: `true`
- total_noninferior_in_every_environment: `true`
- correction_magnitude_bounded: `false`

## Pooled Paired Intervals

- reward_relative_delta: -0.0050 [-0.0428, 0.0359]
- component_correction_reduction: 0.2855 [0.2362, 0.3255]
- total_correction_reduction: 0.4365 [0.3818, 0.4820]

## Claim Boundary

support establishes only fresh-root MuJoCo evidence for correction reduction with projected-reward noninferiority; it does not establish reward superiority, cross-domain superiority, or deployment validity
