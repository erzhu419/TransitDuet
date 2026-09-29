# MuJoCo v23 Causal Upper-Target Development

- Status: `v23_causal_upper_projection_target_development_stops`
- Cells: 48
- Selected candidate: `None`
- Evidence role: `fresh_root_causal_upper_target_development_not_confirmatory`

## terminal_reserve_lower_only_decision_time_010

- Advances: `false`
- Reward wins versus old control: 8/12

| Environment | Zero reward | Old reward | Candidate reward | Reward vs zero | Reward vs old | Wins vs old | Component vs zero/old | Total vs zero/old | Total correction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HalfCheetah-v5 | 1842.244 | 2267.205 | 2113.055 | 0.1470 | -0.0680 | 1/4 | 0.3085/-0.1449 | 0.4598/-0.0587 | 0.1189 |
| Hopper-v5 | 184.191 | 194.559 | 200.754 | 0.0899 | 0.0318 | 3/4 | -0.0165/-0.4649 | -0.0448/-0.6592 | 0.4644 |
| Walker2d-v5 | 212.414 | 210.589 | 302.903 | 0.4260 | 0.4384 | 4/4 | 0.3611/0.2657 | 0.6916/0.3411 | 0.0352 |

### Frozen Gates

- all_matched_arms_valid: `false`
- training_activity_audit_supported: `true`
- reward_floor_vs_zero_in_every_environment: `true`
- reward_floor_vs_old_in_every_environment: `false`
- reward_wins_in_every_environment: `false`
- total_reward_wins: `true`
- reward_improves_in_two_environments: `true`
- component_reduces_in_two_environments: `true`
- total_reduces_in_two_environments: `true`
- component_noninferior_vs_zero_and_old: `false`
- total_noninferior_vs_zero_and_old: `false`
- hopper_correction_burden_bounded: `false`

## terminal_reserve_decision_time_uniform_010

- Advances: `false`
- Reward wins versus old control: 5/12

| Environment | Zero reward | Old reward | Candidate reward | Reward vs zero | Reward vs old | Wins vs old | Component vs zero/old | Total vs zero/old | Total correction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HalfCheetah-v5 | 1842.244 | 2267.205 | 2013.837 | 0.0931 | -0.1118 | 2/4 | 0.2821/-0.1887 | 0.4424/-0.0927 | 0.1227 |
| Hopper-v5 | 184.191 | 194.559 | 175.933 | -0.0448 | -0.0957 | 0/4 | 0.2698/-0.0523 | 0.4317/0.0975 | 0.2526 |
| Walker2d-v5 | 212.414 | 210.589 | 242.995 | 0.1440 | 0.1539 | 3/4 | 0.3121/0.2093 | 0.6330/0.2158 | 0.0419 |

### Frozen Gates

- all_matched_arms_valid: `false`
- training_activity_audit_supported: `true`
- reward_floor_vs_zero_in_every_environment: `true`
- reward_floor_vs_old_in_every_environment: `false`
- reward_wins_in_every_environment: `false`
- total_reward_wins: `false`
- reward_improves_in_two_environments: `false`
- component_reduces_in_two_environments: `true`
- total_reduces_in_two_environments: `true`
- component_noninferior_vs_zero_and_old: `false`
- total_noninferior_vs_zero_and_old: `false`
- hopper_correction_burden_bounded: `false`

## Claim Boundary

no confirmatory, superiority, generalization, or manuscript claim may be made from four optimizer roots
