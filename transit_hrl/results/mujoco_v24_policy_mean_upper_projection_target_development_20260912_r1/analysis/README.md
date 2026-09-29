# MuJoCo v24 Policy-Mean Upper-Target Development

- Status: `v24_policy_mean_upper_projection_target_development_stops`
- Cells: 48
- Selected candidate: `None`
- Evidence role: `fresh_root_policy_mean_upper_target_development_not_confirmatory`
- Deterministic-target audit failures: 12

## terminal_reserve_policy_mean_uniform_010

- Advances: `false`
- Reward wins versus causal first-sample control: 7/12

| Environment | Zero reward | Causal reward | Hindsight reward | Candidate reward | Candidate reward delta zero/causal/hindsight | Wins vs causal | Upper MSE causal/candidate/reduction | Component reduction zero/causal/hindsight | Total reduction zero/causal/hindsight | Total correction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HalfCheetah-v5 | 1995.833 | 2199.151 | 2102.626 | 2005.825 | 0.0050/-0.0879/-0.0460 | 2/4 | 0.1669/0.1517/0.0911 | 0.2579/0.1017/0.1860 | 0.3725/0.1148/0.3029 | 0.0671 |
| Hopper-v5 | 176.535 | 161.359 | 188.537 | 200.446 | 0.1354/0.2422/0.0632 | 4/4 | 0.6500/0.6993/-0.0760 | 0.2870/0.1053/0.0976 | 0.4022/0.0847/0.1985 | 0.2406 |
| Walker2d-v5 | 164.479 | 249.831 | 256.892 | 199.745 | 0.2144/-0.2005/-0.2225 | 1/4 | 0.5361/0.7360/-0.3729 | 0.0064/-0.3850/-0.2247 | 0.3894/-0.3425/-0.1181 | 0.0876 |

### Frozen Gates

- all_matched_arms_valid: `true`
- training_activity_audit_supported: `false`
- upper_mse_reduces_in_two_environments: `false`
- upper_mse_noninferior_in_every_environment: `false`
- reward_floor_vs_all_controls_in_every_environment: `false`
- reward_wins_in_every_environment: `false`
- total_reward_wins: `false`
- reward_improves_vs_causal_in_two_environments: `false`
- component_reduces_in_two_environments: `true`
- total_reduces_in_two_environments: `true`
- component_noninferior_vs_all_controls: `false`
- total_noninferior_vs_all_controls: `false`
- hopper_correction_burden_bounded: `true`

## Claim Boundary

no confirmatory, superiority, generalization, or manuscript claim may be made from four optimizer roots
