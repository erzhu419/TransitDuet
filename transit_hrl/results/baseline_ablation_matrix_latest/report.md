# Baseline And Ablation Matrix

Baseline/ablation evidence is paired over identical seeds and stress scenarios. It checks whether Freq-HRL beats non-frequency, misrouted-frequency, no-promotion, and no-leakage alternatives; it does not replace native Transit learned-policy validation. Flat PPO/SAC/TD3 and generic learned HRL are registered separately and are not credited unless their paired rows are present.

- claim status: `supported`
- scenario Freq-HRL-family win rate: `1.000`
- required baselines positive: `['allfreq_alllayers', 'hrl_raw', 'no_leakage', 'no_promotion', 'swapped', 'vanilla_rl']`
- support overrides: `[{'baseline': 'no_promotion', 'source_artifact': 'native_promotion_v47', 'status': 'supported', 'supported_metrics': ['avg_wait_min', 'ep_reward'], 'boundary': 'No-promotion ablation is credited from the native promotion stress artifact, where interval_only is the no-promotion control. Raw global trading Sharpe remains reported separately.'}]`
- strong learned baseline status: `registered_missing`
- required baselines inconclusive: `[]`
- required baselines not supported: `[]`
- required baselines missing: `[]`

## Strong Learned Baseline Registration

| baseline | evidence status | required metrics | supported metrics | paper role |
|---|---|---|---|---|
| flat_ppo | registered_missing | sharpe,total_return,FocusScore |  | must_complete_or_limit |
| flat_sac | registered_missing | sharpe,total_return,FocusScore |  | must_complete_or_limit |
| flat_td3 | registered_missing | sharpe,total_return,FocusScore |  | must_complete_or_limit |
| generic_hrl_ppo | registered_missing | sharpe,total_return,FocusScore |  | must_complete_or_limit |

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| freq_hrl_vs_vanilla_rl_sharpe | supported | sharpe | 35 | +54.2815 | +52.6139 | +56.1042 | 1.00 |
| freq_hrl_vs_vanilla_rl_total_return | supported | total_return | 35 | +0.9734 | +0.9356 | +1.0092 | 1.00 |
| freq_hrl_vs_vanilla_rl_FocusScore | supported | FocusScore | 35 | +0.9917 | +0.9529 | +1.0300 | 1.00 |
| freq_hrl_vs_vanilla_rl_LowerLFDrift | not_supported | LowerLFDrift | 35 | +1.7533 | +1.6985 | +1.7997 | 0.00 |
| freq_hrl_vs_hrl_raw_sharpe | supported | sharpe | 35 | +50.3703 | +48.0434 | +52.7749 | 1.00 |
| freq_hrl_vs_hrl_raw_total_return | supported | total_return | 35 | +0.7933 | +0.7577 | +0.8271 | 1.00 |
| freq_hrl_vs_hrl_raw_FocusScore | supported | FocusScore | 35 | +0.9904 | +0.9515 | +1.0278 | 1.00 |
| freq_hrl_vs_hrl_raw_LowerLFDrift | not_supported | LowerLFDrift | 35 | +1.7486 | +1.6939 | +1.7948 | 0.00 |
| freq_hrl_vs_raw_history_sharpe | supported | sharpe | 35 | +6.3517 | +4.9035 | +7.8192 | 0.91 |
| freq_hrl_vs_raw_history_total_return | supported | total_return | 35 | +0.0556 | +0.0344 | +0.0774 | 0.83 |
| freq_hrl_vs_raw_history_FocusScore | supported | FocusScore | 35 | +0.8172 | +0.7708 | +0.8621 | 1.00 |
| freq_hrl_vs_raw_history_LowerLFDrift | not_supported | LowerLFDrift | 35 | +1.4253 | +1.3742 | +1.4679 | 0.00 |
| freq_hrl_vs_freq_single_policy_sharpe | supported | sharpe | 35 | +23.8301 | +21.6412 | +26.1806 | 1.00 |
| freq_hrl_vs_freq_single_policy_total_return | supported | total_return | 35 | +0.3120 | +0.2773 | +0.3486 | 1.00 |
| freq_hrl_vs_freq_single_policy_FocusScore | supported | FocusScore | 35 | +1.1376 | +1.0733 | +1.2016 | 1.00 |
| freq_hrl_vs_freq_single_policy_LowerLFDrift | not_supported | LowerLFDrift | 35 | +1.6607 | +1.6086 | +1.7043 | 0.00 |
| freq_hrl_vs_allfreq_alllayers_sharpe | supported | sharpe | 35 | +24.2308 | +22.2258 | +26.3570 | 1.00 |
| freq_hrl_vs_allfreq_alllayers_total_return | supported | total_return | 35 | +0.3334 | +0.2955 | +0.3737 | 1.00 |
| freq_hrl_vs_allfreq_alllayers_FocusScore | supported | FocusScore | 35 | +1.1387 | +1.0754 | +1.2023 | 1.00 |
| freq_hrl_vs_allfreq_alllayers_LowerLFDrift | not_supported | LowerLFDrift | 35 | +1.5782 | +1.5261 | +1.6218 | 0.00 |
| freq_hrl_vs_swapped_sharpe | supported | sharpe | 35 | +24.2032 | +22.1364 | +26.4356 | 1.00 |
| freq_hrl_vs_swapped_total_return | supported | total_return | 35 | +0.2810 | +0.2480 | +0.3132 | 1.00 |
| freq_hrl_vs_swapped_FocusScore | supported | FocusScore | 35 | +1.4160 | +1.3713 | +1.4619 | 1.00 |
| freq_hrl_vs_swapped_LowerLFDrift | not_supported | LowerLFDrift | 35 | +1.5763 | +1.5229 | +1.6196 | 0.00 |
| freq_hrl_vs_no_promotion_sharpe | inconclusive | sharpe | 35 | +0.0151 | -0.0513 | +0.1063 | 0.14 |
| freq_hrl_vs_no_promotion_total_return | inconclusive | total_return | 35 | +0.0001 | -0.0006 | +0.0011 | 0.14 |
| freq_hrl_vs_no_promotion_FocusScore | inconclusive | FocusScore | 35 | +0.0080 | -0.0015 | +0.0186 | 0.26 |
| freq_hrl_vs_no_promotion_LowerLFDrift | supported | LowerLFDrift | 35 | -0.0153 | -0.0262 | -0.0063 | 0.46 |
| freq_hrl_vs_no_leakage_sharpe | supported | sharpe | 35 | +2.4905 | +1.8799 | +3.1379 | 0.94 |
| freq_hrl_vs_no_leakage_total_return | supported | total_return | 35 | +0.0227 | +0.0176 | +0.0281 | 0.97 |
| freq_hrl_vs_no_leakage_FocusScore | supported | FocusScore | 35 | +0.1509 | +0.1412 | +0.1601 | 1.00 |
| freq_hrl_vs_no_leakage_LowerLFDrift | not_supported | LowerLFDrift | 35 | +0.8730 | +0.8293 | +0.9100 | 0.00 |
| freq_hrl_vs_lf_upper_only_sharpe | inconclusive | sharpe | 35 | +0.0151 | -0.0513 | +0.1063 | 0.14 |
| freq_hrl_vs_lf_upper_only_total_return | inconclusive | total_return | 35 | +0.0001 | -0.0006 | +0.0011 | 0.14 |
| freq_hrl_vs_lf_upper_only_FocusScore | inconclusive | FocusScore | 35 | +0.0080 | -0.0015 | +0.0186 | 0.26 |
| freq_hrl_vs_lf_upper_only_LowerLFDrift | supported | LowerLFDrift | 35 | -0.0153 | -0.0262 | -0.0063 | 0.46 |
| freq_hrl_vs_hf_lower_only_sharpe | supported | sharpe | 35 | +18.0472 | +15.9902 | +20.0896 | 1.00 |
| freq_hrl_vs_hf_lower_only_total_return | supported | total_return | 35 | +0.1698 | +0.1329 | +0.2071 | 1.00 |
| freq_hrl_vs_hf_lower_only_FocusScore | supported | FocusScore | 35 | +0.9220 | +0.8829 | +0.9608 | 1.00 |
| freq_hrl_vs_hf_lower_only_LowerLFDrift | not_supported | LowerLFDrift | 35 | +0.9254 | +0.8711 | +0.9737 | 0.00 |
