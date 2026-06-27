# Freq-HRL Baseline And Ablation Manifest

Date: 2026-06-27

Purpose: separate genuine frequency-responsibility gains from gains caused by more parameters, more history, or more features.

| baseline | tier | headline_status | sharpe_delta | return_delta | focus_delta | n_common | paper_role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vanilla_rl | current | supported | 54.2815 | 0.9734 | 0.9917 | 35 | main_table |
| hrl_raw | current | supported | 50.3703 | 0.7933 | 0.9904 | 35 | main_table |
| raw_history | current | supported | 6.3517 | 0.0556 | 0.8172 | 35 | main_table |
| freq_single_policy | current | supported | 23.8301 | 0.3120 | 1.1376 | 35 | main_table |
| allfreq_alllayers | current | supported | 24.2308 | 0.3334 | 1.1387 | 35 | main_table |
| swapped | current | supported | 24.2032 | 0.2810 | 1.4160 | 35 | main_table |
| no_promotion | current_with_native_override | partial | 0.0151 | 0.0001 | 0.0080 | 35 | main_table |
| no_leakage | current | supported | 2.4905 | 0.0227 | 0.1509 | 35 | main_table |
| lf_upper_only | current_boundary | partial | 0.0151 | 0.0001 | 0.0080 | 35 | main_table |
| hf_lower_only | current_boundary | supported | 18.0472 | 0.1698 | 0.9220 | 35 | main_table |
| flat_ppo | registered_strong_learned | missing |  |  |  |  | must_complete_or_limit |
| flat_sac | registered_strong_learned | missing |  |  |  |  | must_complete_or_limit |
| flat_td3 | registered_strong_learned | missing |  |  |  |  | must_complete_or_limit |
| generic_hrl_ppo | registered_strong_learned | missing |  |  |  |  | must_complete_or_limit |

## Main-Table Rule

The manuscript main table should include all `current` rows. `registered_strong_learned` rows (flat PPO/SAC/TD3 and generic learned HRL) must either be completed with paired evidence or explicitly moved to limitations; they cannot be silently counted as supported.
