# Native Promotion v32 Compact Evidence

v32 supports global wait/score and reward noninferiority; reward improvement remains not_supported.

- source: `transit_hrl/results/transit_native_promotion_reward_guarded_highpressure_wait_v32_512seed_w16_merged/summary.json`
- seeds: `512`
- rows: `1024`
- min_pairs: `32`

| metric | check | status | delta | ci95 low | ci95 high | n |
|---|---|---:|---:|---:|---:|---:|
| ep_reward | native_wait_aware_replan_vs_interval_ep_reward | not_supported | -0.7324414062 | -2.153562646 | 0.2361768066 | 512 |
| avg_wait_min | native_wait_aware_replan_vs_interval_avg_wait_min | supported | -0.00021484375 | -0.0005176269531 | -4.4921875e-05 | 512 |
| score | native_wait_aware_replan_vs_interval_score | supported | 0.0002140625 | 4.4921875e-05 | 0.0005145214844 | 512 |
| shared_ppo_target_headway_project_count | native_wait_aware_replan_vs_interval_shared_ppo_target_headway_project_count | supported | 0.015625 | 0.005859375 | 0.02734375 | 512 |
| shared_ppo_active_target_headway_floor_rejects | native_wait_aware_replan_vs_interval_shared_ppo_active_target_headway_floor_rejects | supported | 0.009765625 | 0.001953125 | 0.01953125 | 512 |
| shared_ppo_wait_replan_count | native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_count | supported | 0.017578125 | 0.0078125 | 0.029296875 | 512 |
| ep_reward | native_wait_aware_replan_vs_interval_ep_reward_noninferiority | supported | -0.7324414062 | -2.153562646 | 0.2361768066 | 512 |
| avg_wait_min | native_wait_aware_replan_vs_interval_avg_wait_min_noninferiority | supported | -0.00021484375 | -0.0005176269531 | -4.4921875e-05 | 512 |
