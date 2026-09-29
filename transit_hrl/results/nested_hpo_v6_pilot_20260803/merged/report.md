# Nested-Validation Hyperparameter Pilot

- protocol: `nested_validation_hpo_v1`
- coverage: `complete`
- held-out test accesses: `0`

| policy | candidate | rank | robust score | mean utility |
|---|---|---:|---:|---:|
| freq_hrl | ppo_lr1e3_std10 | 1 | +0.08097957 | +0.08489739 |
| freq_hrl | ppo_lr1e3_std15 | 2 | +0.07832637 | +0.08269187 |
| freq_hrl | ppo_lr3e4_std10 | 3 | +0.07063980 | +0.07228330 |
| freq_hrl | ppo_lr3e4_std05 | 4 | +0.06907788 | +0.07067378 |
| freq_hrl | ppo_lr3e4_std15 | 5 | +0.06837290 | +0.07071634 |
| freq_hrl | ppo_lr1e4_std10 | 6 | +0.01951267 | +0.02002317 |
| freq_hrl | ppo_lr1e4_std15 | 7 | +0.01945568 | +0.01996113 |
| freq_hrl | ppo_lr1e4_std05 | 8 | +0.01941609 | +0.01994107 |
| flat_ppo | ppo_lr3e4_std05 | 1 | -0.00771379 | +0.00386272 |
| flat_ppo | ppo_lr3e4_std10 | 2 | -0.00771379 | -0.00606728 |
| flat_ppo | ppo_lr1e4_std05 | 3 | -0.00771379 | -0.00613195 |
| flat_ppo | ppo_lr1e4_std10 | 4 | -0.00771379 | -0.00701395 |
| flat_ppo | ppo_lr1e4_std15 | 5 | -0.00779316 | -0.00762588 |
| flat_ppo | ppo_lr1e3_std10 | 6 | -0.00851658 | -0.00813199 |
| flat_ppo | ppo_lr3e4_std15 | 7 | -0.00851658 | -0.00813199 |
| flat_ppo | ppo_lr1e3_std15 | 8 | -0.00851658 | -0.00813199 |
| generic_hrl_ppo | ppo_lr1e3_std10 | 1 | +0.07183045 | +0.07440965 |
| generic_hrl_ppo | ppo_lr1e3_std15 | 2 | +0.06852475 | +0.07065950 |
| generic_hrl_ppo | ppo_lr3e4_std05 | 3 | +0.02166878 | +0.02402516 |
| generic_hrl_ppo | ppo_lr3e4_std10 | 4 | +0.02068889 | +0.02326971 |
| generic_hrl_ppo | ppo_lr3e4_std15 | 5 | +0.01973087 | +0.02247204 |
| generic_hrl_ppo | ppo_lr1e4_std05 | 6 | +0.00401617 | +0.00439329 |
| generic_hrl_ppo | ppo_lr1e4_std10 | 7 | +0.00388407 | +0.00428269 |
| generic_hrl_ppo | ppo_lr1e4_std15 | 8 | +0.00373538 | +0.00416456 |
| flat_sac | off_lr1e4_w2048_b64 | 1 | +0.04850113 | +0.04975550 |
| flat_sac | off_lr3e4_w4096_b64 | 2 | +0.04761029 | +0.04978679 |
| flat_sac | off_lr1e4_w1024_b64 | 3 | +0.04706393 | +0.05136139 |
| flat_sac | off_lr1e3_w4096_b64 | 4 | +0.04217351 | +0.04948737 |
| flat_sac | off_lr3e4_w2048_b64 | 5 | +0.04036441 | +0.04904103 |
| flat_sac | off_lr1e3_w2048_b64 | 6 | +0.04000692 | +0.04558178 |
| flat_sac | off_lr3e4_w1024_b64 | 7 | +0.03557329 | +0.04466069 |
| flat_sac | off_lr3e4_w2048_b128 | 8 | +0.03464869 | +0.04152596 |
| flat_td3 | off_lr3e4_w1024_b64 | 1 | +0.02225188 | +0.03496448 |
| flat_td3 | off_lr3e4_w4096_b64 | 2 | +0.01970281 | +0.03441065 |
| flat_td3 | off_lr1e4_w1024_b64 | 3 | +0.01934711 | +0.03151822 |
| flat_td3 | off_lr1e3_w4096_b64 | 4 | +0.01010630 | +0.03111236 |
| flat_td3 | off_lr1e4_w2048_b64 | 5 | +0.00658922 | +0.03139657 |
| flat_td3 | off_lr3e4_w2048_b64 | 6 | +0.00516067 | +0.02523561 |
| flat_td3 | off_lr3e4_w2048_b128 | 7 | +0.00404387 | +0.02676930 |
| flat_td3 | off_lr1e3_w2048_b64 | 8 | -0.00976938 | +0.01792718 |
