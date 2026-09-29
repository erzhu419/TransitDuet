# Nested-Validation Hyperparameter Pilot

- protocol: `nested_validation_hpo_v1`
- coverage: `complete`
- held-out test accesses: `0`

| policy | candidate | rank | robust score | mean utility |
|---|---|---:|---:|---:|
| freq_hrl | ppo_lr1e4_std15 | 1 | -0.00015749 | -0.00015749 |
| flat_ppo | ppo_lr1e4_std15 | 1 | -0.00239945 | -0.00239945 |
| flat_gru_ppo | ppo_lr1e4_std15 | 1 | -0.00031530 | -0.00031530 |
| generic_hrl_ppo | ppo_lr1e4_std15 | 1 | -0.00041678 | -0.00041678 |
| generic_hrl_gru_ppo | ppo_lr1e4_std15 | 1 | +0.00032015 | +0.00032015 |
