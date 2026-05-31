# Transit Surrogate PPO Validation

- trainer: `shared_dual_level_ppo`
- domain: `transit_surrogate`
- scenario: `persistent_shift`
- train seeds: [11, 23, 37, 41, 53]
- eval seeds: [101, 131, 151, 181, 211]
- reward mean: -4.4601
- wait proxy mean: 4.2035
- headway CV mean: 0.0755
- hold mean: 7.56
- leakage penalty mean: 1.0003
- LowerLFDrift mean: 1.0000

This uses the same `freq_hrl.rl.train_dual_ppo` loop as the trading PPO validation, with Transit frequency features and a transit-control surrogate adapter.
