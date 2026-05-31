# Transit Surrogate PPO Validation

- trainer: `shared_dual_level_ppo`
- domain: `transit_surrogate`
- plan mode: `learned_bernstein`
- scenario: `persistent_shift`
- train seeds: [11, 23, 37, 41, 53]
- eval seeds: [101, 131, 151, 181, 211]
- reward mean: -4.4997
- wait proxy mean: 4.1998
- headway CV mean: 0.0753
- hold mean: 7.50
- leakage penalty mean: 1.0018
- LowerLFDrift mean: 1.0002
- plan smoothness mean: 0.0016
- plan coefficient abs mean: 7.7939

This uses the same `freq_hrl.rl.train_dual_ppo` loop as the trading PPO validation, with Transit frequency features and a transit-control surrogate adapter.
