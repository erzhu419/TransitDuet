# Transit Surrogate PPO Validation

- trainer: `shared_dual_level_ppo`
- domain: `transit_surrogate`
- plan mode: `learned_bernstein`
- lower LF constraint: coef=0.08, target=0.2, dual_lr=0.5
- scenario: `persistent_shift`
- train seeds: [11, 23, 37, 41, 53]
- eval seeds: [101, 131, 151, 181, 211]
- reward mean: -4.4999
- wait proxy mean: 4.1999
- headway CV mean: 0.0753
- hold mean: 7.51
- leakage penalty mean: 1.0019
- LowerLFDrift mean: 1.0003
- plan smoothness mean: 0.0010
- plan coefficient abs mean: 7.8263

This uses the same `freq_hrl.rl.train_dual_ppo` loop as the trading PPO validation, with Transit frequency features and a transit-control surrogate adapter.
