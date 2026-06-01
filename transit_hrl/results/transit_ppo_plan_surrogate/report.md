# Transit Surrogate PPO Validation

- trainer: `shared_dual_level_ppo`
- domain: `transit_surrogate`
- tracker method: `ema`
- plan mode: `learned_bernstein`
- upper decision interval: 1 steps, promotion forced replan=False
- promotion gate: residual_threshold=1.5, persistence_ratio=0.35
- wait attribution weights: upper=0.0, lower=0.0, board_credit=0.0
- wait credit control gain: 0.0
- lower LF constraint: coef=0.0, target=0.0, dual_lr=0.0
- lower LF effect projector: window=0, gain=1.0
- scenario: `persistent_shift`
- train seeds: [11, 23, 37, 41, 53]
- eval seeds: [101, 131, 151, 181, 211]
- reward mean: -4.4967
- wait proxy mean: 4.1992
- headway CV mean: 0.0746
- hold mean: 7.44
- leakage penalty mean: 1.0022
- LowerLFDrift mean: 1.0007
- RawLowerLFDrift mean: 1.0007
- plan smoothness mean: 0.0014
- plan coefficient abs mean: 7.8340
- wait high-share mean: 0.0426
- wait attribution penalty mean: 0.0000
- wait credit relief mean: 0.0000
- promotion replan count mean: 0.00

This uses the same `freq_hrl.rl.train_dual_ppo` loop as the trading PPO validation, with Transit frequency features and a transit-control surrogate adapter.
