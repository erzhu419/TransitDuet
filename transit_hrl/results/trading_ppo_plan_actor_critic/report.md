# PPO Dual Actor-Critic Trading Validation

- trainer: `shared_dual_level_ppo`
- plan mode: `learned_bernstein`
- lower LF constraint: coef=0.0, target=0.0, dual_lr=0.0
- lower LF effect projector: window=0, gain=1.0
- raw lower drift recenter: gain=0.0, scale=0.1
- scenario: `persistent_shift`
- train seeds: [42, 123, 456, 789, 2026]
- eval seeds: [31415, 27182, 16180, 11235, 4242]
- return mean: 0.2991
- Sharpe mean: 14.750
- max drawdown mean: 0.0254
- turnover mean: 6.36
- leakage penalty mean: 1.5490
- LowerLFDrift mean: 1.5481
- LowerLFDriftAbs mean: 0.000034
- RawLowerLFDrift mean: 1.5481
- RawLowerLFDriftAbs mean: 0.000034
- raw recenter boost mean: 0.0000
- plan smoothness mean: 0.0000
- plan coefficient abs mean: 0.0792

This validates the shared upper/lower PPO actor-critic training core. It uses trading as a domain adapter; the trainer itself only depends on upper/lower states, latent actions, rewards, and done flags.
