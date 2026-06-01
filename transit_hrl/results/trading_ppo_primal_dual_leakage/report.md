# PPO Dual Actor-Critic Trading Validation

- trainer: `shared_dual_level_ppo`
- plan mode: `learned_bernstein`
- lower LF constraint: coef=0.08, target=0.2, dual_lr=0.5
- lower LF effect projector: window=24, gain=1.0
- scenario: `persistent_shift`
- train seeds: [42, 123, 456, 789, 2026]
- eval seeds: [31415, 27182, 16180, 11235, 4242]
- return mean: 0.2991
- Sharpe mean: 14.750
- max drawdown mean: 0.0254
- turnover mean: 6.36
- leakage penalty mean: 0.7707
- LowerLFDrift mean: 0.7699
- RawLowerLFDrift mean: 1.5481
- plan smoothness mean: 0.0000
- plan coefficient abs mean: 0.0792

This validates the shared upper/lower PPO actor-critic training core. It uses trading as a domain adapter; the trainer itself only depends on upper/lower states, latent actions, rewards, and done flags.
