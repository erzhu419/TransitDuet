# PPO Dual Actor-Critic Trading Validation

- trainer: `shared_dual_level_ppo`
- scenario: `persistent_shift`
- train seeds: [42, 123, 456, 789, 2026]
- eval seeds: [31415, 27182, 16180, 11235, 4242]
- return mean: 0.2351
- Sharpe mean: 15.402
- max drawdown mean: 0.0150
- turnover mean: 7.17
- leakage penalty mean: 1.7166
- LowerLFDrift mean: 1.7155

This validates the shared upper/lower PPO actor-critic training core. It uses trading as a domain adapter; the trainer itself only depends on upper/lower states, latent actions, rewards, and done flags.
