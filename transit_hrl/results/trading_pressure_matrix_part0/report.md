# Trading Pressure-Test Matrix

- scenarios: ['persistent_shift', 'stationary_low_noise', 'stationary_high_noise', 'localized_burst', 'ood_period']
- seeds: [42, 123, 456, 789, 2026]
- bars per seed: 720
- policies are deterministic heuristics, not trained RL policies

## Scenario Winners

| scenario | best Sharpe | best return | Freq-HRL Sharpe | Freq-HRL return | LF-only Sharpe | NoPromotion Sharpe |
|---|---|---|---:|---:|---:|---:|
| persistent_shift | freq_hrl (16.060) | freq_hrl (0.2442) | 16.060 | 0.2442 | 16.047 | 15.848 |
| stationary_low_noise | lf_upper_only (8.460) | lf_upper_only (0.0252) | 7.924 | 0.0246 | 8.460 | 7.924 |
| stationary_high_noise | lf_upper_only (-0.765) | lf_upper_only (-0.0047) | -1.389 | -0.0122 | -0.765 | -1.103 |

## Interpretation

- This matrix checks whether the frequency-responsibility claim survives beyond the default persistent-shift setting.
- `lf_upper_only` is tracked explicitly because it is close to `freq_hrl` in the default validation.
- Rows where `freq_hrl` is not the scenario winner should drive the next policy or promotion tuning pass.
