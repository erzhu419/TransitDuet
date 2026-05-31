# Trading Pressure-Test Matrix

- scenarios: ['persistent_shift', 'stationary_low_noise', 'stationary_high_noise', 'localized_burst', 'ood_period']
- seeds: [42, 123, 456, 789, 2026]
- bars per seed: 720
- policies are deterministic heuristics, not trained RL policies

## Scenario Winners

| scenario | best Sharpe | best return | Freq-HRL Sharpe | Freq-HRL return | LF-only Sharpe | NoPromotion Sharpe |
|---|---|---|---:|---:|---:|---:|
| localized_burst | lf_upper_only (1.877) | lf_upper_only (0.0156) | 1.282 | 0.0127 | 1.877 | 1.721 |
| ood_period | lf_upper_only (13.897) | lf_upper_only (0.2380) | 13.406 | 0.2327 | 13.897 | 13.685 |

## Interpretation

- This matrix checks whether the frequency-responsibility claim survives beyond the default persistent-shift setting.
- `lf_upper_only` is tracked explicitly because it is close to `freq_hrl` in the default validation.
- Rows where `freq_hrl` is not the scenario winner should drive the next policy or promotion tuning pass.
