# Trading Pressure-Test Matrix

- scenarios: ['persistent_shift', 'promotion_recovery', 'stationary_low_noise', 'stationary_high_noise', 'localized_burst', 'ood_period']
- seeds: [42, 123, 456, 789, 2026]
- bars per seed: 720
- promotion min age: 0.0
- promotion activation strength threshold: 0.0
- promotion startup strength age: 0.0
- promotion startup strength threshold: 0.0
- policies are deterministic heuristics, not trained RL policies

## Scenario Winners

| scenario | best Sharpe | best return | Freq-HRL Sharpe | Recovery-tuned Sharpe | LF-only Sharpe | NoPromotion Sharpe |
|---|---|---|---:|---:|---:|---:|
| persistent_shift | freq_hrl (16.062) | freq_hrl_recovery_tuned (0.2439) | 16.062 | 16.006 | 16.047 | 16.047 |
| promotion_recovery | freq_hrl_recovery_tuned (21.133) | raw_history (0.3664) | 20.606 | 21.133 | 20.621 | 20.621 |
| stationary_low_noise | lf_upper_only (8.440) | lf_upper_only (0.0251) | 8.440 | 8.440 | 8.440 | 8.440 |
| stationary_high_noise | freq_hrl (-0.679) | freq_hrl (-0.0036) | -0.679 | -1.788 | -0.770 | -0.770 |
| localized_burst | freq_hrl_recovery_tuned (2.312) | freq_hrl_recovery_tuned (0.0131) | 2.301 | 2.312 | 2.306 | 2.306 |
| ood_period | freq_hrl (13.901) | freq_hrl_recovery_tuned (0.2423) | 13.901 | 13.838 | 13.897 | 13.897 |

## Interpretation

- This matrix checks whether the frequency-responsibility claim survives beyond the default persistent-shift setting.
- `lf_upper_only` is tracked explicitly because it is close to `freq_hrl` in the default validation.
- `freq_hrl_recovery_tuned` is the shorter-window, stricter-regime-buffer promotion gate from the dedicated recovery validation.
- Rows where neither default nor recovery-tuned `freq_hrl` is the scenario winner should drive the next policy or promotion tuning pass.
