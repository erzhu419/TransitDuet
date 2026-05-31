# Trading Performance Validation

Synthetic high-cost noisy market with low-frequency alpha, high-frequency shocks, and a persistent regime shift.

- seeds: [42, 123, 456, 789, 2026]
- bars per seed: 720
- scenario: `persistent_shift`
- frequency encoder: `ema`
- promotion config: threshold=0.00035, persistence_ratio=0.5, cooldown_s=600.0, regime_threshold=3e-05, min_age_s=0.0, activation_strength_threshold=0.0, startup_strength_age_s=0.0, startup_strength_threshold=0.0, mid_gain=0.5, adapt_gain=0.05
- HF lower config: residual_gain=0.0, recenter_gain=0.0, speed_gain=0.0, energy_speed_gain=0.0
- leakage reward scale: 5e-05
- promotion adaptation cost scale: 5e-05
- policies are deterministic heuristics, not trained RL policies
- task metrics include return, Sharpe, drawdown, turnover, transaction cost, and inventory drift
- frequency diagnostics include UpperHFPower, LowerLFDrift, FocusScore, PromotionDelay, ShockResponseTime, regime-promotion accuracy, and recovery cost

## Headline

Best Sharpe baseline: `freq_hrl` (16.062 +/- 1.513).
`freq_hrl`: Sharpe 16.062, return 0.2436, max drawdown 0.0093, turnover 5.76, FocusScore 0.905.
Against `no_promotion`, tuned promotion changes Sharpe by +0.016, return by -0.0003, and post-shift-120 PnL by -0.00013.

## Interpretation

- Frequency routing is useful in this validation: `freq_hrl` beats raw-history, all-frequency, swapped, no-promotion, and no-leakage baselines on Sharpe. Against `lf_upper_only`, the current gain is a small Sharpe edge rather than clean return dominance.
- `allfreq_alllayers` and `hrl_raw` overtrade heavily under noisy high-frequency shocks, which is visible in turnover and transaction cost.
- `lf_upper_only` remains close on this synthetic task, so the incremental value of HF lower control should be claimed as modest and scenario-dependent.
- `swapped` has negative FocusScore and poor task metrics, supporting the direction of the LF-to-upper / HF-to-lower assignment.
- Promotion is now a small positive contributor on headline Sharpe, but the return and immediate post-shift deltas remain effectively flat in this conservative setting.
- Leakage regularization is applied online to the reward signal sent to learners; `no_leakage` disables this shaping path.
- Reward attribution splits each episode into LF cost, HF cost, leakage cost, and promotion adaptation cost for credit diagnostics.
- The immediate post-shift 120-bar window is still not improved by the best Sharpe setting, so promotion should be claimed as task-positive here, not as fully optimized shock recovery.

## Summary Table

| baseline | return | Sharpe | shaped reward | LF cost | HF cost | leak cost | promo cost | max DD | turnover | cost | post_shift_120 | recovery_cost | PromotionDelay | ShockResponse | promo_acc | UpperHFPower | LowerLFDrift | FocusScore | promotions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla_rl | -0.8164 | -37.200 | -0.002387 | 0.00000578 | 0.00296201 | 0.00002606 | 0.00000000 | 0.8157 | 335.99 | 2.0159 | -0.17144 | 0.17925 | -1.0 | 0.0 | 0.500 | 0.3978 | 0.008 | -0.013 | 0.0 |
| hrl_raw | -0.6210 | -31.253 | -0.001406 | 0.00000287 | 0.00196581 | 0.00002649 | 0.00000000 | 0.6201 | 214.83 | 1.2890 | -0.07466 | 0.08992 | -1.0 | 0.0 | 0.500 | 0.3978 | 0.013 | -0.011 | 0.0 |
| raw_history | 0.2028 | 11.547 | 0.000235 | 0.00000029 | 0.00039274 | 0.00002044 | 0.00000000 | 0.0320 | 28.37 | 0.1702 | 0.07874 | 0.00364 | -1.0 | 0.4 | 0.500 | 0.0196 | 0.333 | 0.150 | 0.0 |
| freq_single_policy | -0.0777 | -4.113 | -0.000127 | 0.00000071 | 0.00078870 | 0.00001351 | 0.00000000 | 0.1835 | 75.12 | 0.4507 | 0.04887 | 0.01244 | -1.0 | 0.0 | 0.500 | 0.0753 | 0.099 | -0.122 | 0.0 |
| lf_upper_only | 0.2439 | 16.047 | 0.000207 | 0.00000433 | 0.00016922 | 0.00009607 | 0.00000000 | 0.0092 | 5.71 | 0.0343 | 0.06032 | 0.00143 | -1.0 | 59.7 | 0.500 | 0.0007 | 1.831 | 0.880 | 0.0 |
| hf_lower_only | -0.0044 | -1.596 | -0.000053 | 0.00000043 | 0.00010923 | 0.00003958 | 0.00000000 | 0.0157 | 8.80 | 0.0528 | 0.01142 | 0.00186 | -1.0 | 13.5 | 0.500 | 0.0000 | 0.856 | 0.057 | 0.0 |
| allfreq_alllayers | -0.1013 | -5.182 | -0.000168 | 0.00000080 | 0.00082771 | 0.00001893 | 0.00000008 | 0.1974 | 79.73 | 0.4784 | 0.04593 | 0.01421 | 23.4 | 0.0 | 0.482 | 0.0754 | 0.180 | -0.126 | 21.4 |
| swapped | -0.0881 | -6.972 | -0.000154 | 0.00000189 | 0.00057286 | 0.00001715 | 0.00000008 | 0.1098 | 52.37 | 0.3142 | 0.03817 | 0.01230 | 23.4 | 0.0 | 0.482 | 0.1089 | 0.197 | -0.422 | 21.4 |
| no_promotion | 0.2439 | 16.047 | 0.000207 | 0.00000433 | 0.00016922 | 0.00009607 | 0.00000000 | 0.0092 | 5.71 | 0.0343 | 0.06032 | 0.00143 | -1.0 | 59.7 | 0.500 | 0.0007 | 1.831 | 0.880 | 0.0 |
| no_leakage | 0.2234 | 14.734 | 0.000278 | 0.00000228 | 0.00022230 | 0.00000000 | 0.00000008 | 0.0166 | 11.89 | 0.0714 | 0.06461 | 0.00192 | 23.4 | 3.1 | 0.482 | 0.0008 | 0.912 | 0.756 | 21.4 |
| freq_hrl | 0.2436 | 16.062 | 0.000207 | 0.00000432 | 0.00016808 | 0.00009561 | 0.00000008 | 0.0093 | 5.76 | 0.0346 | 0.06018 | 0.00143 | 23.4 | 59.7 | 0.482 | 0.0008 | 1.821 | 0.905 | 21.4 |

## Current Validation Boundary

This is performance validation for the Freq-HRL protocol on a controlled synthetic trading task. It is not yet learned-policy validation, and it is not yet a TransitDuet simulator training result.
