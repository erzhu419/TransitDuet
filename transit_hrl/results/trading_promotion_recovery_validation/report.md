# Trading Promotion Recovery Validation

- scenario: `promotion_recovery`
- seeds: [42, 123, 456, 789, 2026, 31415, 27182, 16180, 11235, 4242, 7, 11, 19, 23, 29, 31, 37, 41, 43, 47]
- bars per seed: 720
- variants: `no_promotion`, `default_promotion`, `recovery_tuned`
- paired deltas are `variant - no_promotion`
- lower is better for recovery cost/regret, drawdown, turnover, and transaction cost
- `recovery_regret_120` is post-shift shortfall to an oracle low-frequency target; it is an evaluation diagnostic, not a learner input

## Headline

`recovery_tuned` vs `no_promotion`: Sharpe delta +0.430 (CI95 [+0.315, +0.536]), return delta +0.0115 (CI95 [+0.0087, +0.0145]).
Shock recovery: post-shift PnL delta +0.00854 (CI95 [+0.00622, +0.01090]), recovery-regret delta -0.00839 (CI95 [-0.01075, -0.00613]).
`recovery_tuned` mean promotion count is 55.9 with mean delay 16.6 bars; `default_promotion` count is 20.1 with delay 45.9 bars.

## Variant Summary

| variant | return | Sharpe | post_shift_120 | recovery_cost | recovery_regret | promotions | delay | FocusScore |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_promotion | 0.3390 | 20.368 | 0.04513 | 0.00635 | 0.07366 | 0.0 | -1.0 | 0.919 |
| default_promotion | 0.3388 | 20.370 | 0.04480 | 0.00646 | 0.07398 | 20.1 | 45.9 | 0.923 |
| recovery_tuned | 0.3505 | 20.799 | 0.05368 | 0.00601 | 0.06528 | 55.9 | 16.6 | 0.920 |

## Paired Deltas

| variant | Sharpe | return | post_shift_120 | recovery_cost | recovery_regret | post win | regret win |
|---|---:|---:|---:|---:|---:|---:|---:|
| default_promotion | +0.002 | -0.0003 | -0.00033 | +0.00010 | +0.00032 | 0.05 | 0.05 |
| recovery_tuned | +0.430 | +0.0115 | +0.00854 | -0.00034 | -0.00839 | 0.95 | 0.95 |

## Interpretation

- The default conservative promotion setting is intentionally cautious and does not improve the reversal-recovery window.
- The recovery-tuned gate uses a shorter persistence window plus a stricter mid-frequency regime buffer; it triggers earlier without the broad false-promotion behavior seen in low-threshold sweeps.
- On this controlled reversal shock, promotion now supports the recovery claim: it improves post-shift PnL and reduces oracle-regime recovery regret while also improving headline return and Sharpe.
- The original loss-only `recovery_cost_120` is retained as a risk diagnostic; the oracle-regime regret is the cleaner metric for adaptation to a new persistent low-frequency direction.
