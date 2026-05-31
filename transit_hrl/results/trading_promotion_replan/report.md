# Promotion Replan Validation

- seeds: [42, 123, 456, 789, 2026, 31415, 27182, 16180, 11235, 4242]
- scenario: `promotion_recovery`
- comparison: interval-only plan curve vs promotion-triggered forced replan
- forced replan count mean: 10.60
- return delta: +0.0014
- Sharpe delta: +0.075
- post-shift-120 PnL delta: +0.00075
- recovery-regret delta: -0.00075
- LowerLFDrift delta: -0.0326

| variant | return | Sharpe | post_shift_120 | recovery_regret | forced_replans | plan_reuse | LowerLFDrift |
|---|---:|---:|---:|---:|---:|---:|---:|
| interval_plan | 0.3164 | 19.411 | 0.03328 | 0.08480 | 0.00 | 0.798 | 0.983 |
| promotion_replan | 0.3178 | 19.485 | 0.03402 | 0.08405 | 10.60 | 0.789 | 0.950 |
