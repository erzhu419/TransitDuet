# Trading Promotion Sweep

- seeds: [42, 123, 456, 789, 2026]
- bars per seed: 720
- compared baselines: `freq_hrl` vs `no_promotion`
- deltas are `freq_hrl - no_promotion`

## Best Sharpe Delta

| threshold | ratio | mid_gain | adapt_gain | Sharpe delta | return delta | post_shift_120 delta | promotion count | delay |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00035 | 0.40 | 0.50 | 0.25 | +0.212 | +0.0016 | -0.00013 | 99.6 | 20.8 |
| 0.00035 | 0.40 | 1.00 | 0.50 | +0.210 | +0.0026 | -0.00009 | 100.0 | 20.8 |
| 0.00035 | 0.40 | 1.00 | 0.25 | +0.210 | +0.0015 | -0.00013 | 99.6 | 20.8 |
| 0.00035 | 0.40 | 0.00 | 0.25 | +0.208 | +0.0015 | -0.00013 | 99.6 | 20.8 |
| 0.00035 | 0.40 | 0.50 | 0.50 | +0.206 | +0.0025 | -0.00009 | 100.0 | 20.8 |
| 0.00035 | 0.40 | 0.00 | 0.50 | +0.194 | +0.0025 | -0.00009 | 100.0 | 20.8 |
| 0.00035 | 0.40 | 0.00 | 0.10 | +0.119 | +0.0006 | -0.00006 | 88.2 | 20.8 |
| 0.00035 | 0.40 | 0.50 | 0.10 | +0.117 | +0.0006 | -0.00006 | 88.2 | 20.8 |
| 0.00035 | 0.40 | 1.00 | 0.10 | +0.109 | +0.0005 | -0.00007 | 88.2 | 20.8 |
| 0.00035 | 0.30 | 0.00 | 0.10 | +0.103 | +0.0008 | -0.00007 | 132.0 | 50.2 |

## Best Post-Shift Delta

| threshold | ratio | mid_gain | adapt_gain | Sharpe delta | return delta | post_shift_120 delta | promotion count | delay |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00035 | 0.20 | 0.00 | 0.10 | -0.001 | +0.0017 | +0.00039 | 239.6 | 117.4 |
| 0.00035 | 0.20 | 0.50 | 0.10 | -0.010 | +0.0016 | +0.00037 | 239.6 | 117.4 |
| 0.00035 | 0.20 | 1.00 | 0.10 | -0.031 | +0.0014 | +0.00035 | 239.6 | 117.4 |
| 0.00035 | 0.20 | 0.00 | 0.50 | -0.578 | -0.0001 | +0.00003 | 222.6 | 117.2 |
| 0.00035 | 0.20 | 0.50 | 0.50 | -0.545 | +0.0002 | +0.00002 | 222.6 | 117.2 |

## Interpretation

- The best headline configuration is `threshold=0.00035`, `ratio=0.40`, `mid_gain=0.5`, `adapt_gain=0.25`.
- Lower ratio settings improve the immediate post-shift window but reduce Sharpe, so the default chooses the task-metric tradeoff.
