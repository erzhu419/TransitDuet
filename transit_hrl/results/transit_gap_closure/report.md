# Transit Freq-HRL Gap-Closure Matrix

| variant | objective | delta obj | reward | wait | LowerLFDrift | upper decisions | promotion replans |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_ema_direct | -4.8686 | +0.0000 | -4.5443 | 4.2027 | 0.9995 | 96.0 | 0.0 |
| plan_only | -4.9396 | -0.0710 | -4.6132 | 4.2694 | 0.9993 | 12.0 | 0.0 |
| full_no_wait | -4.9406 | -0.0720 | -4.6113 | 4.2681 | 0.9989 | 24.3 | 13.3 |
| full_freqhrl | -4.8275 | +0.0411 | -4.5092 | 4.1431 | 0.9984 | 24.3 | 13.3 |

The `full_freqhrl` row is the integrated claim path: dynamic harmonic NB demand state, learned Bernstein plan actions, low-frequency upper reuse, promotion-triggered learned replanning, native lower HF context, wait-attributed reward, and lower-drift constraint.
