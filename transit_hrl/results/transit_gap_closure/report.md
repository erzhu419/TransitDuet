# Transit Freq-HRL Gap-Closure Matrix

| variant | objective | delta obj | reward | wait | LowerLFDrift | upper decisions | promotion replans |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_ema_direct | -4.8686 | +0.0000 | -4.5443 | 4.2027 | 0.9995 | 96.0 | 0.0 |
| plan_only | -4.9396 | -0.0710 | -4.6132 | 4.2694 | 0.9993 | 12.0 | 0.0 |
| full_no_wait | -4.8766 | -0.0080 | -4.6113 | 4.2681 | 0.1983 | 24.3 | 13.3 |
| full_freqhrl | -4.7632 | +0.1054 | -4.5085 | 4.1431 | 0.2016 | 24.3 | 13.3 |

The `full_freqhrl` row is the integrated claim path: dynamic harmonic NB demand state, learned Bernstein plan actions, low-frequency upper reuse, promotion-triggered learned replanning, native lower HF context, wait-attributed reward, and lower-drift constraint.
