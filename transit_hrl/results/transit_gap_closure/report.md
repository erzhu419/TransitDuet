# Transit Freq-HRL Gap-Closure Matrix

| variant | objective | delta obj | reward | wait | LowerLFDrift | upper decisions | promotion replans |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_ema_direct | -4.8686 | +0.0000 | -4.5443 | 4.2027 | 0.9995 | 96.0 | 0.0 |
| plan_only | -4.9396 | -0.0710 | -4.6132 | 4.2694 | 0.9993 | 12.0 | 0.0 |
| full_no_wait | -4.8400 | +0.0287 | -4.5904 | 4.2520 | 0.4591 | 24.3 | 13.3 |
| full_freqhrl | -4.7794 | +0.0892 | -4.4878 | 4.1269 | 1.0896 | 24.3 | 13.3 |

The `full_freqhrl` row is the integrated claim path: dynamic harmonic NB demand state, learned Bernstein plan actions, low-frequency upper reuse, promotion-triggered learned replanning, native lower HF context, wait-attributed reward, and lower-drift constraint.
