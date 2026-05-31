# Transit Freq-HRL Gap-Closure Matrix

| variant | objective | delta obj | reward | wait | LowerLFDrift | upper decisions | promotion replans |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_ema_direct | -4.8869 | +0.0000 | -4.5625 | 4.2035 | 0.9993 | 96.0 | 0.0 |
| plan_only | -4.9404 | -0.0536 | -4.6143 | 4.2628 | 0.9993 | 12.0 | 0.0 |
| full_no_wait | -4.9269 | -0.0400 | -4.5989 | 4.2447 | 0.9986 | 54.5 | 50.0 |
| full_freqhrl | -4.9349 | -0.0480 | -4.6067 | 4.2453 | 0.9974 | 54.5 | 50.0 |

The `full_freqhrl` row is the integrated claim path: dynamic harmonic NB demand state, learned Bernstein plan actions, low-frequency upper reuse, promotion-triggered learned replanning, native lower HF context, wait-attributed reward, and lower-drift constraint.
