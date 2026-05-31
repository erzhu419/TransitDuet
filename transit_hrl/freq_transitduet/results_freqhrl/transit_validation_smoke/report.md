# Transit Validation Smoke Report

This is the real Transit simulator validation path under `transit_hrl`.
It runs the copied `freq_transitduet` environment/runner, with the copied
frequency entry point bridged to `freq_hrl.domains.transit.TransitFrequencyTracker`.

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/erzhu419/mine_code/TransitDuet/transit_hrl FREQDUET_TORCH_THREADS=1 python3 scripts/run_freqduet_ablation.py --configs T_nofreq_terminal,T_rawhistory_terminal,T_lf_upper_terminal,T_hf_lower_terminal,T_freqhrl_terminal,T_allfreq_terminal,T_swapped_terminal,T_nopromotion_terminal,T_noleakage_terminal --seeds 42 --episodes 1 --last-k 1 --logs-dir logs_freqhrl_validation --out-dir results_freqhrl/transit_validation_smoke --upper-warmup-eps 0 --worker-threads 1 --clean
```

Scope:

- 1 seed: `42`
- 1 episode per config
- upper warmup forced to `0`
- real bus simulator, Excel route/OD/timetable data, learned lower/upper networks initialized and updated
- not enough for statistical performance claims

Summary:

| config | wait min | headway CV | composite | UpperHFPower | LowerLFDrift | attr | MI | shock response s | promotion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T_nofreq_terminal | 5.40 | 0.492 | 1.032 | 0.037 | 0.986 | 0.000 | 0.000 | 0.0 | 0.000 |
| T_rawhistory_terminal | 5.53 | 0.556 | 1.109 | 0.048 | 0.986 | 0.143 | 0.048 | 0.0 | 0.000 |
| T_lf_upper_terminal | 5.50 | 0.418 | 0.968 | 0.075 | 0.986 | -0.001 | -0.018 | 4.3 | 0.000 |
| T_hf_lower_terminal | 5.52 | 0.561 | 1.114 | 0.051 | 0.986 | -0.308 | -0.005 | 5.1 | 0.000 |
| T_freqhrl_terminal | 5.45 | 0.583 | 1.128 | 0.045 | 0.986 | 0.007 | 0.023 | 8.9 | 0.000 |
| T_allfreq_terminal | 5.39 | 0.616 | 1.155 | 0.056 | 0.986 | -0.191 | 0.010 | 10.6 | 0.000 |
| T_swapped_terminal | 5.58 | 0.448 | 1.068 | 0.054 | 0.986 | -0.180 | 0.033 | 4.8 | 0.000 |
| T_nopromotion_terminal | 5.47 | 0.655 | 1.264 | 0.046 | 0.986 | -0.181 | 0.008 | 19.7 | 0.000 |
| T_noleakage_terminal | 5.51 | 0.542 | 1.156 | 0.043 | 0.986 | -0.108 | -0.011 | 4.5 | 0.000 |

Interpretation:

- The real Transit validation path and 9-config baseline matrix are operational under `transit_hrl`.
- The copied runner/env did not modify the original `FreqDuet/` or `transit_duet/`.
- The current single-episode pilot is only an integration check; multi-seed and multi-episode runs are still needed.
- Promotion did not trigger in this short pilot, so it is still not validated in Transit.
