# FreqDuet Progress Log

## 2026-05-30

Validated main-path modules:

- Harmonic prior demand decomposer remains the effective causal intensity path:
  `harmonic_prior` had the best synthetic low-frequency RMSE in
  `scripts/eval_frequency_modules.py --seed 7`.
- Target-headway timetable planner remains the Phase 3 main path. The default
  path does not change terminal launch time.
- Stronger lower drift leakage is effective and has been promoted into
  `F_freqduet_timetable_hiro`.

5-seed protocol:

```text
configs: F_freqduet_timetable_hiro,
         F_freqduet_timetable_drift_tight_hiro,
         F_freqduet_timetable_drift_strong_hiro
seeds:   42,123,456,789,2026
episodes: 20
last_k: 10
warmup: upper_warmup_eps=10
```

Aggregate:

```text
baseline timetable: wait=8.85, cv=0.497, comp=2.023, lower_action=16.22
drift tight:        wait=7.39, cv=0.506, comp=1.840, lower_action=13.70
drift strong:       wait=6.56, cv=0.448, comp=1.723, lower_action=7.04
```

Experimental modules not yet promoted:

- Terminal dispatch: 3-seed looked promising, but 5-seed was unstable
  (`comp=1.689±0.421`), so it remains an experimental config.
- Promotion gate: state-only and conservative replan variants did not beat the
  current timetable path in 3-seed validation, so it remains experimental.
