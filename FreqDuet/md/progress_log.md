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

Follow-up lower action cap sweep:

```text
base strong drift, 4 hard seeds: comp=1.787±0.300
cap30:                         comp=1.589±0.323
cap40:                         comp=1.584±0.243
cap45:                         comp=1.545±0.149
cap50:                         comp=1.639±0.340
cap45, full 5 seeds:           wait=5.13±0.56, cv=0.477±0.041, comp=1.529±0.137
```

The 45s lower action range is promoted into `F_freqduet_timetable_hiro`
because it removes the `seed2026` long-tail failure without the cap40 slowdown
or cap50 drift relapse.

Performance and prior-alignment follow-up:

```text
cProfile main cap45 ep1:        24.329s, 49.5M calls
after frequency-cache/RLS opt:   21.764s, 42.8M calls
time-aligned harmonic prior 5s:  wait=5.84±0.62, cv=0.476±0.023, comp=1.527±0.097
```

The harmonic prior now uses the global service-day bin when updating local and
OD states, instead of each state starting its Fourier clock at first arrival.
That preserves the historical 6:00-19:00 prior alignment. A second cap sweep
under the aligned prior did not beat the 45s main setting:

```text
cap30: wait=5.60±0.43, comp=1.583±0.090
cap40: wait=6.35±1.24, comp=1.585±0.344
cap50: wait=6.53±0.47, comp=1.688±0.156
```

SUMO exp39-style lower stabilization:

```text
main target-headway, 5 seeds:       wait=5.84±0.62, cv=0.476±0.023, comp=1.527±0.097
disc5, 4 hard seeds:                wait=5.53±0.25, cv=0.448±0.015, comp=1.498±0.118
disc7, 4 hard seeds:                wait=7.82±3.12, cv=0.470±0.038, comp=1.887±0.395
disc7 + last action, 5 seeds:             wait=5.55±0.35, cv=0.448±0.017, comp=1.438±0.154
pre-discrete terminal current, 5 seeds:   wait=6.02±0.82, cv=0.478±0.041, comp=1.659±0.192
pre-discrete terminal hold30, 5 seeds:    wait=6.76±1.86, cv=0.472±0.035, comp=1.658±0.335
pre-discrete terminal hold60, 5 seeds:    wait=6.67±1.63, cv=0.452±0.031, comp=1.673±0.206
post-discrete terminal current, 5 seeds:  wait=5.73±0.60, cv=0.444±0.013, comp=1.556±0.172
post-discrete terminal hold30, 5 seeds:   wait=5.13±0.19, cv=0.438±0.009, comp=1.347±0.127
post-discrete terminal hold60, 5 seeds:   wait=5.59±0.33, cv=0.434±0.014, comp=1.383±0.060
terminal promotion state, 3 seeds:        wait=6.01±0.78, cv=0.422±0.007, comp=1.578±0.147
terminal promotion replan, 3 seeds:       wait=5.80±0.66, cv=0.441±0.022, comp=1.535±0.091
terminal promotion trigger, 5 seeds:      wait=5.82±0.42, cv=0.427±0.021, comp=1.417±0.139
fixed 360s + rule lower, 5 seeds:         wait=8.86±0.92, cv=0.521±0.029, comp=2.133±0.118
terminal upper disc5, 3 seeds:            wait=5.95±0.38, cv=0.477±0.009, comp=1.618±0.100
terminal upper disc9, 3 seeds:            wait=5.63±0.74, cv=0.472±0.017, comp=1.566±0.129
```

The lower controller now uses discrete holding bins
`[0, 5, 10, 15, 20, 30, 45]` and appends previous holding action to the lower
state. This follows the SUMO exp39 lesson that continuous online holding can
be too high-variance; reducing the action alphabet plus adding action history
made the online controller beat the previous target-headway main path. Terminal
dispatch became effective only after the lower stabilization; `F_freqduet_terminal_hiro`
now uses the no-early-launch hold30 setting.

The first terminal promotion re-test showed that exposing promotion gate
features directly to the policy state added variance. The useful variant is
trigger-only promotion: gate diagnostics still detect persistent high-frequency
residuals and can trigger rolling timetable replans, but `[flag, strength, age]`
are not appended to the upper/lower policy state. It improved the same-run
5-seed terminal rerun (`comp=1.580±0.214` to `1.417±0.139`), but is kept as an
experimental config until it beats the best hold30 terminal record.

The simple fixed-rule check does not currently explain the gap: fixed 360s
headway with the proportional lower rule was worse than FreqDuet under the same
short elastic-fleet 5-seed protocol. SUMO-style action discretization remains
useful for lower holding, but not for the current upper planner: the upper
action is already a single low-frequency scalar with EMA smoothing, and forcing
disc5/disc9 bins worsened the terminal pilot.

Experimental modules not yet promoted:

- Earlier terminal dispatch with early launch was unstable, but the current
  no-early-launch hold30 terminal timetable has been promoted after 5-seed
  validation.
- Promotion gate: trigger-only promotion is the current useful candidate;
  state-feature and conservative-trigger variants are not promoted.
