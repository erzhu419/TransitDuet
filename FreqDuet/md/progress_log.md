# FreqDuet Progress Log

## 2026-05-30

Validated main-path modules:

- Harmonic prior demand decomposer remains the effective causal intensity path:
  `harmonic_prior` had the best synthetic low-frequency RMSE in
  `scripts/eval_frequency_modules.py --seed 7`.
- Target-headway timetable planner remains the Phase 3 stable path. Phase 4
  terminal hold30 dispatch is the current promoted executable-timetable path.
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
terminal baseline main, 3 seeds:          wait=5.40±0.59, cv=0.453±0.009, comp=1.549±0.099
terminal baseline promotion, 3 seeds:     wait=5.38±0.24, cv=0.467±0.018, comp=1.612±0.066
terminal baseline nofreq, 3 seeds:        wait=5.93±0.30, cv=0.445±0.008, comp=1.527±0.174
terminal baseline rawhistory, 3 seeds:    wait=5.85±0.40, cv=0.464±0.020, comp=1.595±0.229
terminal baseline allfreq, 3 seeds:       wait=7.62±2.96, cv=0.457±0.064, comp=1.795±0.360
terminal baseline swapped, 3 seeds:       wait=8.13±3.38, cv=0.458±0.022, comp=1.942±0.483
terminal baseline lf-upper, 3 seeds:      wait=6.94±2.05, cv=0.449±0.028, comp=1.767±0.401
terminal baseline hf-lower, 3 seeds:      wait=5.81±0.47, cv=0.439±0.009, comp=1.675±0.160
terminal baseline no-leakage, 3 seeds:    wait=6.20±1.83, cv=0.504±0.021, comp=1.504±0.259
terminal close main, 5 seeds:             wait=5.38±0.46, cv=0.461±0.013, comp=1.501±0.099
terminal close nofreq, 5 seeds:           wait=6.06±0.32, cv=0.467±0.036, comp=1.663±0.214
terminal close no-leakage, 5 seeds:       wait=6.03±1.44, cv=0.477±0.039, comp=1.535±0.206
cProfile terminal ep1:                    19.95s, 43.8M calls; top hotspot harmonic updates
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

The terminal baseline family now uses the same executable terminal hold30 path
and lower stabilization for all reviewer-facing comparisons. In the 3-seed
pilot, AllFreq/Swapped/single-side frequency variants were clearly worse or
unstable. Nofreq and No-Leakage looked close enough to require 5-seed retesting;
the 5-seed close check restored the main path advantage. No-Leakage still
occasionally reduces overshoot, but it does so by allowing much larger lower
holding drift and higher wait variance, which is exactly the leakage mechanism
the method is meant to prevent.

The current profile shows that harmonic state updates dominate the remaining
non-neural-network overhead. The first safe optimization is scoped to baselines:
when a config has `od_features: false`, station-level updates no longer build OD
detail dictionaries and the frequency tracker no longer stores or updates OD
states. The promoted main path keeps OD features enabled, so its behavior is
unchanged.

Experimental modules not yet promoted:

- Earlier terminal dispatch with early launch was unstable, but the current
  no-early-launch hold30 terminal timetable has been promoted after 5-seed
  validation.
- Promotion gate: trigger-only promotion is the current useful candidate;
  state-feature and conservative-trigger variants are not promoted.

## 2026-05-30 step-1 reward attribution

Implemented actual passenger-wait frequency attribution as a candidate main
path in `F_freqduet_terminal_waitattr_hiro`. The simulator now records boarded
passenger waiting time at each stop; the runner assigns the low-frequency share
of that wait to upper timetable credit and the high-frequency local shock share
to lower holding reward shaping. The older `F_freqduet_terminal_hiro` is left
unchanged so reviewer baselines that extend it do not inherit the new reward.

Corrected same-run 5-seed check, 20 episodes, `upper_warmup_eps=10`, last 10
BiLevel episodes, workers=8:

```text
terminal main:      wait=5.84±0.62, cv=0.449±0.013, comp=1.590±0.170
terminal waitattr:  wait=5.66±0.70, cv=0.442±0.020, comp=1.501±0.162
```

The result is strong enough to use `F_freqduet_terminal_waitattr_hiro` as the
next main candidate for subsequent dev-manual modules. The lower wait penalty
is deliberately small (`~0.00145` mean) and the upper wait credit is zero-mean
within episode (`std=0.15`), so this adds frequency-attributed passenger-wait
credit without overwhelming the existing headway/cost learning signal.

## 2026-05-30 step-2 timetable curve action

Implemented direction-specific multi-coefficient timetable planning support.
`TimetableCurvePlanner` can now optionally write a rolling plan for both
directions at once (`plan_all_directions: true`), and active plan reuse can use
a global timetable-plan key. A slope penalty was added for 2-basis curves so
linear timetable actions are not unconstrained.

Corrected same-run protocol, 5 seeds, 20 episodes, `upper_warmup_eps=10`, last
10 BiLevel episodes:

```text
scalar waitattr:        wait=5.66±0.70, cv=0.442±0.020, comp=1.501±0.162, U_HF=0.090±0.019
spline3 all-dir:        wait=5.67±0.36, cv=0.471±0.019, comp=1.567±0.147, U_HF=0.040±0.007
spline2 all-dir r10:    wait=5.71±0.40, cv=0.455±0.013, comp=1.522±0.160, U_HF=0.054±0.011
spline2 per-dir:        wait=5.39±0.64, cv=0.441±0.012, comp=1.440±0.162, U_HF=0.041±0.005
```

Promote `F_freqduet_terminal_spline2dir_waitattr_hiro` as the next main
candidate. It keeps the upper as a direction-specific 2-basis timetable curve
while preserving the scalar path's per-direction event timing and upper decision
count. The all-direction variants are implemented and kept as ablations, but
are not promoted because they reduce upper decision count or add variance under
the current short training protocol.

## 2026-05-30 step-3 terminal schedule bounds

Tested fuller executable terminal schedule bounds on top of the promoted
`F_freqduet_terminal_spline2dir_waitattr_hiro` path. The planner implementation
already writes rolling scheduled launches; this sweep checks whether allowing
early release or wider terminal delay improves the executable schedule.

Corrected same-run protocol, 5 seeds, 20 episodes, `upper_warmup_eps=10`, last
10 BiLevel episodes:

```text
spline2dir hold30:      wait=5.39±0.64, cv=0.441±0.012, comp=1.440±0.162
release15/hold30:       wait=6.13±0.72, cv=0.438±0.011, comp=1.517±0.152
spline2dir hold45:      wait=5.62±0.35, cv=0.470±0.011, comp=1.513±0.169
spline2dir hold60:      wait=5.19±0.48, cv=0.452±0.033, comp=1.468±0.151
```

Do not promote early release: even a 15s early window raises wait and composite.
Hold60 improves mean wait but costs too much in CV/overshoot, and hold45 is
worse than hold30. Keep no-early-launch hold30 as the promoted executable
terminal schedule bound for the current main path. The release/hold45/hold60
configs remain as ablations documenting the operational-bound sweep.

## 2026-05-30 step-4 promotion gate v2

Implemented promotion-adaptive low-frequency absorption for the harmonic demand
state. When configured, a persistent residual gate can nudge the current
harmonic low-intensity estimate toward the sustained residual and logs
`freq_promotion_absorptions` / `freq_promotion_absorbed`. This completes the
dev-manual mechanism where persistent high-frequency shock can be promoted into
the low-frequency state, but it is only used by explicit configs.

Corrected same-run protocol, 5 seeds, 20 episodes, `upper_warmup_eps=10`, last
10 BiLevel episodes:

```text
spline2dir main:          wait=5.39±0.64, cv=0.441±0.012, comp=1.440±0.162
promotion adapt-low:      wait=5.43±0.64, cv=0.455±0.010, comp=1.485±0.170
promotion trigger:        wait=5.68±0.79, cv=0.437±0.026, comp=1.493±0.222
promotion conservative:   wait=5.59±0.36, cv=0.452±0.024, comp=1.342±0.080
```

Promote `F_freqduet_terminal_promotion_conservative_spline2dir_waitattr_hiro`
as the next main candidate. The useful mechanism is conservative trigger-only
replanning: it raises upper plan decisions moderately (about 95 vs 88 per
BiLevel episode tail) and lowers composite variance. Do not promote adapt-low
yet; it proves the high-to-low absorption mechanism is wired, but the current
gain (`adapt_gain=0.10`) over-adjusts under the short protocol.

## 2026-05-30 step-5 middle-band regime buffer

Implemented explicit middle-band residual tracking in the harmonic demand state.
The estimator now keeps a 5-30 minute residual EMA (`middle`,
`middle_energy`) alongside low/high components, and `low_mid` / `high_mid`
feature modes expose it to the upper/lower policies. Diagnostics now log
`freq_middle` and `freq_middle_energy`.

Corrected same-run protocol, 5 seeds, 20 episodes, `upper_warmup_eps=10`, last
10 BiLevel episodes:

```text
conservative main:       wait=5.59±0.36, cv=0.452±0.024, comp=1.342±0.080
middle shared U+L:       wait=6.53±1.91, cv=0.451±0.014, comp=1.608±0.344
middle upper-only:       wait=6.06±0.94, cv=0.468±0.038, comp=1.717±0.220
```

Do not promote middle-band state under the current short training protocol.
The mechanism is now implemented and visible for diagnostics/longer ablations,
but adding these state dimensions increases wait variance and degrades
composite. Keep `F_freqduet_terminal_promotion_conservative_spline2dir_waitattr_hiro`
as the current main candidate.
