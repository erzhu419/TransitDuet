# Supplementary Material

## S1. Frozen evidence and decision ledger

The current paper package binds the successful V8 confirmation and failed V9
long-training gate. They use disjoint training and evaluation seeds and are not
pooled.

| Phase | Controller | Train seeds | Eval seeds | Pairs | Decision | Eligible |
| --- | --- | --- | --- | --- | --- | --- |
| v8_independent_confirmation_ep40 | F_freqduet_protocol_v6_confirmed_main_hiro | 6 | 4 | 24 | primary_confirmed | True |
| v9_independent_longtrain_ep200 | F_freqduet_protocol_v6_confirmed_main_hiro | 8 | 8 | 64 | longtrain_not_confirmed | False |

The term `confirmed_main` identifies the V8-selected configuration; it does not
mean that V9 passed. The machine-readable package remains
`submission_ready: false` with blocker `v9_longtrain_not_confirmed`.

## S2. Exact current-controller configuration

The current controller resolves through the following inheritance chain:

1. `config_v2.yaml`
2. `configs_freqduet/F_freqduet_harmonic_hiro.yaml`
3. `configs_freqduet/F_freqduet_timetable_hiro.yaml`
4. `configs_freqduet/F_freqduet_terminal_hiro.yaml`
5. `configs_freqduet/F_freqduet_terminal_waitattr_hiro.yaml`
6. `configs_freqduet/F_freqduet_terminal_spline2dir_waitattr_hiro.yaml`
7. `configs_freqduet/F_freqduet_terminal_aligned_nopromotion_hiro.yaml`
8. `configs_freqduet/F_freqduet_terminal_lowerhf_poswait_hiro.yaml`
9. `configs_freqduet/F_freqduet_terminal_lowerhf_localcap10_w06_hiro.yaml`
10. `configs_freqduet/F_freqduet_terminal_promotion_localcap10_w06_hiro.yaml`
11. `configs_freqduet/F_freqduet_terminal_energydenoise_hold45_hiro.yaml`
12. `configs_freqduet/F_freqduet_terminal_energydenoise_hold45_holdpen03_hiro.yaml`
13. `configs_freqduet/F_freqduet_terminal_energydenoise_hold45_holdpen03_strictprom_hiro.yaml`
14. `configs_freqduet/F_freqduet_terminal_energydenoise_hold45_holdpen03_adaptstrict_hiro.yaml`
15. `configs_freqduet/F_freqduet_terminal_energydenoise_hold45_holdpen03_adaptstrict_local_hiro.yaml`
16. `configs_freqduet/F_freqduet_protocol_v2_main_hiro.yaml`
17. `configs_freqduet/F_freqduet_protocol_v2_uppercompact_hiro.yaml`
18. `configs_freqduet/F_freqduet_protocol_v2_uppercompact_markov_hiro.yaml`
19. `configs_freqduet/F_freqduet_protocol_v2_uppercompact_markov_smdp_hiro.yaml`
20. `configs_freqduet/F_freqduet_protocol_v2_uppercompact_rebuild_nophys_hiro.yaml`
21. `configs_freqduet/F_freqduet_protocol_v2_uppercompact_rebuild_hiro.yaml`
22. `configs_freqduet/F_freqduet_protocol_v2_uppercompact_rebuild_strict_hiro.yaml`
23. `configs_freqduet/F_freqduet_protocol_v2_uppercompact_strict_intervaladd_hiro.yaml`
24. `configs_freqduet/F_freqduet_protocol_v3_compact_b30_hiro.yaml`
25. `configs_freqduet/F_freqduet_protocol_v4_main_hiro.yaml`
26. `configs_freqduet/F_freqduet_protocol_v5_main_hiro.yaml`
27. `configs_freqduet/F_freqduet_protocol_v6_main_hiro.yaml`
28. `configs_freqduet/F_freqduet_protocol_v6_noguard_hiro.yaml`
29. `configs_freqduet/F_freqduet_protocol_v6_avlcompact_hiro.yaml`
30. `configs_freqduet/F_freqduet_protocol_v6_avlcompact_w2_hiro.yaml`
31. `configs_freqduet/F_freqduet_protocol_v6_confirmed_main_hiro.yaml`

Key resolved settings are: harmonic historical prior; 60-s causal bins;
four Fourier harmonics over 14 h; 30-min low-frequency forecast; 15-min upper
replanning over 45 min; rolling zero-sum V6 headway budget; executable terminal
dispatch; low-frequency upper and high-frequency lower authority; compact
same-time APC/AVL lower context; previous-action state; holding actions
`{0, 5, 10, 15, 20, 30, 45}` s; and a weight-two pre-action two-sided
regularity reward. The legacy holding guard, promotion, and leakage penalty are
disabled.

## S3. Causal and physical contract

Passenger arrivals become visible only after their within-bin arrival times.
APC counts enter the frequency tracker only when a complete 60-s bin closes.
The lower regularity tuple freezes the matched predecessor departure, same-time
follower AVL estimate, target headway, and action before transition settlement.
The categorical action is executed without post-policy clipping. Commanded and
realized holding are recorded separately. The upper timetable materializes
future launch times once; cached reuse is read-only; each closed rolling budget
block conserves its headway adjustment; and actual launch cannot precede either
vehicle readiness or the executable scheduled time.

## S4. Complete V8 and V9 policy-reference outcomes

| Outcome | V8 delta [95% CI] | V8 Holm p | V9 delta [95% CI] | V9 Holm p |
| --- | --- | --- | --- | --- |
| Restricted passenger journey (min) | -0.263 [-0.837, +0.175] | 0.938 | -1.242 [-2.204, -0.536] | 0.047 |
| Restricted passenger wait (min) | -0.185 [-0.470, +0.050] | 0.375 | -1.011 [-1.878, -0.397] | 0.047 |
| Restricted in-vehicle time (min) | -0.078 [-0.376, +0.147] | 1.000 | -0.232 [-0.355, -0.121] | 0.047 |
| Headway coefficient of variation | -0.022 [-0.038, -0.008] | 0.125 | -0.009 [-0.028, +0.006] | 0.031 |
| Unserved passengers (percentage points) | +0.02 [-0.00, +0.11] | 1.000 | +0.00 [+0.00, +0.02] | 0.250 |
| Realized holding (s/launched trip) | -1.7 [-41.8, +29.4] | 1.000 | -37.6 [-59.8, -16.2] | 0.070 |
| Trips denied at least once (percentage points) | +0.48 [-9.21, +8.81] | 1.000 | -1.65 [-3.96, -0.22] | 0.117 |
| Restricted service cost | -0.040 [-0.079, -0.007] | 0.125 | -0.110 [-0.207, -0.042] | 0.047 |

All entries are current policy minus `F_freqduet_protocol_v6_noguard_hiro`. The V8 row corresponds to
the source config `F_freqduet_protocol_v6_avlcompact_w2_hiro` and the V9 row to its exact paper alias
`F_freqduet_protocol_v6_confirmed_main_hiro`. Both arms disable the legacy holding guard. The contrast
combines compact APC/AVL context with the incremental regularity objective.

## S5. Complete V9 external-baseline outcomes

| Baseline | Outcome | FreqDuet mean | Baseline mean | Delta [95% CI] | Holm p |
| --- | --- | --- | --- | --- | --- |
| Fixed headway | Restricted passenger journey (min) | +20.349 | +17.879 | +2.470 [+1.874, +3.188] | 0.023 |
| Fixed headway | Restricted passenger wait (min) | +8.009 | +7.001 | +1.009 [+0.439, +1.704] | 0.023 |
| Fixed headway | Restricted in-vehicle time (min) | +12.340 | +10.878 | +1.461 [+1.367, +1.542] | 0.023 |
| Fixed headway | Headway coefficient of variation | +0.213 | +0.432 | -0.219 [-0.238, -0.198] | 0.023 |
| Fixed headway | Unserved passengers (percentage points) | +0.14 | +0.21 | -0.07 [-0.20, -0.00] | 0.023 |
| Fixed headway | Realized holding (s/launched trip) | +286.8 | +0.0 | +286.8 [+269.7, +302.0] | 0.023 |
| Fixed headway | Trips denied at least once (percentage points) | +85.85 | +21.76 | +64.10 [+61.97, +66.03] | 0.023 |
| Fixed headway | Restricted service cost | +1.021 | +1.143 | -0.122 [-0.180, -0.050] | 0.023 |
| Rule holding | Restricted passenger journey (min) | +20.349 | +23.573 | -3.224 [-4.727, -1.885] | 0.023 |
| Rule holding | Restricted passenger wait (min) | +8.009 | +10.799 | -2.790 [-4.208, -1.519] | 0.023 |
| Rule holding | Restricted in-vehicle time (min) | +12.340 | +12.774 | -0.434 [-0.544, -0.335] | 0.023 |
| Rule holding | Headway coefficient of variation | +0.213 | +0.286 | -0.073 [-0.082, -0.062] | 0.023 |
| Rule holding | Unserved passengers (percentage points) | +0.14 | +0.13 | +0.01 [+0.00, +0.03] | 0.023 |
| Rule holding | Realized holding (s/launched trip) | +286.8 | +360.2 | -73.3 [-91.2, -57.3] | 0.023 |
| Rule holding | Trips denied at least once (percentage points) | +85.85 | +89.03 | -3.17 [-5.42, -1.90] | 0.023 |
| Rule holding | Restricted service cost | +1.021 | +1.372 | -0.351 [-0.496, -0.221] | 0.023 |
| Rule MPC | Restricted passenger journey (min) | +20.349 | +46.860 | -26.511 [-31.003, -21.628] | 0.023 |
| Rule MPC | Restricted passenger wait (min) | +8.009 | +33.527 | -25.517 [-29.945, -20.699] | 0.023 |
| Rule MPC | Restricted in-vehicle time (min) | +12.340 | +13.333 | -0.994 [-1.116, -0.881] | 0.023 |
| Rule MPC | Headway coefficient of variation | +0.213 | +0.207 | +0.006 [-0.006, +0.020] | 0.023 |
| Rule MPC | Unserved passengers (percentage points) | +0.14 | +0.05 | +0.09 [+0.00, +0.27] | 0.023 |
| Rule MPC | Realized holding (s/launched trip) | +286.8 | +409.6 | -122.7 [-139.7, -107.3] | 0.023 |
| Rule MPC | Trips denied at least once (percentage points) | +85.85 | +24.33 | +61.52 [+59.09, +63.77] | 0.023 |
| Rule MPC | Restricted service cost | +1.021 | +3.562 | -2.541 [-2.984, -2.060] | 0.023 |

Means and differences use the same outcome units shown in each row. The
denied-trip rate is the fraction of scheduled trips denied at least once by
the fixed-pool readiness check; retry duration is reported separately in the
source artifacts. A launch rate of one does not imply a zero denial rate.

## S6. Statistical procedure

Each train-seed policy is evaluated on every registered evaluation seed under
common random numbers. Crossed-bootstrap intervals resample training seeds and
a shared set of evaluation seeds, preserving policy pairing. Sign-flip tests
operate on train-seed mean deltas and therefore target conditional
training-seed inference rather than the crossed population represented by the
bootstrap. Holm adjustment is performed across compared methods separately for
each outcome. No V8 and V9 estimate is pooled, and no failed gate is rescued by
a favorable secondary endpoint.

The V8 gate required a headway-CV improvement of at least 0.02 against the
Protocol V6 reference, at least 0.01 against compact context alone, passenger
journey within +0.15 min of both references, complete paired rollouts, at least
50% same-time follower coverage, zero execution adjustment, and preservation
of holding and denied-dispatch gains. V9 reused those gates and additionally
required a headway-CV interval below zero, a journey CI upper bound no larger
than +0.15 min, and negative train-seed CV differences for at least 75% of
training seeds. V9 met the latter two added directional/no-harm conditions but
failed the inherited CV magnitude gates and CI-exclusion requirement.

## S7. Frequency-claim boundary and negative development evidence

The current confirmatory package does not contain a same-stage NoFreq,
RawHistory, AllFreq, swapped-layer, promotion, or leakage ablation. Earlier V6
engineering screens used some of these controls, but they were exploratory,
incomplete, or tied to superseded controller semantics. They cannot be used as
confirmatory evidence that frequency separation itself caused the V8 effect.
Likewise, V28-V32 counterfactual value/rank/margin planners all failed their
registered development gates and were not promoted. These negative results
define the stopping decision: the manuscript uses the frozen V6 controller and
does not tune another gate on the same development contexts.

## S8. External-data provenance

Figure 5 uses separately normalized demand shapes from the local FreqDuet OD
input, 39 complete station-complex days selected from a bounded public MTA AFC
cache, and seven complete routes spanning 37 route-days selected from a bounded
Halifax APC cache. Incomplete pagination fragments were excluded before
aggregation. MTA Bus Time is used only as route/stop and AVL audit data, not APC
or onboard load. MBTA boarding, alighting, and load data provide separate
calibration targets but are not a same-day matched calibration of the simulated
network. API credentials are not stored in the repository or paper package.

## S9. Reproducibility boundary

The paper evidence directory tracks the small V8/V9 CSV and JSON artifacts,
exact seed contracts, source commits, and resolved config lineage. Checkpoints
and full training logs remain on the HPC filesystem and are intentionally not
part of the manuscript package. A build validates the frozen package before
rendering any text or table. A successful build verifies consistency; it does
not change the scientific decision or create a field-deployment claim.
