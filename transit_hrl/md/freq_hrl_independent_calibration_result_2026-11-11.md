# Independent Calibration Result And Next Decision

All eight sampling jobs t214476-t214483 exited 0: 64 new prefixes, 192 states,
1,536 conditional future pairs and exactly 4,224,000 environment steps. All four
A4-only cut-freeze jobs t214490-t214493 also exited 0, with zero simulation/fits.
All eight cuts are frozen. Measured sampling/selection CPU: 3,701.214/6.055 seconds.
Retrieved 100,006 public-result bytes; banks and models remain server-side.

## What Changed

B4 gains below are independent-cut minus original quantile on unchanged full scores;
positive W/G means better short-window/full-horizon ISE. R/C are the two label heads.
The old-calibration column holds scores, states and futures fixed. Now counts /48.

| Task / root | Head | B W gain | B G gain | B G versus old calibration | Now |
| --- | --- | ---: | ---: | ---: | ---: |
| PointMaze / 1972110000 | R | +0.001254 | +0.001266 | +0.000378 | 39 |
| PointMaze / 1972110000 | C | +0.001042 | +0.001121 | +0.002346 | 35 |
| PointMaze / 1972120000 | R | +0.005730 | -0.001136 | +0.021272 | 23 |
| PointMaze / 1972120000 | C | +0.005990 | -0.000991 | +0.021261 | 23 |
| Reacher / 1972210000 | R | +0.003780 | +0.013078 | +0.010422 | 0 |
| Reacher / 1972210000 | C | +0.010713 | +0.028784 | +0.007770 | 0 |
| Reacher / 1972220000 | R | +0.004511 | +0.000170 | -0.000869 | 44 |
| Reacher / 1972220000 | C | +0.003596 | +0.001051 | +0.002358 | 44 |

W point gains are positive for all heads in BOTH B2 halves, against quantile and
old calibration. Six pooled G gains are positive, but every G-versus-quantile
prefix interval crosses zero. Only Reacher root1 has positive G in both halves;
both heads defer at every sampled state, not evidence of conditional timing skill.

Retain the real partial positives. PointMaze root1 C versus actual C0 has W
+0.002760, interval [0.000200, 0.005321], and G +0.005196, interval
[0.000477, 0.009915]. PointMaze root2 improves old calibration G by about 0.0213,
with both heads' prefix intervals above zero, yet its G versus C0 is -0.002842
with an interval crossing zero. Beating the old cut does not establish beating C0.
Reacher root2 R goes from A G +0.018267 to B +0.000170; its B halves are
-0.020219/+0.020560. Conditional MC SE is 0.015296 for G versus 0.000304 for W:
long-horizon future uncertainty is substantial, not a demonstrated systematic loss.

## Frozen Readout Plan

Run the already declared whole-cohort E8 readout once: all four roots, both heads,
eight disjoint prefixes/root, three checks/prefix, eight futures/state, 2,112,000
steps. Keep cuts, models, comparators and half splits frozen; no B-driven head or
root selection. This is the missing prefix-transfer test, not another calibration
search. If it still fails joint W/G stability, stop this scalar-cut repair branch
before revisiting plan-value representation/long-horizon credit. E8 was not yet
submitted at calibration close; original confirmation HOLD remained.

## Frozen E8 Result

All four readouts t214504-t214507 exited 0: 32 disjoint prefixes, 96 states, 768
future pairs and exactly 2,112,000 steps. Measured CPU: 1,847.610 seconds; no fits
or controller updates. Only 76,587 public-result bytes were retrieved.

| Task / root | Head | E8 W/G vs. quantile | E8 W/G vs. old cut | E8 W/G vs. C0 | Now /24 |
| --- | --- | ---: | ---: | ---: | ---: |
| PointMaze / 1972110000 | R | -0.001469/-0.008262 | -0.001684/-0.004884 | +0.000813/-0.004371 | 17 |
| PointMaze / 1972110000 | C | +0.000835/-0.003169 | 0/0 | +0.002497/+0.000513 | 14 |
| PointMaze / 1972120000 | R | +0.002123/+0.002382 | +0.003368/+0.006357 | +0.001976/+0.001498 | 10 |
| PointMaze / 1972120000 | C | +0.001646/+0.001969 | +0.002891/+0.005944 | +0.001499/+0.001085 | 9 |
| Reacher / 1972210000 | R | +0.000214/-0.019771 | +0.000593/-0.018771 | -0.004901/+0.007318 | 0 |
| Reacher / 1972210000 | C | -0.001933/+0.006662 | 0/0 | -0.001597/+0.012954 | 2 |
| Reacher / 1972220000 | R | -0.003649/+0.005288 | -0.003649/+0.005288 | -0.003820/+0.015580 | 21 |
| Reacher / 1972220000 | C | -0.003669/+0.006501 | -0.003850/+0.001447 | -0.003820/+0.015580 | 21 |

Only PointMaze root2 has joint positive W/G points in both E4 halves against all
three controls. Neither head clears the both-root criterion on either task; no
E8 prefix interval is strictly positive. Preserve root2's directional evidence.
Root1 C's positive calibration-B G versus C0 shrinks from +0.005196 to +0.000513,
with opposing E4 signs and interval [-0.015079, +0.016104]. Its new cut changes
zero sampled decisions versus the old cut. Reacher root1 R still chooses now at
zero states; C chooses now at two and matches the old cut on all 24 states.
Reacher root2 loses W in both halves; R versus quantile has nominal interval
[-0.007149, -0.000149]. This is not just an uncertain G gain over a preserved W floor.

Stop the scalar-cut repair branch as planned. Keep E8 as exposed evidence, not
tuning data; no new jobs, fits or confirmation are released. Redesign plan-value
representation/long-horizon credit before another fresh evaluation; simply
regressing raw G already [failed to improve reliably](freq_hrl_target_credit_result_2026-11-10.md).
The [E8 summary](../results/plan_validity_independent_calibration_20261111/e8_summary.json)
retains all E4 contrasts, prefix intervals, conditional MC errors and action counts.

## Limitations

B4 uses independent future noise on prefixes seen by A4 cut selection; prefix
intervals are descriptive, unadjusted and not nested policy validation. The new
cohort and per-state A information both differ from the old fit-A calibration,
so a benefit does not isolate independence as its sole cause. Zero-now behavior
refers to these sampled interventions, not never replanning in complete episodes.
E8 intervals use eight prefixes/root and are descriptive, not multiplicity-adjusted.
Two trained roots/task and single interventions do not establish closed-loop or
cross-root performance; lifecycle costs remain incomplete.

Sources: [summary](../results/plan_validity_independent_calibration_20261111/calibration_summary.json),
[frozen protocol](freq_hrl_independent_calibration_2026-11-11.md).
