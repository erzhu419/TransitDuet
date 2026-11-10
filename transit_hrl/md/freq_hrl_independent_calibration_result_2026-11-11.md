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

## Next

Run the already declared whole-cohort E8 readout once: all four roots, both heads,
eight disjoint prefixes/root, three checks/prefix, eight futures/state, 2,112,000
steps. Keep cuts, models, comparators and half splits frozen; no B-driven head or
root selection. This is the missing prefix-transfer test, not another calibration
search. If it still fails joint W/G stability, stop this scalar-cut repair branch
before revisiting plan-value representation/long-horizon credit. No E8 jobs are
submitted in this analysis; original confirmation HOLD remains.

## Limitations

B4 uses independent future noise on prefixes seen by A4 cut selection; prefix
intervals are descriptive, unadjusted and not nested policy validation. The new
cohort and per-state A information both differ from the old fit-A calibration,
so a benefit does not isolate independence as its sole cause. Zero-now behavior
refers to these sampled interventions, not never replanning in complete episodes.
Two trained roots/task and single interventions do not establish closed-loop or
cross-root performance; lifecycle costs remain incomplete.

Sources: [summary](../results/plan_validity_independent_calibration_20261111/calibration_summary.json),
[frozen protocol](freq_hrl_independent_calibration_2026-11-11.md).
