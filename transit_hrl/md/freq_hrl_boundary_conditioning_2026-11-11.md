# A-Only Fixed-Capacity Observable Conditioning

The [frozen protocol](../protocols/plan_validity_boundary_conditioning_2026-11-11.json)
compares tracking-distance and state-RMS gates using training covariates only.
Both use 240 columns/DF3 and a norm-preserving partition of the shared potential.
The cached uniform model has an exact 240-column lift, requiring no refit.
Keep matched31-prefix controls, reciprocal A views, strata and window halves.

## Completed Result

t214743 completed on node004: exact 512 fits, zero baseline refits/native steps,
8.330 CPU seconds, 21 tests passed. Retrieved 116,147 public bytes; models stay
remote. [Result](../results/plan_validity_boundary_conditioning_20261111/summary.json),
[diagnosis](../results/plan_validity_boundary_conditioning_20261111/diagnosis.json)
and [receipt](../results/plan_validity_boundary_conditioning_20261111/release.json).

Prefix-held MSE change versus cached uniform, two-view descriptive means:

| Root | Tracking gate | RMS gate |
| --- | ---: | ---: |
| PointMaze 1972110000 | +1.30% | +0.50% |
| PointMaze 1972120000 | +1.33% | +0.65% |
| Reacher 1972210000 | -2.38% | -1.79% |
| Reacher 1972220000 | +0.27% | -0.29% |

Reacher1 passes the predeclared development rule; matched-future MSE improves
0.99% and the prefix gap shrinks 14.88%, but held MSE remains 2.82% above zero.
Prefix 1972277014 supplies 81.53% of the net gain; 21/32 prefixes improve on
the mean view. Omitting its readout without refitting retains -0.000162 MSE.

## Followup And Limits

This is A-only development, not fresh-prefix or policy confirmation. DF matching
is conditional on X, not identical nonlinear function classes. E remains untouched.
Freeze both recipes. Next prepare eight full-A-mean models, then separately
release fresh-prefix validation retaining uniform Wide and older eight-prefix
base controls. No further fit/native budget or root-specific selection is released.
