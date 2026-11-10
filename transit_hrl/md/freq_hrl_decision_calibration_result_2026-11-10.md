# Frozen-Score Decision Calibration Result

All four cache jobs t214445-t214448 exited 0. Reconstructed 256 leave-prefix-out
ridge folds; all saved quantiles and OOF-B errors reproduced. The eight full heads
are unchanged. New work used 2.417 CPU seconds, zero full-model fits and zero native
steps. Retrieved 222,668 public-result bytes; recovered OOF scores remain server-side.

## Full-Score Comparison

Utility cut minus original quantile cut on independent-future B, with positive
W/G meaning improvement. Each root has 96 states from 32 known training prefixes.
R/C denote reference-continuation and C0-continuation label heads.

| Task / root | Head | B W gain | B G gain | Choose-now: utility / quantile |
| --- | --- | ---: | ---: | ---: |
| PointMaze / 1972110000 | R | +0.001795 | -0.000443 | 40 / 24 |
| PointMaze / 1972110000 | C | +0.001693 | -0.001506 | 42 / 26 |
| PointMaze / 1972120000 | R/C | +0.000845 | +0.004610 | 84 / 27 |
| Reacher / 1972210000 | R | -0.000269 | -0.003518 | 14 / 22 |
| Reacher / 1972210000 | C | +0.000121 | -0.001916 | 0 / 24 |
| Reacher / 1972220000 | R | 0 | 0 | 24 / 24 |
| Reacher / 1972220000 | C | -0.000341 | -0.002408 | 3 / 26 |

PointMaze root2's two heads choose identical actions, not two independent wins.
Its G gains persist across B halves, but second-half W is -0.000155. No changed
full-score policy improves both W and G in both halves. Every nonzero G contrast's
descriptive prefix interval crosses zero. Reacher C heads are all/near-all wait.

Two distinct failures remain. PointMaze root1 C has OOF A G gain +0.005168 but
OOF B G gain -0.001239: the selected gain fails independent futures even before
changing score sources. Reacher root2 C has OOF A W gain +0.000484 but full-score
A W gain -0.001221; B W/G go from +0.000771/+0.007894 to -0.000341/-0.002408.
The relative constraint/gain does not transfer between scoring views, which also
change the quantile baseline's actions; this is not evidence of an implementation bug.

## Decision And Next

PILOT_RECIPE_HOLD remains. Next isolate score-source mismatch using saved OOF
and frozen full scores, the same A-only utility objective, and B readout only.
Full-score A calibration would be in-sample, not confirmation. No further jobs or
native/closed-loop evaluation are submitted by this readout.

## Limitations

B has independent future noise but was previously revealed in development, and
contains known training prefixes. The pooled cut sees all A prefixes: OOF head
scores are not nested policy validation. Prefix intervals are descriptive; two
roots/task do not establish cross-root improvement. Lifecycle costs remain incomplete.

Sources: [summary](../results/plan_validity_decision_calibration_20261110/summary.json),
[frozen protocol](freq_hrl_decision_calibration_2026-11-10.md).
