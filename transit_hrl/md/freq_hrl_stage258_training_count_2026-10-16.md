# Stage258: Match Inner Training Futures

Only training futures change: cached inner8 halves versus their inner16 union.
Old keys, full actions, fit states, full basis/prior/scale/RMS cap and rules stay
fixed. Evaluate on common opposite16; no reselection or held inner16 policy test.

All 33 tasks completed exit 0, no retries, node004/005/006. Six tests passed;
640 inner16 fits / 5,120 solves, no inner8/raw/full refits or new episodes.
Controls reproduced within 3.47e-18. Pulled a 4,015,482-byte scalar report and
small preflight receipt; new models/actions remain remote.

Mixed new8 / pooled-safe removed group, 12 bank-cell exposures per family:

| Family | Action Disagreement With Full, 8 -> 16 | Absolute Wait Response Gap, 8 -> 16 (min) |
|---|---:|---:|
| free_old | 2.986% -> 2.778% | 0.000464670 -> 0.000478299 |
| shrunk | 2.847% -> 2.639% | 0.000278646 -> 0.000425174 |

Root397 action gaps improve more (4.167% -> 0.556%; 4.333% -> 1.667%).
Mixed cost/wait effects still trade off; opposite-future wait effects change sign
between halves. Matching future count partly helps, but does not close the gap.

Limitations: selection-conditioned development strata; views/banks are not seed
replicates. Full agreement is not truth accuracy or new performance proof.
Prefix exclusion still changes fit/calibration populations. Stage253/248/169 closed.

Next: reuse inner16 models to compare removal effects on excluded states,
retained states, other roots and held prefixes. Separate local extrapolation from
global/cross-root coupling before another loss or gate change; no refits needed.
