# Stage120 Bounded Plan Authority

Stage119 gain was about 0.0007 on a 1141-1152 return baseline after 65.1M steps.
Next: test plan amplitude and execution authority, with all Stage112 learned lowers,
392 feedback features, standard deviations, forecaster, native task and clocks frozen.
Intervene in one anchored Bernstein coordinate for one option by +/-0.05 or +/-1.0;
other options execute zero residual. Compare advice with bounded donor tracking:

`mean_new = mean_existing + 0.05*tanh((donor(feedback+plan_delta)-donor(feedback))/0.05)`.

Only the donor query changes error/velocity by curve-minus-forecast position/velocity.
The main feedback stays intact; zero residual reproduces forecast exactly. Account for
two extra donor forwards/reference step. This is not a free constant action bias.

Eight roots, periods 50/100, four fresh queries at 0/300/600/900. Pair scenario/prefix
and innovations within independent suffix-noise panels A/B. A selects zero or one of
16 directions, B scores full suffix; reverse/average. Ten Bonferroni-corrected endpoints,
equal-root bootstrap65,536. Prospective minimum conditional gain: **0.5 suffix return**.

Train the new channel only if both periods' large-reference gain CI is above 0.5 and
reference-minus-advice CI above zero. Large-advice alone above 0.5 supports amplitude
learning instead. All tested gain CI upper bounds below 0.5: stop this fixed-lower branch.
Otherwise: inconclusive, no automatic seed extension. Preflight authorizes no training.

Full: 8,576 episodes / 10,291,200 steps; four workers/root, five CPUs, 8 GiB RAM;
scheduleurm dynamic node001-node006. Preflight: 268 episodes / 80,400 steps. Only compact
JSON returns locally; no checkpoint/trace writes or performance-based protocol revisions.

## Execution

Code `1bb51f58b8`; six new tests plus Stage117/119 regressions passed. Native preflight
`t135538` and aggregation `t135539` passed mechanically. Frozen full run
`pointmaze_plan_authority_stage120_full_20261005_r1`: workers `t135541`-`t135548`,
aggregation `t135549`; formal results pending.

## Limitations

Future-suffix selection is not deployable. Finite candidate headroom is not a global
upper bound, HRL/promotion proof or equal-compute comparison. Corrections are mean-space
bounded; curves are world-clipped, not collision-certified. Stage67 HOLD remains.
