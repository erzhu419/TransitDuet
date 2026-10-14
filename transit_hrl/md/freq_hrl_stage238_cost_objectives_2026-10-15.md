# Stage238: same-capacity cost objectives

Effective run: `native_transit_cost_objectives_stage238_development_20261015_r3`.
Tasks `t153402`-`t153433` and aggregate `t153434`: all done/0; 8 unit tests passed.

## Design

All 32 outer cells, 33 designs/cell, and matched 54/2, 54/4, 60/2, 60/4 blocks retained. Fixed Stage226 reference and Stage237 fixed-arm paths reused. Same 198 causal features, TSVD subspace, frozen six Stage234 rank/alpha keys, raw gate, and all-MSE residual RMS budget; only the objective changes. No raw-reference fits, native episodes, or actor updates.

All-action MSE uses cached weights. Feasible MSE equally weights states and averages their nonanchor gate-eligible actions. Rank/regret uses gate-eligible pairwise logistic loss weighted by training cost gaps; it is a cost-ranking surrogate. Feasible MSE is solved analytically in the RMS ball; ranking uses a Lagrangian dual and SciPy stationarity/root solvers. Actual projected KKT <=1e-8 and training-loss increase <=1e-12 are required.

Effective run: 6,402 objective solves. All 2,177,280 cached-control choice comparisons match; scope metric difference = 0; reconstructed subspace prediction difference <=4.05e-16. Only the 4,291,179-byte aggregate JSON was pulled. Models/panels remain remote. Earlier r1/r2 numerical solver failures are archived separately and excluded.

## Results

Mixed pooled 60/4, frozen **raw-safe capacity**; deltas against cached raw, lower is better:

| Objective | Training cost delta | Training regret vs raw | Held-other cost delta | Held-other wait delta (min) |
|---|---:|---:|---:|---:|
| All-action MSE | +0.000087206 | +1.36% | +0.000291461 | -0.000406250 |
| Feasible MSE | -0.000017935 | -0.28% | +0.000204496 | -0.000204948 |
| Rank/regret | -0.000071787 | -1.12% | +0.000141397 | -0.000346354 |

In the preidentified `mixed_b/fold1/even_odd/direction1` rank3@1 case, feasible MSE leaves training actions unchanged. Ranking changes training cost delta from +0.000167246 to -0.001490467 and regret from +3.74% to -33.29% versus raw, in the same subspace. Its training wait delta increases from +0.000437500 to +0.000645833 min. This demonstrates a decision-objective mismatch, not merely insufficient representation capacity.

Ranking reduces mixed held-other cost harm by 51.49% at 60/4 but still loses to raw. At 54/2 and 60/2 its held-other wait deltas are +0.000238763 and +0.000254557 min. Mixed source397 still has positive cost/wait deltas; standalone source401 raw-safe ranking changes held wait from -0.001235417 to +0.001106250 min. Cost-only ranking does not close transfer or wait performance.

## Boundary And Next Step

These are reused development states/futures, not fresh CI. Fixed outer reference saw all outer training labels; this is an objective diagnostic, not clean inner admission. Frozen old safe keys do not certify safety of the new objective. Defaults unchanged; complete learned HRL remains unproved.

Next: reuse Stage237 cached changing-native references for clean prefix/future OOF objective selection, then add wait-aware ranking and two-future-half cost/wait/risk admission. Freeze the resulting choice before held-state/other-future evaluation; retain abstentions and source-specific regressions.
