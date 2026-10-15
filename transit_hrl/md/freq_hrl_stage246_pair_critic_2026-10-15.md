# Stage246: variance-weighted paired-margin regression

Run `native_transit_pair_critic_stage246_development_20261015_r1`:32 qualified cells
and report,33 done/0;3 failed attempts and1 cancelled attempt retained (37 total,
4 retries).24 tests passed. All92,400 Stage245 control fields reproduce exactly.
Pulled2,033,503-byte aggregate plus small paired MC diagnostics; no models/panels.

## Design And Results

Freeze raw/all-MSE/feasible-MSE/rank-regret controls, B targets/shrinkage,198 features,
17 candidates, k-dimensional sum with fixed head contrast, A RMS cap and OOF gate.
Pair-MSE and pair-GLS share feasible-pair component targets. GLS uses same-future
action-difference variance, prefix/global half pooling and per-head normalization.
Primary: `pair_gls_pooled_safe`. Mixed Transit held-prefix/fresh8 versus raw:

| Pooled-Safe Objective | Cost Delta | Wait Delta (min) |
|---|---:|---:|
| Frozen All-MSE | -0.000129902 | -0.000213411 |
| Frozen Feasible MSE | -0.000095842 | -0.000419531 |
| Frozen Rank/Regret | +0.000039061 | -0.000127214 |
| Pair-MSE | +0.000092162 | -0.000142969 |
| Pair-GLS (primary) | -0.000097005 | -0.000427214 |

GLS improves pooled cost/wait in both fresh halves and both Transit mixtures.
Root401 still harms both (+0.000010550/+0.000095833); train-prefix fresh also harms.
Only12 states change; fleet benefit rests on1 sampled event. Cost/wait are1.48/2.72
conditional MC SE from zero. Paired GLS-minus-pair-MSE cost is-0.000189167 (1.88 SE);
GLS-minus-feasible-MSE cost/wait gains are only1.26/0.95 SE. No new strong claim.
Pair-MSE has fewer pair-sign errors (24.42% versus GLS24.77%) yet worse cost.
Half-safe GLS admits1/16 mixed cells and does not improve raw. Weight audit finds
no consistent root401 downweighting; effective state/head weights do vary.

## Repair And Next

Fixed small-loss dual-root scaling and the actual false KKT rejection: an interior
optimum incorrectly acquired a multiplier from roundoff. Inactive multiplier now0;
same1e-8 gate and optimization problem. Failed lineage remains recorded.
Next: state-balanced GLS to isolate within-state confidence from cross-state
reweighting, and separate fleet-tail sampling from service sign errors. No raw refits.
Limits: development futures/conditional MC, not training-seed CI or end-to-end HRL;
pair-GLS versus pair-MSE isolates weighting, comparisons with old losses do not isolate
geometry alone. Defaults and reserves unchanged; full models remain remote.
