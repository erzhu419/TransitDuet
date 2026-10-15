# Stage239: clean OOF objective selection

Run: `native_transit_objective_oof_stage239_development_20261015_r1`.
Fits `t153437`-`t153468`, aggregate `t153477`: all done/0. Six unit tests passed.

## Design

All 32 outer cells retained. Each has 10 complete-prefix folds x2 future directions: fit54 states/2 draws, validate6 disjoint-prefix states/opposite2 draws. Stage237 changing-native references and TSVD paths reused; **zero raw-reference fits**. Four objectives: cached all-MSE, feasible MSE, cost ranking, and wait-aware ranking of cost/cost-scale + wait/wait-scale (fixed weight1). Same causal features, per-rank subspace, RMS ball, raw gate, and 17 rank/strength candidates. Independent objective selection and joint selection compare pooled per-source cost/wait/risk floors with additional per-source, per-future-half floors. Selection precedes full60/4 fitting and outer evaluation.

All 3,840 cached OOF raw decisions, old all-MSE curves, and 192 old selected keys reproduced exactly. Subspace prediction error <=3.14e-16. 16,128 low-dimensional objective fits; no new episodes or actor updates. Only the 6,689,036-byte aggregate was pulled; model/panel caches remain remote.

## Results

Mixed pooled held-state/other-future deltas versus raw; negative is better:

| Raw-anchor selection | Cost delta | Wait delta (min) | Unsafe delta | Switched decision evaluations /960 |
|---|---:|---:|---:|---:|
| All-MSE pooled-safe | +0.000291461 | -0.000406250 | -0.006250000 | 30 |
| Cost ranking half-safe | -0.000006165 | -0.000066927 | -0.001041667 | 2 |
| Wait-aware pooled-safe | -0.000136782 | -0.000305208 | 0 | 8 |
| Wait-aware half-safe | 0 | 0 | 0 | 0 |
| Joint pooled-safe | +0.000263827 | -0.000658594 | -0.008333333 | 45 |
| Joint half-safe | +0.000077710 | -0.000333333 | -0.003125000 | 9 |

Wait-aware pooled gains come from two outer cells; source397 unsafe delta is +0.002083333 despite pooled zero. Cost-ranking half-safe gains come from one cell/two actions. Joint half-safe reduces cost harm by70.55% versus joint pooled-safe but still loses to raw. Its mixed admission falls12/16 to7/16 cells; four of the seven admitted candidates make no training action changes after full fitting. The prior `mixed_b/fold1/even_odd/direction1` all-MSE rank3@1 case remains the largest mixed joint half-safe cost regression. Standalone source401 joint half-safe lowers cost but increases wait by0.000570313 min. No universal objective/selection winner.

## Boundary And Next Step

Reused development data, not fresh CI. OOF safety floors do not guarantee held-state safety; utility-score MSE is not cost calibration. Tiny switched-action support and training/deployment model differences prevent a stable performance claim. Defaults unchanged.

Next: freeze these candidates and separate OOF selection instability from final-refit policy changes. Reuse cached54/2 and60/2 references alongside the current60/4 endpoint, comparing physical held outcomes and action consistency under matched objectives; inspect prefix concentration with existing OOF curves. Do not widen the encoder or rerun raw-reference training before this distinction is resolved.
