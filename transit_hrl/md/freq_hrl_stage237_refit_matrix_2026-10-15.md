# Stage237: matched reference and residual refits

Run: `native_transit_refit_matrix_stage237_development_20261015_r1`.

## Protocol

Freeze six Stage234 candidates; cross 54/60 states and 2/4 draws under fixed reference, changing reference/fixed residual coordinates, and native coordinates. Preserve 32 cells, 33 designs/cell and 20 matched blocks/cell; reuse full60/4. Measure original-anchored physical MSE, own-gate regret, and cost/wait/risk against common/own raw in five scopes. Cache new models remotely.

## Results

- t153212-t153244 completed; six tests passed. All 53,760 outer choices/107,520 scores, raw OOF and prefix contributions reproduced with zero differences. Only the 4.56 MB summary was pulled.
- With fixed reference/coordinates, the rank3 regression case's full-training MSE fell **5.61%**, but gate cost regret rose **3.74%**; cost/wait worsened **+0.000167246 / +0.0004375 min**. Two-draw fits also reduced MSE while worsening decisions. Proxy fit improvement is not policy improvement.
- Changing54/2 raw disagreed with common raw on **34.5%** of mixed-source held decisions. Its cost/wait deltas were **+0.000766703 / +0.001606367 min**. Coordinate effects were smaller in the regression case; reference drift is material, but freezing it does not fix the residual.
- Four draws helped root401's wait outcomes; mixed raw-safe held cost remained worse in all four fixed-reference configurations. Defaults unchanged.

## Next

Reuse cached references/bases. Compare all-action MSE, feasible-action weighted MSE and ranking/regret objectives at identical capacity, retaining zero residual as a control. Validate training decision gains before enlarging encoders or running more seeds.

## Limitations

The fixed outer reference has seen all outer-training labels; this is a diagnostic, not a new clean inner selector. Reused matched blocks are correlated and give no fresh CI. Own-gate regret changes its feasible set when the raw reference changes.
