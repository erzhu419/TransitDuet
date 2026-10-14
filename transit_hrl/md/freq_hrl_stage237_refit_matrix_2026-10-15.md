# Stage237: matched reference and residual refits

Run: `native_transit_refit_matrix_stage237_development_20261015_r1`.

## Protocol

Freeze the six Stage234 candidates. Cross 54/60 training states and 2/4 training draws with fixed reference, changing reference/fixed residual coordinates, and changing reference/native coordinates. Thirty-three unique designs map to 20 matched prefix/half blocks per cell. Preserve all 32 cells and both sources; reuse the full60/4 endpoint.

Compare physical original-anchored cost MSE, own-gate cost regret, and cost/wait/risk versus common and own raw actions. Evaluate own fit data plus all four outer scopes. Reproduce cached outer controls, clean54/2 raw OOF and selected-prefix contributions. Cache new references/paths remotely; pull summaries only. Scheduler budget: 1024 raw refits/614400 updates, 6144 analytic SVDs, no native episodes or actor updates.

## Results

Pending. Separate reference/coordinate changes from residual sample/draw refits before increasing capacity or tightening admission.

## Limitations

The fixed outer reference has seen all outer-training labels; this is a diagnostic, not a new clean inner selector. Reused matched blocks are correlated and give no fresh CI. Own-gate regret changes its feasible set when the raw reference changes.
