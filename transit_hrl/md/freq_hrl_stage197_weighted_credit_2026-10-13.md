# Stage197: weighting versus action credit

Stage196 found weak fleet action credit even on training data, and held-out
cost pair errors worse than zero differences. Stage197 separates two changes:

- Replicate-count weighting: mean8 states receive weight8, singleton states
  weight1; normalized weighted state losses, unchanged training-only scales.
- Direct action credit: remove the absolute fleet anchor, fit fleet/nonfleet/wait
  pair gaps, normalize fleet by training pair RMS instead of absolute risk RMS.

The 2x2 comparison reuses Stage195 anchored/uniform models and fits the other
three variants. Same600 inputs,240 mean8 targets,40 evaluation prefix groups,
47 grouped folds,1635 parameters,100 updates, seed178 and AdamW settings.
All three heads train jointly. Replicate counts are a reliability proxy, not
heteroskedastic inverse-variance estimates. Direct credit does not forecast
absolute fleet risk; the network initialization is shared, not its decoded
fleet output across objectives.

Primary: advantage/count-weighted, wait-feasible. Gate unchanged: negative mean
physical cost and nonpositive wait on both lower roots across all three splits.
Native episodes and actor updates:0. Reserved scenes remain closed. This is an
action-credit development screen under the original greedy continuation.

Run: `native_transit_weighted_credit_stage197_development_20261013_r1`.
Scheduler:47 fit tasks plus one report,1CPU/3072MB, unbound node001-node006.
Models, target maps, draws and raw predictions stay server-only.
Tests:37 passed, including the existing floor/sequence/mean-target paths.

Next: compare weighting effects within each objective and objective effects
within each weighting scheme, using physical outcomes, false-feasibility and
noise-corrected pair errors against the zero-difference reference. A failed
gate will not trigger more native collection or open the reserved screen.
