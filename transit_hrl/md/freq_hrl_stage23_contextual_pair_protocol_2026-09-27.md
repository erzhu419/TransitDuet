# Stage-23 Contextual Paired-Value Screen

Freeze roots 209011/209061, 16 states/root, Stage-20 whole-path folds and
training-endpoint normalization. Fit only Stage-18 eight-draw means; use
Stage-21 64-draw means only for scoring. Preflight 208001 retains its existing
two states and four-draw caches. No new replay, root, alpha or label-budget sweep.

Let z be the saved training-normalized endpoint, r its remaining fraction,
s its 13 state coordinates and c its 25 causal exogenous coordinates.
Context comprises target/current force/distractor, target velocities and their
change norm, force RMS, distractor deltas and change norm. State is every other
coordinate except within-bin and remaining fractions. These schema groups are
fixed by feature names, not selected from labels or observed performance.

Use the shared value basis phi(x) = r [z, s tensor c_b], where
c_b = tanh(c)/sqrt(25). Predict [phi(wait)-phi(now)]' beta, preserving
antisymmetry. The added block's pairwise norm cannot exceed the state-difference
norm times r. Fit mean MSE plus unit L2, as in Stage-20. Use 14x14 dual systems,
with 365 coefficients and no intercept, versus the frozen 40-column linear fit.

Controls: zero, frozen linear, and a matched random-context model. The latter
uses the same basis size, penalty, labels and normalization, replacing c_b by
its norm times a random Rademacher unit direction. Direction is fixed by
SeedSequence(root, path, check, 23023), shared between arms. This preserves
every train/query design row norm; report effective degrees of freedom too.
It retains context magnitude and controls direction, not all contextual input.

Sole candidate: contextual model. Development gate: corrected held-out MSE
strictly below all three controls on BOTH roots. Retain negative corrected
MSE, all path results and failed gates. No retuning or automatic deployment.
Sixteen new closed-form fits/root, zero gradient updates or environment steps.
Use scheduleurm, one CPU/1.5 GB per task, dynamic node001-node006 placement;
sync only compact predictions/diagnostics JSON, no checkpoints.

Twenty-two focused tests passed: context activation, antisymmetry, bounded and
matched row norms, weighted primal/dual equivalence, training-only normalization,
scoring-label isolation, fold separation, gate semantics and scheduler resources.

## Limitations

These scoring labels informed prior diagnostics. This is a reused-development
screen, not independent confirmation; no new significance claim is made.
Matched parameter count and row norm do not ensure identical effective model
capacity. Even a pass would not establish policy gain or cross-task frequency HRL.
