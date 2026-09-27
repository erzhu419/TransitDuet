# Stage-21 Independent-Future Precision Test

37 focused tests passed, including new-draw disjointness, frozen predictions,
paired MC algebra, interval adjustment, replay boundaries and resource budgets.

Freeze Stage-20 coefficients and its out-of-path predictions, all 16 selected
states/root, roots 209011/209061 and the Stage-12 continuation controller.
The sole candidate remains mean-label ridge; controls are zero and the frozen
Stage-19 mean-label neural prediction. No continuation-critic fits, alpha changes, state
selection, new roots, deployment or optional stopping.

At each state draw exactly 64 NEW paired futures, using SeedSequence
(root, path, check, replica, 21021). Old Stage-18 namespace was 18019;
explicitly reject duplicate/overlapping draw seeds. Share each future between
now/wait arms, preserving the prefix and fork boundary. Never combine old
future labels with new labels for scoring. Preflight 208001 uses its two
existing states and four new futures/state.

For frozen candidate p, control b and fresh contrast A, score
D = (b-A)^2 - (p-A)^2 = b^2-p^2-2(b-p)A. Average equally over fixed states
and futures. Estimate Monte Carlo variance as sum_i var(D_i)/(K*N^2), with
Welch-Satterthwaite degrees of freedom. Report approximate Welch-t intervals,
Bonferroni adjusted over TWO controls times TWO roots (nominal family alpha
0.05; individual two-sided intervals 98.75%). A precision pass requires every
lower endpoint above zero. Also report the frozen point gate, raw MSE against
new means and variance/K-corrected MSE, retaining negative estimates.

These intervals concern future randomness conditional on frozen states,
predictions and present simulator latent state. They are not confidence
intervals over tasks, training seeds or unseen states. The original Stage-20
failure remains recorded regardless of this independent-label result.

Old runs saved no controller checkpoint. Reconstruct once/root using the exact
384-iteration recipe, then verify original endpoints/factual returns before
forking. This costs 3,686,400 training + 470,400 selection + 86,400 evaluation
steps/root. Each state replays one reference, two original arms and 128 fresh
arms: 2,515,200 further steps/root, total 6,758,400. Preflight costs 5,100
reconstruction + 6,600 replay = 11,700 steps. Critic fitting cost remains zero.

Use spawn workers (16/root, two in preflight), one CPU thread/worker, with one
controller reconstruction in the parent. Request 16 cores/12 GB per full task,
two cores/3 GB preflight, dynamically on node001-node006 via scheduleurm.
Return compact contrasts, seeds, predictions and metrics only; no histories or
checkpoints are downloaded. No extension beyond this fixed budget follows.

## Execution

Implementation `95e310a2b9`; preflight `t101396` completed on node006.
Its 7,489-byte result contains two states with four fresh paired futures each.
Frozen predictions, reconstructed endpoints and the 11,700-step accounting
passed; independent recomputation matched the paired intervals and corrected
MSE. The tiny preflight failed both scientific gates: control-minus-candidate
MSE is -3.29567e-7 for both controls, with interval
[-3.37942e-7, -3.21192e-7]. Formal settings remain unchanged.
