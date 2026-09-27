# Stage-17 Paired-Value Qualification Protocol

Implementation revision: `c23b05640f`; 26 focused tests passed.

Frozen roots: 209011/209061; operational preflight: 208001. Use only the
cached Stage-16 branch-fit endpoint pairs. No environment rollout, controller
retraining, evaluation-path labels, new roots, trigger fit or policy deployment.

Train two shared scalar value networks on identical endpoint states:
absolute cost-to-go regression and paired wait-minus-now cost regression.
Both use Stage-16's 40 causal state features, two hidden layers of width 64,
64 full-batch Adam updates at 0.001, and the same initial weights. RNG key:
(root, held-out path, 16016). Identical training-only normalization, remaining-
time baseline and output scale are used in both. The sole change is the loss.
Pair weights equal the sum of their endpoint weights, giving each path equal
mass. A shared scalar potential makes reversed contrasts antisymmetric and
identical-state contrasts zero; absolute value accuracy is not claimed for
the paired objective.

Hold out one entire branch-fit path, including all its branches, per fold.
Compare predicted endpoint differences with actual suffix-cost differences.
Record pooled and per-path MSE against zero, the matched absolute-value
control, and the frozen Stage-16 predictions. The matched control uses the
same endpoints as the paired model, not Stage-16's denser suffix-state set.

Advance only if paired MSE is strictly below ALL three comparators on BOTH
roots. Otherwise stop this critic variant without epoch, width, threshold,
seed or normalization tuning. A pass admits a later frozen deployment test;
it does not establish control improvement or independent confirmation.

Compute/root: 16 critic fits, 1,024 optimizer updates, zero new environment
steps. Preflight has four fits/256 updates. Use scheduleurm, dynamic
node001-node006, one CPU/1536 MB per cell. Transfer result JSON only.

Preflight `t101249` completed on node004: four fits/256 updates, zero new
environment steps. Paired/absolute/zero MSE: 0.000002292/0.000004148/0.000000705;
operational checks passed, scientific qualification did not. Development
tasks `t101251/101252` completed on node004/node006 under the unchanged
protocol. The [result](freq_hrl_stage17_paired_value_result_2026-09-27.md)
failed qualification on both roots; no deployment test is admitted.
