# Stage-26 Temporal Plan Supervision

Freeze controller roots 209011/209061 and Stage-12 reconstruction. New path
bases 3260000/3261000: offsets 1-16 train, 101-108 evaluate. No old branch or
evaluation paths enter these sets. Preflight 208001 uses base 3259000 and two
paths/role. Preserve one upper call/bin; lower feedback is unchanged.

Choose 20 eligible bins/path without replacement using the existing timing
sampler with root+26026. Cycle offsets 0/5/10/15/20, four each/path. Paired
arms differ only by renewing now versus five steps later; other calls follow
the same balanced-jitter schedule. Simulate identical prefixes, then 50 steps;
record wait-minus-now cumulative ISE at 10/25/50, all with equal call counts.
This is a one-check timing-response curve, not a remaining-lifetime label.

Inputs are 64 pre-intervention frames: physical/achieved state, target and
waypoint error, measured task channels, previous executed action, plan age,
bin/episode clocks and budget-spent indicator. Initial padding has a validity
indicator. No post-intervention state, future disturbance or regime label.

Compare history, current-frame repetition, and shuffled past (current frame
unchanged; permutation namespace 26028). All use identical GRU(32)+linear(3),
zero-initialized output heads, initialization namespace 26027, training-only
feature normalization and target RMS scaling. Predict ISE/horizon-seconds.
Adam 1e-3, clip-norm 1, 128 epochs, batch <=64, identical minibatch orders;
no hyperparameter/epoch selection. Preflight uses two pairs/path/four epochs.

Primary curve-rate MSE must beat zero and both matched controls on BOTH
roots. The sign of the 50-step prediction must also yield positive paired
ISE benefit versus both controls and always-now/always-wait. Ties fail.
Report all evaluation paths, not selected windows; no episode-gain claim.

Full tasks request 16 replay workers plus one coordinator and 24 GB RAM, dynamically
on node001-node006. Charge controller reconstruction, factual replay and every
paired prefix/window. Save controller, raw NPZ and curve models in a sibling
`replicate_ROOT_raw` directory on the server; sync only result JSON.
Sixteen focused tests cover causal capture, action-dependent replay agreement,
window signs/call budgets, path isolation, matched models and scheduler scope.

Execution at `b7bc0c587c`: preflight `t101584` completed on node004 with
4 training/4 evaluation pairs, 64x23 inputs and 8,860 charged steps. Only
10,928 bytes of JSON were retrieved; all three raw artifacts stayed remote.
Both preflight scientific gates failed; full settings remain frozen.
Tasks `t101585/101586` are running on node004/node006 for 209011/209061.
Expected total steps are 4,860,700/4,851,700, each including 4,243,200 for
controller reconstruction and 1,200 for factual replay. Full results pending.

## Limitations

New paths do not make these reused controller roots independent confirmation.
Randomized timing-state coverage differs from deployed-trigger visitation.
Windowed response excludes later credit and does not establish policy gains.
