# Stage-18 State and Future-Noise Diagnostic

Implementation: `8a0ea255d6`; 29 focused tests passed. Unforked generation
matches the previous committed driver exactly on three 1,200-step paths.

Frozen roots: 209011/209061; preflight: 208001. Preserve Stage-16/17 failures.
Reconstruct the same frozen controller and Stage-12 reference trigger, then
replay all 96 Stage-16 fit pairs/root. Require the recorded boundary features,
suffix costs, factual trajectory and one-call-per-bin budgets to match.
Record the 384-element causal history actually available to both controllers.

Compare paired-value critics on compact summaries versus summaries plus raw
history. Both have 424 inputs, two width-64 hidden layers and 31,425 parameters;
compact has zero-filled history slots. Use identical initialization, 64 Adam
updates at 0.001, path weighting and training-only normalization. Hold out all
branches of one path per fold. RNG key remains (root, held-out path, 16016).
History qualifies only if its original-label MSE beats compact and zero on
BOTH roots. This is a representation diagnostic, not a new deployed policy.

Select two of the 12 opportunities/path without labels using RNG key
(root, path, 18018). At the 50-step window boundary, replay eight new paired
futures with keys (root, path, check, replicate, 18019). Now/wait arms share
each future realization. Preserve all observations through the boundary and
redraw unrevealed event times conditional on elapsed dwell/gap/pulse duration:
discrete duration D is uniform on [max(original minimum, age+1), maximum].
Keep the current hidden regime, route progress, active force vector and
distractor; subsequently use the original transition laws. Hidden state is
used only by the simulator's conditional sampler, never by either critic.

Report within-state continuation variance, between-state mean variance minus
the finite-replicate contribution, and prediction MSE against replicate means
minus that same contribution. Retain negative finite-sample estimates. These
describe future randomness conditional on present latent driver state, not
all uncertainty given observable history. A failed history comparison cannot
prove state sufficiency. No root expansion, tuning or deployment follows here.

Additional replay/root: 9,600 reference + 230,400 original pairs + 307,200
resampled pairs = 547,200 primitive steps; 16 critic fits/1,024 updates.
Controller reconstruction (including its existing evaluation workload) and
inherited supervision are additional. Preflight: two paths, two pairs/path,
one noise opportunity/path, four futures, 300-step horizon, 7,800 replay steps.
Run via scheduleurm on dynamic node001-node006, one CPU/1536 MB per task.
Sync compact metrics/predictions/future contrasts only, not histories or ckpts.

Preflight `t101286` completed on node004: exact cached endpoints and preserved
fork boundaries, four pairs, two noise states/four futures each, 7,800 replay
steps and 256 critic updates. Retrieved JSON: 6,970 bytes. Compact/full-history/
zero MSE: 5.473e-7/6.781e-7/7.054e-7. Operational checks passed, but history
qualification did not. Negative finite-replicate variance/MSE estimates were
retained; no settings were changed before the full development diagnostic.
