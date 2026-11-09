# V401: budget-conditioned upper control

The sole method change is explicit knowledge of the operator's fixed per-day
retention. Both learners have 181 input channels and identical initialization,
network, J0 reward, update count and canonical budget/joint actions. The first
180 channels retain V350 semantics. The last is rho for `explicit` and zero for
`implicit`; the latter still observes physical remaining allowances. Rho is not
an actor action and never changes within a path. Native seven-bin holding,
prediction, 12 vehicles, 262 trips and irreversible commitments remain unchanged.

Each learner starts fresh with seed 314011 and the original 32 training days,
eight four-day batches and 192 updates per batch. Both receive exactly the same
day/rho assignments: .25/.50/.50/.25 in every batch, sixteen days per retention.
Do not warm-start from V400, double the training budget or select a checkpoint.
The candidate remains the eighth batch. Only the documented target audit may
stop an update; no replacement batch or tuning follows a failure.

This phase authorizes only server-side synthetic checks: zero physical paths,
native Runner constructions, candidate training or optimizer updates. Checks
cover legacy state-prefix equality, matched initialization, per-instance quota,
irreversible accounting, terminal credit and real actor/critic input dimensions.
They do not establish traffic equivalence or control benefit.

Proposed next budget, not released: two old-policy engineering reproductions,
64 training paths and 24 development paths, at most 90 physical starts and eight
concurrent paths. Engineering reuses consumed reference days only to test old
behavior, not to select a new candidate: V400 day 2400001, K4 at .50 and .25.
The old frozen actor receives its original 180-channel prefix during these two
checks; the new channel is recorded, not used to warm-start a new learner.
Actions, clocks, endpoints and budget execution must retain the V400 contract.
The four proposed development days in
protocol.json must be checked and reserved before traffic authorization. V400
days and outcomes are excluded from training and development selection.

Development compares manual joint, implicit-rho and explicit-rho at each of the
two frozen retentions. Screen the explicit candidate separately at each rho:
mean journey must improve against both controls, with mean actual CV squared,
readiness, original-public total delay and service completion non-harmful. Report
every day and both retentions; do not pool away a failed setting. Four days are
a development screen, not statistical confirmation. A failed screen closes this
explicit-budget candidate; it does not authorize more rho values or ETA patches.
Any later multi-seed or harmonic confirmation requires a separate frozen budget.

All checks and experiments use scheduler high, manual dispatch, node001--006.
Failures stay recorded and are not automatically retried. Old source/results,
V400's failed harmonic gate and the current main policy remain untouched.
