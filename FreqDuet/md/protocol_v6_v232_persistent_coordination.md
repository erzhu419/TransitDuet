# V232: persistent physical-cycle coordination preregistration

Canonical contract:`preregister_freqduet_v232_persistent_coordination.py:registration`.
Current work is protocol sealing,synthetic reference-state tests and AST interface
checks only. All archive/collection/replay/shadow/fit/training/future budgets are0.

Keep V107,V220,12 vehicles,seven bins and all original action permissions,
cadence,caches,projections,predictors and calibration versions. Main is unchanged.
Coordination inputs are current pending obligations/versions,ready stock and
vehicle positions with clocks,and the executed holding prefix of the actual
bus/trip/observed-launch cycle. Native gap control remains unchanged.

Spend each eligible ready bus once:due plans first in native order,then future
plans with remaining stock. A current-tick return is usable next dispatch tick.
Separate currently uncovered due plans from future plans without current ready
backing. Order compatible incoming buses by remaining distance,original inventory
ties;associate ordinally with unbacked due then future plans. Recompute at every
query. This is a position-only hypothesis,not actual assignment,ETA or slack;
incoming buses never authorize an upper advance. Preserve unmatched entries.

Launch opens H=0. Query records actual H,delta since last sample,current versions
and the refreshed association. Replans,pressure relief,association changes and
new stations do not reset H or create spendable credit. Do not attribute the whole
between-query delta to the current obligation. Actual return closes the account;
only a later observed same-bus launch opens a new cycle and separate parent link.
No future supply identity is backfilled or used to choose an intervention point.

A future action rule may retain/increase/decrease native feasible bins;it is not
a uniform cut or strict local-R-nonharm guard. No selector or shadow action is
computed now. Later tests require native continuation,independent selection/eval
futures,V107/upper/lower/joint controls and a memoryless-state ablation. Joint must
improve passenger time without harming actual CV2,readiness,public-total delay or
service completion,against V107 and each single-layer control. Permissions stay
identical;strong fixed+bilateral remains the external baseline.

Next:obtain separate authorization for a read-only audit of this frozen state
and position association on the8 saved V230 paths,before any passive integration
or candidate action rule. No trajectory read or intervention is admitted here.

Limitations:V231 has2 consumed actors/2 days. Distance rank is not return order;
future plans lacking current stock do not alone prove shortage. Persistent state
correctness does not establish a useful action value or a deployable controller.
