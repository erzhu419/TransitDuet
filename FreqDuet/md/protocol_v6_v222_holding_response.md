# V222: read-only V221 own-trip response decomposition

Use only V221's16 completed paths and fixed task IDs in all four contexts.
No replay,training,fitting,new future or change to controller/gates/parameters.
Freeze comparisons: lower-V107,joint-V107,joint-upper,joint-lower.

Align logical decisions by trip/stop/direction/occurrence,checked against the
stored event-uniform identities. Never overwrite repeated calls. Join native
base bins from responses and actual selected bins from recorded actions.
Reconcile issued actions with completed-trip holding. Unmatched decisions and
untraced holding remain separate,not zero-filled state-response evidence.

For matched decisions use the exact identity:
actual action difference = native bilateral request difference + override
difference. Partition each task trip before/at its first override and into
later overridden/native decisions;retain trips without overrides. Reconstruct
the frozen .25 rule and its fixed seven-bin quantization,without refitting.
Split continuous forward/follower terms,clipping and discretization;keep branch
changes separate. These are accounting terms,not independently identified causes.

For changed actions,reconstruct prepare/query clocks and the arrival-matched
predecessor. Check stored release+executed-holding proxy against completed
release/holding labels and check the forward context against the prepare clock.
Compare proxy,observed-forward and estimated bilateral squared losses at the
two recorded actions. Separately label actual chronological departure neighbours
with fixed-neighbour loss;abstain where an alternative crosses a neighbour.
This is not an unrun counterfactual rollout or same-predicted-follower ETA error.

Report every context and support count. Preserve V221 endpoints and prior
holding totals. Only a source-grounded mechanism may inform a later repair;
do not pick a new threshold,guard or gain using these consumed days. Scheduler
high/manual,node001--006;raw details stay remote,small summary only returns.

## Result and repair direction

High/manual t211232 on node006 exited0;source26cf09b4d0 passed8 tests. All
task decisions match,with0 repeated/unmatched calls or untraced holding delta.
Native requests and prior V221 holding totals reconstruct exactly. Only51258
bytes returned;no replay,training,fit,parameter change or main switch occurred.

Both actors reproduce the following task-trip accounting (seconds):

| Comparison | Day | Direct override | Native request change | Net holding |
| --- | --- | ---: | ---: | ---: |
| joint-V107 | 2151001 | -295 | +200 | -95 |
| joint-V107 | 2151002 | -395 | +305 | -90 |
| joint-upper | 2151001 | -295 | +190 | -105 |
| joint-upper | 2151002 | -395 | +260 | -135 |

In joint-upper,180/250s of the190/260s request increase occurs at LATER
overridden decisions;only10s occurs at later non-overridden decisions. Aggregate
quantization terms are-12.1/-27.7s,not the main positive response. There is no
branch-switch explanation. This is state-response accounting,not an independent
causal isolation of forward versus follower dynamics.

All214 joint reductions pass the proxy,154 harm observed-forward loss,and174
harm fixed actual-neighbour loss (214 supported,no predecessor switches).
Prepare/query clocks and proxy signs reconstruct exactly;query lag is1s.
Observed forward gap = proxy + already-realized nonholding elapsed difference;
the omitted term averages about-100s. All214 also harm the estimated bilateral
quadratic loss,so simply adding that hard nonharm guard would block this entire
candidate,not reconcile the objectives. Actual-neighbour labels are not rollouts
of the unexecuted action or same-predicted-follower ETA-error measurements.

Next design ONE lower command around an upper/lower common observable
gap/service-phase reference,including actual nonholding timing. Stop treating
release-only loss as a regularity certificate or stacking independent one-bin
cuts on bilateral025. Keep original fleet/bins and strong controls;preregister
any future independent-day validation before execution. Do not tune on these
two consumed days or start new collection/replay/training under this audit.
