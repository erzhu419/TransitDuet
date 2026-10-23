# V223: one observable phase reference, algebraic preflight only

V222 identified conflicting commands,not a new successful controller. Replace
the independent one-bin subtraction with ONE phase-relative bilateral request.
No V221 outcome fitting,archive scan,replay,training or new future is admitted.

For prepared actual forward gap F and frozen native follower estimate B,use
delivered upper plan offsets u_front,u_self,u_follower. Each offset is the
coordinated executable launch tick minus the native tick at the SAME upper
admission,not public lateness,the raw +30 request or continuous plan advance.
Keep the original gain g=.25,clipping[0,45],seven feasible bins and shorter ties:

    request = g/2 * ((B-F) + (2*u_self-u_front-u_follower))

This is bilateral control around forward reference H+u_self-u_front and backward
reference H+u_follower-u_self. H cancels. Zero offsets reproduce native V107;
equal offsets cancel. A pure delivered phase change is not corrected again,
while real nonholding/holding changes remain in the measured gaps. No per-stop
credit,override,action-band floor or later suppression is added. Forward-only
fallback is UNCHANGED,avoiding duplicate use of projected target metadata.

## Causal handoff

Latch offsets after same-reserved-bus launch at the intended executable clock.
Zero-offset launched trips need explicit ledger rows;missing is not zero.
At the same PREPARE event as the native gaps,resolve the matched completed
predecessor and native AVL predicted physical follower by trip,bus,direction.
All relevant launch events must precede prepare in global event order. Missing
phase/identity or undelivered nonzero plans keep native bilateral025. Never use
future actual neighbours or a fixed trip-number relation as follower evidence.
Actual gap F retains already-realized nonholding time;release-only gaps are not
substitutes. Follower ETA and all calibrations remain frozen.

The needed identities exist in native bus fields (`forward_predecessor_trip_id`,
`pre_action_follower_bus_id/trip_id`),but V221 did not archive the latter or a
prepare-time phase ledger. Do not reconstruct them from query-time snapshots
one tick later. Any later runtime handoff must freeze these fields at prepare,
verify predecessor departure linkage and preserve native event ordering.

## Engineering gate and next decision

Only the standalone kernel and synthetic tests execute in V223. Compare zero
offsets with the unchanged V106 rule_action oracle across masks,gaps,clipping
and ties;test phase equivariance,real disturbance response,identity/clock
abstention and repeated-stop algebra. Failure blocks runtime work.

An own1s upper offset changes continuous bilateral demand by only .25s when
neighbour offsets are zero. It usually cannot change a seven-bin command;
threshold crossings are possible but no forced bin decrease is allowed. The
algebraic invariance assumes an ideal phase-only gap change;it does not prove
the frozen follower ETA is equivariant or downstream feedback is eliminated.

Passing tests admits only separately authorized PASSIVE runtime handoff and
equivalence preflight,not an intervention/future matrix. Preregister that
recording and its fixed path budget before execution. Require supported actual
command differences before any efficacy budget;zero authority stops progression
without gain escalation. Subsequent independent-day/future confirmation must
retain V107,fixed+bilateral,upper-only/lower-only controls and joint passenger,
actual CV2,readiness,original-public delay and completion criteria.

V107/main,original12 vehicles,V220 upper authority/cadence,actors/critics/duals,
V151/V146/V167 and all other frozen predictors/calibrations remain unchanged.
Run tests only via scheduler high/manual on node001--006;return a small log.
