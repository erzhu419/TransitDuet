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
