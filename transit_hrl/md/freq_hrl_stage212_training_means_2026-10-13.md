# Stage212: conditional means for the remaining training targets

Stage211 bounded/shrunk corrections failed the full development gate.
Upgrade the remaining360 singleton states to eight independent conditional
futures each:120 old states in20 already-opened scenes;240 fresh states in
40 Stage187 replica4-7 scenes. No new prefixes or reserved scene labels.
Original600 causal inputs,240 mean8 evaluation states,47 outer folds,
cost heads,menu and operating threshold remain unchanged.

Separate actual draw_count8 from optimization sample weights: retain
the original240 weights8 and360 weights1. Otherwise target precision
would be confounded with reweighting. Existing evaluation mass records
are copied unchanged;future panels remain the original ones.

Collection:2880 draw tasks,23040 native episodes,each61380 ticks;no
policy updates. First20 cells cover both origins,both roots,all5 regimes;
release2860 only after a server-side qualified preflight report.
Unbound scheduler,node001-node006,1CPU/1536MB native task. Rollouts,
draw labels and prepared targets stay on servers;small receipts only.

Run:`native_transit_training_means_stage212_development_20261013_r1`.
Collection source`a04a546838`;preflight`t148478-t148497` and server report
`t148498` all done/exit0.160 native episodes qualified,max RSS727.7MB.
One2376-byte report pull;no raw,labels or checkpoints pulled.
Remaining`t148499-t151358` released from source`a132692ab2`,unbound;
initial snapshot149 running/265 launching/2446 queued across five eligible
nodes. Source pushed;23 collection/continuation tests passed.
GLM and bounded-head loss/scaling now consume explicit sample_weight;
37 targeted tests passed,including exact old-weight model reproduction
when actual draw depth changes. Eight tests passed after matching the
scheduler results interface;full collection and preparation remain pending.

Next: refit outer/inner symmetric GLMs on the new training masses;
repeat fixed B1/ridge0-or0.1 heads with original inner validation states
and sample weights. Compare against both newly fitted GLM and Stage211.
No threshold search or larger neural capacity in this comparison.

Limits: conditional development data and one-action continuations;
not independent confirmation,full learned upper policy or end-to-end HRL.
