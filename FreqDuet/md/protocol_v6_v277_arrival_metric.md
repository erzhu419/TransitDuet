# V277: arrival-metric alignment and separate current-action comparison

Keep V275/V276 sources,results and gate decisions frozen. Read the twelve V276
archives once on the server:reproduce all actual CV2 endpoints,attribute pooled
CV2 differences to station/hour moments and summarize1517 intervention queries'
resource forecasts. Forecast errors use executed-path labels,not counterfactual
reference labels. Repeated queries do not create independent samples.

The only model change is CV2:pool all observable nonterminal arrival headways'
count,sum and squared sum,then append only genuinely projected arrivals. Arrival
visibility is not delayed until service completion;actual same-tick observations
follow native recorder visibility. First arrivals create no gap;simultaneous
distinct arrivals create zero gaps. Current observed arrivals and completed
history are not counted twice. Terminal events remain in physical/resource
simulation but never in regularity. Assert prefix moments match the native
recorder. This is prefix-plus-lookahead CV2,not knowledge of the unobserved day.

Keep the indexed V276 native continuation,demand forecasts,costs,resource
constraints,seven legal bins,irreversible commands,horizon,beam4,variables6 and
128-evaluation cap unchanged in the arrival-metric beam arm. The default legacy
controller still calls its original chooser through a small selection method;
frozen source trees are not edited.

Implement current-action enumeration as a distinct arm with the same corrected
CV2/model/constraints. Evaluate native once and each other legal current bin
once,then use native continuation;at most seven model calls. Keep native on
equal cost;other equal improvements prefer the shorter bin. No future proposal
warm start. This is not output-equivalent to finite beam search. Under the
existing max(full,fallback) objective,the best feasible single current command
has model cost no greater than any admitted complete plan's objective;actual
traffic superiority remains an experimental question.

The user subsequently approved20 physical starts after checks and protocol
sealing:days2077001--2077004 x fixed+bilateral / frozen V275 / frozen V276 /
corrected-CV beam / corrected-CV current enumeration. Require metadata-only
nonconsumption check before launch. Eight sequential shards,at most8 concurrent
paths,high/manual on node001--006. Failures consume slots;no automatic retry,
training or main switch. Historical arms resolve their actual frozen modules
in separate interpreters;do not substitute the new controller for old references.

Use10000 paired-day bootstrap draws(seed2770201). Each corrected arm must pass
the original V276 gates versus fixed and V275,and additionally be non-harmful
versus frozen V276 on passenger cost,CV2,readiness and public delay(upper95<=0).
Require no extra unlaunched/incomplete trip on any day against each comparator.
Describe enum-minus-beam outcomes separately;model score does not promote it.
Failure closes the candidate without expansion;passing only permits reviewing
an independent-confirmation proposal. No threshold,support or search-budget
tuning on V276;no APC or frequency superiority claim. One actor,four independent
development days,no independent-future or actor-level inference.

Existing-archive readout t211986 reproduced all12 endpoints and all1517
intervention-query keys. Relative to V275,the two CV2 losses were+0.0029612 and
+0.0054171;their raw-variance terms were+0.0029099 and+0.0053556,with normalization
terms only+0.0000513 and+0.0000615. The leading hour buckets were[12h,13h) on
day2076001 and[3h,4h) on day2076004;these are additive temporal accounting,
not identified station-level treatment effects. Current-bus full-plan holding
bias was-43.55/-42.27/-42.51s on the three active days,return bias
-20.57/-26.06/-47.26s and return MAE93.18/79.59/101.29s. Full and fallback
forecasts were nearly identical here;later replanning remains part of the
executed labels. These residuals motivate retaining the diagnostic fields,
not simultaneously changing the frozen predictor. Small readout:
`../freqduet/results_freqduet/protocol_v6_v277_metric_alignment/readout/results.json`.

Admission:source60e0b7e682/t211990 passed27 synthetic/regression checks and7602
metadata files with zero day collisions. Separate-interpreter probes confirmed
V275 uses894ad14707,V276 uses indexed4a216af00c with frozen7023e2e784 flow,
and corrected arms use the new source. The earlier same-state synthetic timing
was6.268s/128 calls versus0.333s/7 calls,both retaining native with identical
objective;this is not a traffic-effect or full-day speedup result.

Startup correction:the first8 slots stopped before simulator initialization
because the nested matrix root mislocated the frozen runtime proof. They remain
failed/charged in `matrix/`;the user approved only8 identical replacements,
raising the total start cap to28 for the same20 unique paths. Use a separate
`matrix_repaired/` attempt and the outer experiment root for the legacy runtime
loader. Before further traffic,all five arm probes must call that actual loader,
not just import modules. Keep days,controllers and gates unchanged;no auto retry.
