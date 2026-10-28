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

This stage authorizes only existing-archive analysis and synthetic checks via
scheduler high/manual on node001--006. Zero new traffic starts,no training,no
main switch. Proposed next bounded development comparison:four new demand days
x fixed+bilateral / frozen V275 / frozen V276 / corrected-CV beam / corrected-CV
current enumeration. Seal actual day IDs,budget and paired gates before launch;
this document does not launch or authorize those20 physical starts. No threshold,
support or search-budget tuning on V276;no APC or frequency superiority claim.
