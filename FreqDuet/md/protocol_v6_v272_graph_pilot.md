# V272: authorized unique-path event-graph pilot

The V271 native no-op passed. Frozen graph0a68748858,forecasts,seven bins,
12-vehicle pool,exact360 public upper,solver limits and all service constraints
remain unchanged. V107 bilateral is the reference;the two graph arms differ
only in APC/location-specific versus homogeneous onboard exposure cost.

The user authorized the original24-path release,then explicitly approved
deduplication:the sampler seeds27011001/2 only alter categorical action uniforms,
not scenario_seed. Deterministic three-arm paths cannot obtain an independent
traffic future from that parameter. Use sampler27011001 once,actor anchor12011,
consumed demand days2071001-4 and three arms:12 unique physical paths,not24.
No independent-future repetition claim. No actual pilot path has run at sealing.

Inference uses4 paired demand days and10000 day bootstrap draws,seed2700201,
shared resamples for all contrasts. APC must have passenger upper95<0,and
actual CV2/readiness/total public delay upper95<=0 against BOTH bilateral and
the homogeneous-cost control;no extra unfinished trips on any paired day.
The old two-future-half requirement is inapplicable to deterministic aliases;
fresh traffic-day/actor confirmation remains separate and unreleased.

Every graph arm recomputes at each native query. Execute only the current legal
first request;later graph proposals remain revisable. Already-issued same-tick
commands and ongoing service/holding stay fixed. Each request must match its
actual native command acknowledgment,and cleared service must execute all H.
Unsupported input falls back to the unchanged bilateral request,not a relaxed
capacity or support estimator. Common day contracts and logical-event uniforms
must align across arms. The first baseline additionally reproduces V271 exactly.

After server-only synthetic execution-chain checks,launch8 high/manual scheduler
units. Each contains1 or2 sequential fresh-process paths,max8 concurrent.
All12 slots are reserved before submission;failure remains charged,no automatic
retry,no replacement or scope expansion. Each unit attempts its fixed next item
even if its previous independent process failed. Raw traces remain server-side.
Only small summaries return locally. Main,actors,duals and training remain frozen.

This is a development screen,not a field deployment or a realtime guarantee.
Observed supported-query p95 is about2.73s. Do not repair forecasts,weights or
windows on pilot outcomes;failure closes this frozen candidate. A development
pass permits proposing independent confirmation,not switching the main policy.

Execution source07e565545a:server task t211639 passed7 synthetic checks,exit0.
Matrix reference p0 in t211640 exactly reproduced V271,including5200 requests
and acknowledgments,38930s executed H and all query/action/timing endpoints.
Eight high/manual units t211640-t211647 completed all12 reserved paths,exit0,
without failed/replacement paths. No unlaunched or incomplete trips remained.
Bridge progress retains its V271_NOOP_QUERIES label;the V272 wrapper executes
the current selected action through the native policy output and checks its receipt.

## Result And Decision

High/manual read-only analysis t211720,node004,exit0,source9ef6acdd3f used the
original07e565545a aggregate and10000-draw paired day bootstrap. Both gates failed;
full intervals and four-day values are in summary/results.json.

APC versus bilateral:passenger+832 person-s/day,CI[-5789.75,11855.25];readiness
and public delay worsened on all4 days,total delay+122.5s/day,CI[70,204.25].
Versus homogeneous cost:passenger-7385.75 person-s/day,CI[-16053.25,-89],but CV2
increased on all4 days,mean+0.0002474,CI[0.0000990,0.0003666].
APC changed368/20800 requests(156 up,212 down),net+255s in the same visited
states. Actual H was+206.25s/day versus bilateral;fallback92.51%(homogeneous
92.13%),mainly capacity censoring. Details:summary/executed_diagnostics.json.

Decision:close this frozen candidate,no confirmation or outcome-driven retuning.
Retain the partial APC load-cost signal and strong rule comparator. Any next
method needs a separate design and new development data;main/training stay frozen.
Limitations:4 consumed days,one actor,no independent futures. Same-state H
accounting does not identify the cause of the actual full-day constraint losses.
