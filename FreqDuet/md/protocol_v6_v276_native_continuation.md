# V276: native bilateral continuation

Freeze V275 source 894ad14707 and its failed joint gate. Change only the model's
baseline continuation: reuse native predecessor/departure matching, strict
spatial follower selection, speed/service proxy, PREPARE clock and legal-bin
selection. Observe actual estimator inputs at PREPARE before later callback
state changes. The frozen optimized runtime uses the inline fleet roster,
including same-tick changes by earlier physical vehicles. Future
AVL is projected from the unchanged completed-prefix running forecasts; it is
not actual future telemetry. No flow, cost, horizon, beam, budget, fleet, masks,
action, training, main-policy or acceptance-margin changes.

Server synthetic checks must pass before one passive baseline path on consumed
day2075001. Reuse V275 p0 as reference. Require exact queries/actions, event
uniforms, arrivals/departures, dispatches, physical contract and endpoints, plus
5200 same-state rule checks. Failure consumes the single path; no automatic retry.

After engineering admission,12 development paths: new days2076001--2076004 x
fixed360+bilateral025 / frozen V275 APC / V276 APC. One actor12011,sampler27011001;
four independent days,not independent futures. At most8 processes,high/manual
scheduler,node001--006. No homogeneous or frequency arm until service success.

Use10000 paired-day bootstrap draws(seed2760201). Versus strong baseline require
passenger-cost upper95<0 and actual CV2/readiness/public-delay upper95<=0,without
extra unlaunched or incomplete trips on any day. Versus V275 require readiness
and public-delay upper95<0 and passenger/CV2 upper95<=0. Do not count readiness
and total delay as independent evidence. Do not revise thresholds after results.

Record predicted reference/full/fallback return,next launch and remaining holding
for accepted current changes,with query/bus/trip keys and actual outcome labels.
Actual labels describe the executed branch,not a realized counterfactual baseline.
The holding field excludes already committed commands and their common residual;
it records new controllable requests,not all remaining physical hold exposure.
Small JSON summaries only return locally;raw traces remain on server. A failed
joint gate closes this candidate;passing permits proposing independent confirmation,
not switching main or claiming APC/frequency superiority.

Admission completed: source7023e2e784, t211826 passed 92 server checks; 7565 metadata
files showed zero new-day collisions. Earlier synthetic-only attempts exposed
the frozen runtime's inline AVL API and a fixture missing its follower's next
published trip;both were corrected before any physical start. t211827 then
passed 5200 same-state rule checks and exact cached V275 p0 trace/endpoint parity.
High/manual t211829--t211836 launched on node001/node004 (max8 concurrent).
t211837 waits for their eight small receipts. The launch snapshot has p0--p7
started, no completed development paths and no failures. No control result yet.
