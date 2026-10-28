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

## Completed development readout (2026-10-28)

All12 paths and aggregate t211926 completed. Reuse nine original results and
three authorized indexed replacements; the full small summary is in
`../freqduet/results_freqduet/protocol_v6_v276_indexed_replacements/summary/results.json`.
All event-uniform comparisons have zero mismatches. Every path completed262
trips, verified5200 commands and executed all commanded holding. No training,
confirmation or main-policy change occurred.

Paired four-day differences (candidate minus comparator;95% day-bootstrap CI):

| Metric | fixed360+bilateral025 | Frozen V275 |
| --- | --- | --- |
| Journey, seconds/person | -3.651 [-6.557,-0.745] | -3.577 [-6.984,-0.327] |
| Actual CV2 | -0.001897 [-0.004257,-0.000326] | +0.001036 [-0.002437,+0.004189] |
| Readiness delay, seconds/trip | -4.080 [-8.149,-0.857] | -5.848 [-10.580,-1.568] |
| Public launch delay, seconds/day | -1069.00 [-2135.00,-224.50] | -1532.25 [-2772.00,-410.75] |

The strong-baseline joint gate passes. The V275-preservation gate fails solely
on CV2 non-harm;the combined preregistered decision remains
`close_candidate_no_expansion`. Retain the positive traffic evidence without
changing the gate,expanding confirmation or switching main.

Days2076001/3/4 improve all four strong-baseline metrics;day2076002 has zero
supported queries and exactly reproduces bilateral. Across four days,V276
supports11762/20800 queries and executes1517 same-state overrides. Relative to
the strong baseline,holding falls2.166s/trip,waiting falls50191.75 person-s/day
and onboard time falls23564.75 person-s/day. Relative to V275,holding falls
6.784s/trip,but CV2 worsens on days2076001/4. This is a partial tradeoff repair,
not preservation of every V275 benefit. Future proposal matches are not an
execution-success rate or an identified causal explanation of the endpoints.

Limitations:four development demand days,one actor anchor,no independent-future
replication and no APC/homogeneous or frequency ablation. Baseline continuation
alignment now has service evidence,but APC-specific and frequency-separation
claims remain unestablished. Readiness and public delay represent the same
operational loss at different scales,not independent benefits.
