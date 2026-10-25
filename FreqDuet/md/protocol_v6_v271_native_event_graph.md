# V271: V270 native reference and no-op admission

Uses the TWO existing V270 engineering slots,not a new budget. Fixed actor
anchor12011,consumed demand day2071001,new runtime future27011001. Reference
must reproduce archived exact360+bilateral025 actions,timing and endpoints before
no-op is submitted. No matrix,intervention,training or main-policy change.

The bridge reads APC counts and AVL route cursors. Run labels use completed
departure-to-arrival segments; APC alighting is incoming+boarded-outgoing,without
reading destinations or latent queues. A visit becomes training support only
after departure(or actual terminal return),with completion AND label clocks
strictly before query. First arrivals have no inferred gap; zero gaps are kept
and mark the frozen positive-gap estimator unsupported,not replaced or deleted.

Stopped phases use last_station;TRAVEL uses next_station and remaining observed
distance. H is since actual trip launch,never credit. Already-issued same-tick
commands become fixed departure nodes,distinct from previous-tick DWELLING
countdowns. Only the current native query remains uncommitted.

No-op actually calls the frozen rolling solver on supported queries,records its
latency and shadow proposal,and returns the SAME original rule tensor. Native
actions,arrivals,departures,public/actual dispatch,query identities and event
uniforms must be exact;endpoints,physical contract and command acknowledgments
must also match. Require at least one model query and one modeled pending lock.
Unsupported/censored fractions and all-query/optimizing-query latency are reported.

Reservations are written before scheduler submission;entry creates an exclusive
physical-start marker. Failure consumes its slot. No automatic retry or fallback
replay. Only high/manual scheduler jobs on node001-006. Raw traces stay server-side;
local artifacts are small acceptance summaries. Synthetic checks consume0 slots.
The budget lives under V270/engineering;V271's output is a direct child of
results_freqduet,as required by the existing verified-runtime loader. This does
not create a fresh two-slot budget in the V271 output directory.

Admission83d681d0ae:high/manual t211631 passed7 synthetic checks;t211632
reproduced all5200 native queries/commands and cached actions,timing,endpoints.
Reference wall76.43s,callback mean1.43ms/p951.61ms. The second slot,t211633,
failed at its first snapshot:the bridge incorrectly equated max_agent_num with
physical fleet size. Native max_agent_num is a policy buffer of at least30;
the physical fleet remains12. The regression fixture now uses that native layout.
Both physical slots are consumed. No-op equivalence,its endpoint comparison and
solver latency remain UNCONFIRMED;the failed partial path is not a zero effect.
At that point only a synthetic correction check was permitted;no retry was made.

The user subsequently approved exactly ONE extra no-op after the padded-agent
regression passes. Its independent noop_fix reservation raises the engineering
cap to3;the failed noop and both original starts remain charged and unchanged.
Reuse the accepted83d681d0ae reference,not a fresh reference run. No additional
scene,future,training,matrix or main-policy change;failure remains charged with
no automatic retry.
