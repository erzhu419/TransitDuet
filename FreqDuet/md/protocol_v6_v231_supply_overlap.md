# V231: read-only supply/holding overlap

Use all8 accepted V230 paths:2 actors x2 consumed demand days xV107/upper.
No replay,training,fit,new future,shadow execution or main-policy change.
Inputs are read-only;details stay on server,only the compact summary returns.

At each original query,use next-pass uncovered posted obligations at the
opposite terminal as pressure. Incoming distance rank is descriptive only,
not a return forecast,reservation or slack. Keep original prepare/query clocks.
Report native positive/zero holding and supported/missing evidence separately.
R(0)-R(h)>0 denotes only local predicted compensation;record availability of
smaller feasible bins with delta R<=0 without choosing or executing them.

Measure actual holding from original departure minus query prefix and close
query/trip totals to the executed ledger. Match the FIRST later same-bus launch
to query-time uncovered,covered-due,future-posted or absent obligations. This
future correspondence is a label,never an observable feature or anchor filter.
Count unique cycles/next launches separately from repeated query exposure.
Final returns without another launch remain missing outcomes,not zero delay.

Compare upper vsV107 by decision occurrence and physical bus/trip/next trip;
retain unmatched/changed identities. Report all4 actor/day contexts,not a
query-level CI. Decompose each matched next-launch time difference into inbound
launch,executed holding,other trip time and observed post-return waiting.
Observed terminal waiting is not assumed recoverable slack;the identity is
accounting,not causal mediation. No action/value model or selector is defined.

Next decision:use observed overlap and native-response/supply-chain evidence
to specify a separate persistent-coordination hypothesis and budget,or name
the missing causal evidence. This analysis alone admits no intervention.

Completed source2dc70e4bf7,t211297(high/manual,node004),exit0;5 tests and all8
paths passed holding/supply accounting. Only the34KB summary returned locally.
Each path has721--898 pressure-positive queries;82--155 link to the then-uncovered
obligation,most instead supply a future posted trip. There are46--67 unique
cycles that match an uncovered obligation at least once;each next launch is
exactly1 native tick after return. All supported positive requests have local
benefit versus0,but none has a smaller feasible bin with delta R<=0.
On day2151002,both actors have33 later next launches,total+249s:inbound launch
124s,holding120s,other trip time5s,post-return waiting0s. Whole-day holding falls
35/25s,yet public-total delay rises221s. All4 contexts worsen CV2/public delay.

Next:separately preregister persistent vehicle-cycle coordination using current
posted obligations,positions and executed prefixes,not a terminal-wide boolean
cut or a strict local-R-only guard. Keep actual next-trip labels out of the
rule;no intervention budget is opened here.

Limitations:2 consumed days/2 actors,not independent efficacy. Local R and
query-onboard-times-holding are immediate proxies,not actual CV2 or passenger
delay. Tight observed return/launch timing does not prove a removable action.
The scheduler started a second instance on node006 after node004 completed;
it stopped at the existing output directory before trajectory reads. Preserve
both logs and the completed node004 summary;no replay or extra archive scan.
