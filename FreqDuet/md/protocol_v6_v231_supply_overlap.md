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
