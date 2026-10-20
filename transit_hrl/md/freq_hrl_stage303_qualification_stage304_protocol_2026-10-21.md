# Stage303 Qualification And Stage304 Evaluation

Stage303 t208491-t208520: 30 done/exit0, zero retries; all 4,590 native
episodes passed their execution-time weight, timetable, scoring and window
ledger assertions. Read-only audit t208523/node005 also passed. Its 7,339-byte
summary rechecked all 640 P8/S8 draw records, six-action identities and cost
decompositions, all prior-action histories and all ten nontrivial-history
identity witnesses. There are 320 distinct inner seeds, intentionally shared
between controllers for paired sampling; P8 and S8 are disjoint.

All controllers start from the same state. Actual first-action divergence
occurs only at 397/hour_burst: raw versus service-only and raw versus joint
both produce different second contexts. Identical first actions give
identical second contexts in all remaining comparisons. Thus native feedback
is exercised, but policy divergence is sparse; no pilot performance claim.

Stage304 freezes profile0, probes8/9, P8/S8 and the Stage301 cutoffs. It uses
320 new full native scenes, 32 per root/regime, with raw/base/service-only/
joint paired in each scene: 1,280 jobs, 219,520 native episodes including
inner sampling and identities. No fitting, actor update or pilot outcome
selection. Full-scene seeds2080000000 and inner seeds2030000000 namespaces
are disjoint from the previous cohorts and from each other.

Primary: service-only versus raw cost/wait/service improve under both
stratified episode bootstrap95 and paired normal95, cost/wait improve in
both 16-scene halves, and at least half the base wait gain is retained with
paired-margin CIs. Joint/raw and joint/service-only are separate claims.
Both CIs describe the same independent episode sampling; neither is the old
fixed-prefix conditional MC interval. This is a two-decision fixed-lower
evaluation, not all44 decisions or end-to-end learned HRL. 66 tests pass.

Execution: first freeze t208533 failed on a local-only metadata path, before
any formal simulation; its failure receipt is retained. The corrected r2
freeze t208538/node001 completed done/exit0 with the same rules and seeds.
Formal evaluation t208544-t209823 is submitted: 1,280 jobs, 1 CPU / 1,536 MiB
each, unpinned over node001-node006. Full profiles and raw trajectories stay
server-side; only qualification and freeze summaries were pulled.
Physical worker t208546/node002 is alive at 98% CPU / 533 MiB. The initial
snapshot records 15 running, 108 launching and 1,157 queued jobs, across five
active nodes, without failures. Formal performance results have not been read.
