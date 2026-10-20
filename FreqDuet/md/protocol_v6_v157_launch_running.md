# V157: consumed launch-query running accounting

Read only the7 launch queries from V156,identified by its paired saved-cell
readout. Reconstruct the original full/single/none points and require all saved
launch endpoints/support to match before interpretation. No new forecast arm.

Record per-leg native completed-prefix support,query/predicted-entry clocks,
frozen prior-day moments,selected running mean,and retrospective true running
duration. Preserve partial-leg handling. Reconcile leg totals and full
running/service/residence/cancellation accounting. First inspect the two actors
on2451015 where both full and single worsen,then the other consumed launches.

All seven are outcome-selected diagnosis,not fresh confirmation;all eight prior
days are shown,not scored or ranked as candidates. No prior refit,weight change,
replay,training,main/V151/V146 change,window/tail change or gate reinterpretation.
Scheduler,node001--006,small summary only. No additional trajectories collected.

Source6b2aedd1b2,t204549,node006;2 tests and all7 original endpoint checks passed.
Only118548-byte summary pulled. On2451015,12161/933059 share the same follower
and route:19->18 actual168s,native107.9s,eight-day115.322s,single97.45s. The
eight-day gains on that leg are outweighed by shorter predictions on21->20 and
20->19,total running MAE+7.488/+7.377s;single+15.806s. Service/residence errors
remain unchanged and nearly cancel. These four mode/actor queries are one day,
not four independent shocks. On2451023,the same19->18 leg instead takes85s.
On2451027,forward2->3 takes98s:eight-day103.594s versus single75.8s;its frozen
day means span72--186.75s. The earliest prior day is not uniformly representative.

Decision:retain eight-day points as an unpromoted candidate. Next diagnose
cross-day prior reliability and running/service/residence error dependence on
completed prefixes before designing full-ETA uncertainty. Do not choose a prior
day or change shrinkage from these cases;no further point-bias patch is justified.
