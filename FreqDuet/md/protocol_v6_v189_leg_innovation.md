# V189: road/entry-hour innovation, support before estimation

Original64 calibration days2951001-2951064 only;12 actors,both modes. V188
completed clocks and native-input replication are prerequisites. Keep V151,
V146,V172,3600s window,.1/.1 tails and main frozen. No V185 outcomes,new
confirmation,replay,training,point/interval fitting or residual-target scoring.

For each physical road traversal,define I=duration minus the frozen native
mean available strictly before entry. Missing entry support stays unavailable;
the innovation becomes available at arrival,not ready. On the first remaining
leg,match road/direction and reference-entry hour. Use actual entry only if
already observed strictly before the query;otherwise use query time plus the
recorded initial-residence forecast. Never match using a future realized entry.

Use only matched arrivals in the frozen one-hour window. Separate I from the
update already absorbed by the current native mean:remaining=I-update=matched
duration mean-current native mean. Verify this identity;the remainder is zero
when the matched cohort is the full native cohort. This is not a new predictor
or a renamed same-hour mean candidate;V153-V157 failures remain unchanged.

Reproduce V172 query/episode denominators and finite support. Report availability
with1 and9 completed physical traversals,and independent-day support separately
per actor/mode/road/entry-hour;19-day threshold unchanged. Nine is a diagnostic
count,not a replacement native minimum. Equal query within episode,equal episode
per path,equal actor/day;show actor/day margins and missing mass. No pooling
across actors to manufacture day support. Full-route extrapolation is not tested.

Seal protocol before reading support counts;tests and one archive preflight
before all1536 paths. Scheduler high/manual,node001-node006. Detailed cells stay
server-side;pull small summary only. Support cannot authorize fitting or establish
incremental prediction value;the next decision must reflect actual support.
