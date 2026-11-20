# FreqDuet Manuscripts

The [Public Transport revision](public_transport_freqduet_v7_202611/README.md)
is the current fixed-fleet joint-control manuscript. V504 is the main
corrected comparison. V505 isolates stop holding with the upper actors
frozen, adds a matched rolling benchmark, and records the circulation
failure and field timing check. The mainline controller is unchanged.

The [TR-C v7 draft](trc_freqduet_v7_202611/README.md) studies joint dispatch
and stop holding in V500--V503. It reports as-run simulation trade-offs
and the corrected routing comparison, which did not meet its joint criterion. It is distinct
from the fixed-timetable capped holding study below.

The [state-dependent capped holding
manuscript](tits_capped_holding_20261116/README.md). Its primary evidence is
the V410--V412 fixed-timetable, fixed-fleet comparison with the within-day
transition rule. A common 10 s ceiling, matched aggregate commanded holding
and matched 0/5/10 s shares support a bounded action-placement result.
Nominal demand misses holding equivalence; slower running misses the
joint circulation criterion. Neither is reclassified as a pass.

V409's whole-day mixture, V418's guarded forecast extension and the earlier
budget and frequency studies are supporting evidence, not grounds for a
learned-supervisor or frequency advantage claim. The experiment main policy
has not changed.

The previous [turnover-coordination manuscript](tits_turnover_submission_20261110/README.md)
and the earlier dated T-ITS and Protocol-V6/TR-C packages remain historical
snapshots.
