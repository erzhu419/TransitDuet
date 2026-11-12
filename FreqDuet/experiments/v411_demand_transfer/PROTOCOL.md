# V411: Frozen Demand-Pressure Transfer

Evaluate the exact V410 capped controller, fixed10, and within-day transition
rule at 1.0x and 1.35x demand. Reuse the sealed V410 counts, zero tilt,
policy seed, legal masks, predictor and its original OD prior. No adjustment
to either controller or recalibration of the rule. Keep the same 12 buses,
corridor, 262 public trips and fixed timetable. This varies demand pressure
on one supply configuration; it is not fleet-size or route transfer.

Eight fresh days per regime, three arms each, at most 48 paths. The first
day of each regime is a formal cell and a physical/CRN preflight for the
remaining 14 day tasks; no launch or data duplication. Release based only
on complete paths, physical accounting, pairing, legal commands and the
original horizon, never favorable effects. No retry or replacement.

Apply V410's paired bootstrap procedure separately to each regime. The
holding and three bin-share intervals must fit +/-3 s/trip and +/-2
percentage points, then require journey improvement and actual CV2,
readiness and public delay nonharm as in V410. Report fixed10 contrasts,
waiting, onboard, unboarded and runtime even when matching fails. Do not
pool regimes, retune the reference, or claim a matched mechanism where
the dose or bin composition did not transfer. These are bounded
demand-transfer screens, not a new universal performance claim.

Scheduler high/manual on node001--006; no training, policy switch, or
large local transfers. The completed first-day paths count toward the
same budget. No changes to V408--V410 or the paper mainline.
