# V179: original-calibration prefix-risk replication

Only V166 days2951001-2951064, all 12 actors and old/handoff modes. Reuse sealed
V176/V172/V167 maps and V178 state partitions/weights. Match saved bounds,
finite masks, query coordinates and physical episodes to the sealed calibration
atom archives. No V177 confirmation records, new intervals, replay or training.

Replicate lower-tail and width profiles, day/component variation and separate
state-mix decompositions. For prefix diagnostics use the first field-complete
eligible frozen-V151 prediction per physical follower-trip/bus/stop/direction
episode. A label becomes usable strictly before the current query; retain only
the existing [query-3600, query) window, at least nine distinct completed episodes.
Do not update the predictor or add historical fallback for this diagnostic.

Inspect global and same-remaining-stop prefixes. Fixed zero-sign partitions and
running/service/residence/total residual means are descriptive features. Report
available mass and within-remaining-stop correlations with current total residual
and lower misses; actor/day bootstrap5000, seed20260803. Repeat on fixed first32
and last32 calibration-seed sets; report all channels, not a selected winner.
Missing signals remain unavailable. Frozen-band coverage is in-sample here;
these checks motivate a possible later preregistration, not controller admission.

All computation high/manual on node001-node006; only small summaries retrieved.

## Result and next decision

Completed1536 paths, six tests and frozen-archive checks passed. Clearance day
variation92.67/92.53%; within-distance share97.21/97.23%. Residence bias at3/5/6
remaining stops: about-11.9/+13.7/+32.0s. V176-minus-V172 score remains worse:
clearance+4.55/+4.72s; hour13 moving-launch+5.71/+5.81s; all95% CIs above zero.

Prefix support100% global,98.53-98.59% same-distance. Total-prefix correlations
remain near zero within distance strata. Global service/lower-miss correlation
is+0.072 to+0.076, positive95% CIs in both32-day halves. Same-distance service
is weaker(+0.057 to+0.063); first-half CIs cross zero in three of four cells.
Running/residence signals do not replicate consistently. Calibration-only
association is not independent validation or evidence of control benefit.

Next: only test service-prefix incremental information within fixed observable
strata and its relation to both tails/score cost on these64 days. Do not fit,
translate bands by total bias, shrink the envelope or start confirmation days.
All maps, predictors and main policy remain frozen.

High/manual on node004/005: t210655-t210662. Initial t210654 failed on an incorrect
data-root path before archive reads. Pulled only827184-byte result plus receipts:
freqduet/results_freqduet/protocol_v6_v179_calibration_prefix/audit_354867e01f/summary/status.json.
