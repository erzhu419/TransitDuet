# V180: service-prefix information within fixed observable states

Original V166 calibration64 days only,12 actors,old/handoff. Frozen V151/V146,
V167/V172/V176,3600s window,.1/.1 tails and main controller unchanged. No fitting,
new confirmation, replay or training. Reuse V179 strict-completion prefix and
episode weights; reproduce every path endpoint and its service correlations.

Fixed nested strata: remaining stops; then native stage,predicted horizon,
direction,target stop,existing clearance-elapsed bands; then existing key-day,
native-support,survivor-count and support-age bands. No data-selected cutoffs.
Report supported mass and day support without dropping sparse strata or
encoding undefined variation as zero. Within-stratum covariance/correlation,
crossed actor/day bootstrap5000 seed20260803, all64/first32/last32, both histories.

Outcomes: total/service residual, lower/upper miss, frozen candidate score,
V176-minus-V172 score, added width and saved upper penalty. The score-covariance
identity must hold. These are descriptive associations, not predicted control
gains or independent interval validation. Next decision depends on direction,
split stability, remaining variation and cost-risk alignment, not a picked cell.

All work high/manual via scheduler on node001-node006; pull small summary only.

## Result and next decision

Completed1536 paths; five tests, all path endpoints and V179 service-correlation
points/CIs reproduced. Source87c79be22d, tasks t210667-t210674 on node004/005;
only333088-byte summary plus small receipts retrieved.

Within full observed-support strata, global service-prefix/lower-miss r is
.0734-.0778 across both modes/scopes, positive95% CIs in both32-day halves.
Service-component r is.334-.340, but total-error r only.070-.077. Same-distance
prefix is weaker; clearance first-half lower-risk CIs still cross zero.
On all64 days, upper-miss, candidate-score, added-width, saved-upper-penalty and
envelope-score-delta correlation CIs all cross zero. Score-covariance identity holds.

Limitations: calibration-only association, not a learned forecast or policy
benefit; about one fifth of full-state scored mass has fewer than19 distinct
support days. Retained-variance ratios are descriptive, not admission gates.

Next: preregister a low-capacity lower-tail conditioning design and inspect its
support before any fitting, keeping V172 as an unchanged score/dual-tail
comparator. Do not use this signal to gate/resize the upper envelope. No mapping,
predictor, window, tail, main policy or confirmation-cohort change was made.
Result: freqduet/results_freqduet/protocol_v6_v180_service_increment/audit_87c79be22d/summary/status.json.
