# V146: independent clearance-service calibration

Preregister before new outcomes. V145 is consumed diagnosis only, never fit or
threshold-selection data. Calibration days2551001--2551032, sampler1461000000+i;
confirmation days2552001--2552032, sampler1462000000+i (i=0..31). Both use the
same12 B200 actors/checkpoint199 and frozen old/handoff controls:768 paths each.
Newness is checked against the archive usage ledger, V145 registration and prior
source generators. All simulation, tests, fitting and inference run on node001--006.

One candidate: on native-finite service requests at/after the fixed demand cutoff,
learn shared nonnegative dL,dU from new calibration days only. Scores are
max(L-y,0),max(y-U,0); each offset is its fixed weighted empirical90th percentile.
Weights are equal across days, then nonempty actor/mode cells, episodes and
queries. All32 days must contribute; every actor/mode needs at least9 days.
Unresolved finite outcomes block sealing. Empty cells/support remain explicit.
Seal once, then release confirmation. No fitting or score inspection on confirmation.

Candidate bounds are[max(0,L-dL),U+dU]. Native support<9 still abstains; no pool
expansion or substitution. Ready and pre-cutoff service bounds stay identical.
V140/V141 point formulas,3600s/all-past window,V142 ranks and10%/90% levels stay
fixed; learned corrections do not feed controls or overwrite native artifacts.

Confirmation: episode-first, equal actor/day, existing5000 crossed draws.
Require one-sided95% lower coverage>=80% in both modes for service clearance,
hour13 and all-day. Also require the two-sided95% upper CI of paired clearance
interval-score change<0 in both modes. Score=width+10*(lower/upper miss distance),
fixed by80% nominal coverage; width is reported, not rescued by a tuned cap.
Incomplete crossed support is inconclusive; missing labels are not zero-filled.
Report baseline/candidate availability, coverage, tails, widths and paired score;
ready remains a negative control. No interim confirmation readout or gate tuning.

Limits: empirical phase calibration, not an exchangeability theorem, full ETA
validation or control efficacy. No RL training, policy switch, native point,
window or tail changes. After a failed/inconclusive confirmation, preserve it;
any revised method needs a separate preregistration and fresh confirmation days.
Only preflight/seal/final small summaries may sync locally; raw paths stay remote.

Execution441553aa65 was pushed before new paths. Preflight t195327/node006 exit0,
10 tests pass; the unchanged collector reuses its V145 exact-replay proof.
Calibration t195329--t195333 (71/22/53/45/66 workers), seal t195334;
confirmation t195335--t195339 and aggregate t195340 wait for the local small
seal_summary_441553aa65/done.json. No confirmation outcome may inform fitting.
