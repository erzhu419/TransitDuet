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
Calibration completed: t195334 exit0,sealed from5635 native-finite clearance
queries,all768 cells/all32 days,zero unresolved labels. Fixed dL=3s,dU=0s;
only1520-byte seal receipt fetched to release confirmation.

## Confirmation

t195335--t195340 completed without retries; all768 confirmation paths resolve.
Both modes pass all three coverage gates and the paired clearance-score gate.
Episode-first, equal actor/day crossed estimates, native -> calibrated coverage%:

| Scope | old | handoff |
| --- | --- | --- |
| Clearance service | 61.42 -> 91.36 | 60.98 -> 90.15 |
| Hour13 service | 65.72 -> 90.37 | 64.70 -> 89.41 |
| All-day service | 85.12 -> 85.82 | 85.07 -> 85.73 |

One-sided95% coverage lower bounds(old/handoff): clearance88.95/87.80%,
hour13 88.09/87.16%,all-day85.37/85.30%; all exceed80%. Clearance score
changes are -4.929s[-5.773,-4.096] and -5.062s[-6.052,-4.097](two-sided95%).
Clearance widths13.50->15.72s and13.60->15.86s: improved score despite widening.
All-day finite availability54.747/54.778% remains identical; no finite unresolved
labels. Read-only audit ade0291e0b/t199000/node006 exit0,5 tests pass: every
path/scope preserves request/support counts,label states and abstention reasons;
active-service and ready negative controls are identical. Both modes have all12
actor coverage margins above80% in every primary scope,and all12 actor/all32 day
score margins improve. Clearance actor minima88.40/89.05%;hour13 87.65/87.01%.
Each mode still has3/32 day margins below80% in clearance and hour13;clearance
minima72.98/72.12%. These are descriptive margins,not new gates or per-day
guarantees. Only48288 bytes of final/audit summaries fetched this analysis;
full marginal rows,traces and checkpoints stay remote.

Next: freeze the sealed3s/0s calibration. Audit full same-vehicle follower ETA
and source-switch gap-change errors on saved paths before any control admission;
retain native low-support abstention. No point/window/tail or main-policy change.
