# V88: locate common-state lower holding differences

V87's symmetric mean/response factorial attributes the passenger/readiness
deterioration mainly to late centered-logit response, with mean preference
partially offsetting it. Saved expected-action transport further separates
+2.3521 s/decision at common states from -0.8105 s/decision visitation change.
These are policy function and occupancy comparisons, not physical mediation.

Replay V87's native early/late corners only: both original actors, all 16
already consumed evaluation days, three matched samplers, 192 labelled paths.
Keep B200 upper/deployment context and both frozen lower actors. Every replay
must reproduce V87's full-day metrics, action counts and expected-action means;
smoke also compares complete event and actual holding paths with uninstrumented
replays for both actors and both corners. No new controller or mean adjustment.

At each current observable state evaluate both native laws and partition their
expected holding difference: missing bilateral evidence; positive extra holding
with predicted local regularity gain; positive extra holding without predicted
gain; no extra holding. Record onboard passenger command exposure, current
return-terminal backlog/ready pressure and accumulated commanded trip holding. No future
label enters this partition or either deployed action. The local gain formula
is the existing observable surrogate, not a validated causal action value;
zero current backlog does not imply future vehicle-readiness safety.

Group contributions are divided by the entire path's decision count, not by
group count, so they add exactly even when a group is absent or visitation
changes. Report common-state-law and visitation contributions, conditional
fractions and day-paired uncertainty for both actors separately and together.
This distinguishes response allocation from changing exposure to contexts;
it does not turn observational categories into a policy gate or new held-out
evidence. Main, actor/critic/dual weights remain frozen. All tests, native smoke,
replays and aggregation use scheduler on node001-node006; retain detailed data
server-side and retrieve only small summaries.

Source `0a2c269836`; smoke `t166489` passes eleven tests and all eight full-day
checks: both actors, early/late native endpoints, expected-action accounting,
complete event uniforms and actual holding paths. It releases shards `t166491-t166494`, covering
`[0,24)`, `[24,48)`, `[48,72)`, `[72,96)` on node002/004/005/006, 24 workers
each. Aggregate `t166496` waits for the exact four shard markers. Context
results remain pending; no controller change is admitted.

All 192 labelled paths and aggregation completed, reproducing V87 endpoints.
Retrieved only 58,003 bytes. Group transport in seconds per decision:

| Group | Common-state law | Visitation | Net |
| --- | ---: | ---: | ---: |
| Missing bilateral | +0.16805 | -0.11517 | +0.05289 |
| More holding, predicted gain up | +0.86199 | -0.34820 | +0.51380 |
| More holding, predicted gain not up | +1.62898 | -0.41053 | +1.21845 |
| No more holding | -0.30690 | +0.06341 | -0.24348 |

The no-gain group covers 33.15% of late and 33.47% of early decisions; its
pooled conditional common-state CI is [1.60725,1.65275], and actor-specific
effects are positive. Missing-evidence
net effects have opposite actor signs. Weighted review `t169480` passes four
tests: no-gain versus gain-up extra-holding-weighted onboard load is 25.90 /
31.91 passengers, current return pressure 1.333 / 1.423, and prior commanded
holding over target 0.571 / 0.451. These are weighted current-state moments,
not joint frequencies, realized passenger costs, or evidence of highest load/
pressure concentration. The next step is a prospective conditional actor
factorial on new days, not promotion from observational surrogate labels.
