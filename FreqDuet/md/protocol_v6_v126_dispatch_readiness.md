# V126: consumed dispatch eligibility/readiness split

V125 localized day 2231004 forecast-arm regularity loss to actual launch already,
with subsequent downstream amplification. Read the same 32 paths only. Rebuild
first eligibility as actual launch minus recorded fleet-readiness delay. Preserve
actual predecessor order; split each actual gap into eligibility gap and change
in waiting delay. Reproduce original dispatch/readiness and V125 effects first.
Report exact covariance allocation/cross terms and crossed actor/day intervals,
plus direction x V125 fixed four-hour cohorts and first clock differences.

First eligibility includes prior actual-release constraints and is sticky; it is
not an independent upper action or the original timetable. No return timestamps
are inferred. This follow-up is descriptive on consumed development paths, not
a new gate or causal intervention. Tests/analysis via scheduler, CPU conda on
node001--node006; only summary JSON returns. No rollout, training or main switch.

## Outcome and next decision

Source fcc9cb64c8, t195172 on node006, one worker, actual child exit 0; two
tests and all 32 original dispatch/readiness paths plus V125 launch effects pass.
Manual node placement used the scheduler launch backend after authorized;
unstarted four-core t195168 was cancelled. Only 376,618 bytes returned.
Day 2231004 split bilateral minus raw bilateral launch CV2 +0.001816 splits
into eligibility +0.000014, readiness +0.001612 and normalization +0.000189.
Split funded: +0.005022 = -0.000016 +0.004837 +0.000200, respectively.
For both actors, trip 16 is the first actual-launch/readiness difference:
waiting falls 5/4 s to zero while eligibility is identical. First eligibility
differences occur later. This localizes clocks, not a causal mediation share.
Keep main frozen. Next passively replay consumed success/failing paths, require
unchanged actions/endpoints, and log same-vehicle prior-trip terminal return,
availability, eligibility and actual release before designing a timing repair.
