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
