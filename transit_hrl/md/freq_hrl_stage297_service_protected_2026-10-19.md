# Stage297: Service-Protected Policy Calibration

Run: `native_transit_service_protected_stage297_20261019_r1`.

Stage296 improves cost/wait versus raw but retains only 29.5% of baseline
waiting gain. Its extra cost value versus matched thinning has a negative
conditional MC interval, not a negative prefix interval. Freeze the critic,
raw/lower weights and P8/S8 proposals; no critic refits, simulation or action search.

Use only the existing 80 fit and 60 calibration prefixes. Define two global
cutoff families from fit-role eligible score quantiles 1/16 through 16/16,
plus zero and shared full-baseline retention. Scores are frozen fleet risk
or that same risk plus S8 paired service cost. Save the family before reading
calibration T16. Stage296 test states and E32 are excluded from this selection.

Require candidate wait minus half baseline wait <= 0 using a 97.5% prefix
max-t bound across both families, 97.5% Bonferroni MC bounds, and both T8
points. Baseline wait must improve under both CI95s and both halves; candidate
cost must improve pooled and in both halves, with negative mean service cost.
Each family selects the smallest worst-of-prefix/MC cost upper bound, using
the same samples, model and rule. This changes retention, not action ranking.

Admit fresh confirmation only if the joint candidate retains a nonzero proper
subset of baseline actions and its calibration cost point beats baseline.
All-raw and unchanged-baseline choices cannot qualify as a new candidate.
If admitted, reserve 120 genuinely new test prefixes (test replicas 12-23)
before independent E32; do not confirm on Stage296 outcomes.

This is development calibration, not independent policy validation or a safety
certificate. Candidate statistics/choices stay server-side. One unpinned
scheduler job, 1 CPU/3 GiB on node001-node006; pull only the small summary.

Validation: 12 focused plus 35 prior regression tests pass, including shared-
future cancellation, profile duplication, both-half retention and source-role
isolation. The family-save-before-calibration-target ordering is exercised.

Execution: t189910/node006 completed done/exit0; 7,859 scalar bytes pulled.
The joint cutoff 0.002181045736174378 keeps 130/169 changed profile-states,
retains 99.30% of calibration waiting gain and has cost -0.000477664 versus
base -0.000427096. Fleet-only selection is full base. Both T8 cost points
are negative and the joint waiting-budget bounds pass, admitting new-prefix
confirmation. These are selection-data results, not independent evidence.
Stage298 expands the initially reserved 120-prefix cohort to 400 before any
new scene is generated; the selected policy and 50% target remain unchanged.
