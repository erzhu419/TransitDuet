# Stage53: Recorded Forecast/Control Diagnosis

Use the complete Stage52 r2 dataset: all eight roots, six arms, two periods, two modes and 16 paired paths per root. No new native paths, optimizer updates or checkpoint access; 3072 raw reads and 128 exogenous-driver regenerations stay on the server.

At primitive step t, use achieved position after the action and target/reference before it, matching native reward. Decompose squared tracking error exactly into controller error, forecast error and their signed cross term. Check raw reward/error sums against saved rows, regenerated measurements against every recorded path, and that disjoint partition contributions sum to full-episode totals.

Eight fixed diagnostic endpoints: curve-minus-hold forecast/controller/cross/total-vector integrals at periods50/100, deterministic mode, equal-root paired bootstrap with 65536 draws and Bonferroni8 intervals. These are newly specified post-outcome diagnostic intervals, not confirmation of Stage52 utility.

Descriptive partitions: stable/regime-only/geometry-only/mixed exposure; clean/history-only/future-only/both timing; and normalized age thirds. A latent regime change at c changes the first target increment at c+1. Geometry markers are dominant motion-axis changes or direction reversals not explained by that regime marker. An option's exposure combines valid 64-target fitting-history increments with its future recorded target increments. All labels are retrospective and never enter a controller.

Report both per-episode signed contributions and conditional error differences, retaining empty strata and all arms/modes/periods. No subgroup CIs, threshold search, root extension or Stage52 window/gain/period retuning. Cross terms preclude naive nonnegative responsibility percentages; accounting is not causal mediation.

Next choice depends on whether recorded forecast deterioration or controller/reference interaction dominates. A new plan class still requires its own learned native utility experiment against held-reference and fixed-clock controls; this offline analysis cannot authorize deployment.
