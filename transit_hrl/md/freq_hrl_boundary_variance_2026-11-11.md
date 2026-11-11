# Cost-Accounted Boundary Variance Diagnostic

Keep the four C0 controllers/rules and the closed E8 decision unchanged. The
four boundary prechecks passed. Freeze eight NEW development prefixes/root,
three outcome-independent reachable checks/prefix, four window draws/check and
four paired suffix redraws/window. Prefixes exclude E8, other historical roles,
registered reservations and effective controller-training seeds.

Collect W, tail and G jointly. Report per-state within-window suffix variance,
corrected between-window variance, W/tail covariance and nested-mean variance.
Retain negative corrected estimates. Separately compare first/last two windows
and first/last two suffixes; include joint-zero counts in sign agreement. Keep
per-prefix summaries and roots separate. Across-fixed-state variance minus mean
sampling variance is descriptive heterogeneity, not proof of predictable signal.

Full cost/prefix is 103 complete 1,200-step episodes: one factual run plus three
checks each with two original arms and 32 nested arms. Total 3,955,200 steps;
eight unpinned 1CPU/2GiB jobs, four prefixes/494,400 steps each, on node001-node006.
No fitting, cut selection, controller update or policy-performance evaluation.
Raw boundaries/draws stay on the server; retrieve compact moments and costs only.

For J windows and K suffixes, the nested mean has variance B/J + S/(J*K).
The equal-pair-count independent-window projection is (B+S)/(J*K). It is an
analytical simulator-step comparison, not an additional sampled control or a
measured total-CPU comparison. With full replay charged, nesting itself is not
an efficiency claim: for nonnegative B it cannot beat independent draws for
estimating the unconditional mean at equal pair count.

Inspect both noise levels and their stability across development prefixes before
choosing a boundary-value training target. No critic fit, extra validation cohort,
budget extension or original confirmation follows automatically.

## Limitations

These are conditional draws given the simulator's realized latent driver state,
not all uncertainty from observable history. Four windows give noisy component
estimates. Suffix siblings, overlapping checks and two roots/task are not new
independent fitted policies. Corrected heterogeneity/sign agreement alone cannot
establish learnability, causal-state sufficiency or a performance gain.
