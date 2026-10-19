# V136: read-only V135 propagation audit

Read all192 completed V135 pairs on the server, retaining the160 held-actor
primary pairs and32 development-actor/new-day secondary pairs separately.
No rollout, training, predictor/controller refit, gate change or policy switch.

Reconcile actual arrival CV2, stop holding, passenger totals, return times and
inclusive launch delay with saved endpoints. Separate candidate same-state
handoff request overrides from paired realized holding on those keys and all
other stops. The remainder is path accounting, not per-action causal value.
Decompose actual CV2 using release, holding, observed nonholding stop and
remaining elapsed, retaining cross-covariances, normalization and neighbour
changes. Allocate matched vehicle-return changes into launch-delay components;
keep remapped vehicles separate. Report actor/day heterogeneity and pooled
station/time localization, without fitting to these consumed validation days.

Four targeted accounting tests and the audit run via scheduler in the existing
independent CPU environment. Full per-stop/per-trip details stay server-side;
fetch only the small final summary. Sparse launch traces lack complete follower
ready labels and passenger-level exposures; do not infer exact ETA bias or
per-action passenger attribution from arrival clocks alone.
