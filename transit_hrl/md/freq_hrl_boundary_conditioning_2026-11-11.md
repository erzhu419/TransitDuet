# A-Only Fixed-Capacity Observable Conditioning

Freeze two gates before results: tracking distance, and generic RMS of the 39
training-parent-standardized state features. Each gate uses training-arm median
and standard deviation only: g=.5*(1+tanh((q-median)/std)). No labels, prefix IDs,
held covariates or E data fit gate geometry. Preserve the shared potential:
psi(s)=[sqrt(1-g(s))*phi(s), sqrt(g(s))*phi(s)], predict psi(wait)-psi(now).
This keeps each arm's feature norm and zero/antisymmetric differences.

Both gates have 240 columns and ridge trace DF3. The cached original 120-column
baseline has an exact uniform-gate lift [phi/sqrt(2),phi/sqrt(2)] with unchanged
kernel/predictions; reuse its saved moments, zero baseline refits. Gate location
and spread are X-only statistics; DF matching is conditional on X, not a claim
of identical nonlinear function classes. Gates are fixed, not learned policies.

Use the same 32 Wide training prefixes/root, 31-prefix fits, reciprocal A views,
matched individual-model losses, all prefix reports, old/new strata and window
halves. Two gates x64 models/root x4 roots =512 fits, 196,608 window predictions
and 1,536 frozen-source checks. One 1CPU/2GiB scheduler task; no native steps.
All A banks and cached baseline losses reproduce before new fits. Twenty-one
focused tests passed. Raw banks and new fold models remain server-side.

## Followup And Limits

Primary Reacher1 followup requires tracking-gate held MSE to beat both uniform
and RMS gate in both A views and each window half, while overall matched-future
MSE does not worsen versus uniform and still beats zero in both views. Report
all four roots, not a root-specific winner. This is a development gate, not
confirmation or complete-episode performance. No capacity/threshold search,
E reuse, frozen-model replacement or automatic next experiment is released.
