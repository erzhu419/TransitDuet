# Stage197: weighting versus action credit

Matched 2x2: absolute fleet anchor versus direct three-head action differences;
uniform weights versus mean8 weight8/singleton weight1. Same600 inputs,
240 mean8 targets,40 test prefix groups,47 folds,1635 parameters,100 updates,
seed178 and AdamW. Weighting retains train-only scales; direct credit instead
uses fleet pair RMS and has no absolute fleet-risk forecast.

Completed: `t147315-t147361` and report `t147367`, all done/exit0.
141 new fits,47 cached controls,14100 critic updates,0 actor/native updates.
Tests:37 passed. Fit wall-time sum180.85s; not an end-to-end runtime benchmark.
Only the708052-byte report was manually pulled; raw data/models remain remote.

Wait-feasible point estimates, cost delta / wait delta in minutes:

| Model | Scene | Regime | Lower Root |
| --- | --- | --- | --- |
| Anchored/uniform | -0.001373 / -0.001371 | +0.001356 / +0.000588 | -0.002459 / -0.004965 |
| Anchored/weighted | -0.000102 / -0.001692 | -0.001241 / -0.003613 | -0.000099 / -0.004118 |
| Advantage/uniform | +0.000752 / -0.000903 | +0.000652 / +0.004709 | -0.000895 / -0.006127 |
| Advantage/weighted | -0.001111 / -0.007728 | -0.000753 / -0.001965 | +0.001913 / -0.004467 |

Anchored/weighted passes both roots within regime; advantage/weighted passes
both roots within scene. No model passes all three splits on both roots.
Advantage/weighted scene cost pair MSE is merely near the zero-difference
reference (0.00045111 vs0.00045212, noise-corrected), not a strong critic win.

Next: check mean8 training-subset errors and cross-lower distribution changes
before changing representation or training budget. No extra seed collection.
Limits: replicate counts approximate reliability; this is a development
one-action/original-greedy-suffix screen, without CI or full HRL validation.
Reserved scenes and native deployment remain closed.
