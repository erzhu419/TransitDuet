# Stage292: frozen abstention, new futures

Freeze Stage291 primary choices and matching weights. Compare raw, old247,
zero-event abstention and expected matched thinning; no gate/head search.
Reuse the same 60 training prefixes/360 states, without held-prefix access.

- Main: 64 new futures/state, Stage288 draw IDs 24-87. Native preflight uses
  IDs 88-89 on one state per root/regime, with no outcome-based selection.
- Replay only within-profile action disagreements plus native identity.
  Restore archived prefix/state/RNG and validate shared continuation tapes.
- Primary cost/wait improvement requires both prefix-bootstrap95 and
  conditional-future normal95 upper bounds below zero, and improvements in
  both 32-draw halves. Report cost/fleet information versus thinning separately.
- Keep thinning probabilities frozen, including inside the descriptive prefix
  bootstrap (unlike Stage291's rematched-cohort bootstrap). Average the 16 fixed
  profiles first; they are not training seeds. No fits or actor updates.
- Scheduler only, node001-node006 allowed, unpinned, one CPU/task. Raw data and
  checkpoints stay remote; pull only small plan/preflight/report JSON.

Limitations: this confirms new conditional futures on known prefixes, not new
prefix transfer or full learned HRL. MC intervals use a normal approximation;
report event counts and split-future consistency. No controller promotion.

Run: `native_transit_abstention_confirmation_stage292_20261017_r2`.

Validation: 14 focused/regression tests and two CLI startup checks passed.

## Completed 2026-10-18

All 2,672 collection tasks exited zero: 10,688 state-futures, 41,216 native
episodes; report t175880 exited zero. Pulled 11,517 report bytes only.
Costs below are multiplied by 1e6; negative means improvement.

| Contrast | Cost | Prefix CI95 | Conditional MC CI95 |
| --- | --- | --- | --- |
| Abstain - raw | -167.24 | [-411.14, +60.15] | [-337.07, +2.58] |
| Abstain - matched thinning | -101.70 | [-231.36, +32.30] | [-177.38, -26.03] |
| Abstain - old247 | -151.48 | [-377.09, +65.96] | [-323.79, +20.82] |

Wait versus raw is -0.001839 min: prefix CI [-0.003845, -0.000378], MC CI
[-0.002163, -0.001516]. Both 32-draw halves improve cost/wait. Wait retains
85.9% of Stage291, but joint retention and cross-prefix cost-information criteria
fail. Fleet versus raw is +20.35; the 408 nonzero weighted state-future effects
are not independent model exposures. No controller promotion.

Next: Stage293 compares inference conventions and attributes new-future effects
without tuning, fitting or simulation. Separate cohort-conditional cost information
from prefix uncertainty; alternate bootstrap intervals cannot overturn this test.

Execution repair: r1 freeze t173117 succeeded, but t173119-t173128 failed before
simulation: output-directory launch mirroring removed the remote-only sealed
ledger. Only freeze now stages local registration metadata; later phases stage
code only. R2 repeats the identical protocol/actions/seeds, not a new candidate.
Explicit results-excluding staging and bounded dispatch also bypassed rsync E2BIG.
