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

Execution repair: r1 freeze t173117 succeeded, but t173119-t173128 failed before
simulation: output-directory launch mirroring removed the remote-only sealed
ledger. Only freeze now stages local registration metadata; later phases stage
code only. R2 repeats the identical protocol/actions/seeds, not a new candidate.
