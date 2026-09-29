# Stage50: Fisher-aware full-task score direction

Stage49 did not support actor-Adam reset. Stage50 tests whether update geometry,
not another gate or credit coefficient, repairs native lower-policy learning.

- Source: fixed Stage42 task-clock warmup checkpoint16 (preflight checkpoint2).
- Arms: bounded GAE Adam, bounded full-task MC Adam, MC Euclidean, MC Fisher.
- Direct arms share one full-batch gradient of normalized time-LOO undiscounted
  task return times log probability, including original entropy coefficient.
- Fisher: double-precision Hessian of mean conditional Gaussian KL at the old
  actor; solve `(F + 0.1 I)d = g` with SciPy CG, at most10 iterations, rtol1e-6.
- Both direct directions start at quadratic mean-episode KL0.1 and try successive
  halves, at most13 proposals. Accept first exact maximum empirical episode
  KL<=0.1. No evaluation or surrogate-improvement acceptance test.
- Critic: original GAE targets and one original critic update; actor Adam remains
  untouched in direct arms. Upper and gate networks remain frozen.
- Full run: eight roots,16 rounds,8 native episodes/round,H1200,16 evaluation
  paths/mode at rounds1/16. Total7,680,000 native steps,6,400 trace audits.
- Fresh Stage50 train/evaluation seeds. Eight frozen endpoints, equal-root paired
  bootstrap65,536 draws, two-sided Bonferroni8 intervals; no checkpoint selection.
- Repair requires positive final Fisher-minus-Adam and Fisher-minus-frozen CIs;
  curvature attribution additionally requires positive Fisher-minus-Euclidean CI.
- Account for all executed Adam steps, full-batch score backwards, Fisher-vector
  products, CG iterations, proposals and KL checks separately from retained work.
- Scheduler only, dynamic node001-node006. Only compact summaries pulled locally.

## Limits
These are reused development roots, not independent confirmation. Equal KL caps
are not equal realized KL or compute. Empirical history KL is not a population
trajectory bound. Truncated CG is an approximate natural direction, not a
monotonic native-return guarantee. Full-batch ascent is not multi-epoch PPO.

Method basis: [TRPO](https://proceedings.mlr.press/v37/schulman15.html);
solver: [SciPy CG](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html).
