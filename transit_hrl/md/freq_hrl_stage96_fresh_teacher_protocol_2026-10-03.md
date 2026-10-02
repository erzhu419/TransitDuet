# Stage96: Fresh Native Teacher Cohort

Stage94/95 supported the four registered reward endpoints conditional on the original eight teachers. Stage96 rebuilds the upstream teacher artifacts rather than relabeling those teachers or changing only MC samples.

- Full roots: 410011, 410023, 410037, 410049, 410061, 410073, 410089, 410101. Separate preflight root: 410001. Every full root stays in the cohort regardless of native return or BC fit.
- Native initializer: Stage33 history-MLP architecture and shared PPO update, balanced-jitter 50-step schedule, 384 iterations x 8 fresh native paths, H=1200. Use the final iteration, not validation-selected checkpoints. Eight persistent workers collect each training batch in parallel.
- Rebuild the Stage42 task-clock critic warmup (16 x 8 paths; no actor updates), Stage54 ridge forecaster (32 disjoint driver paths), and Stage55 velocity-feedback teacher labels (8 paths per period). Fit the lower mean for 64 fixed BC epochs; keep upper, critics and Gaussian std frozen.
- Total per root: 3,248 native episodes / 3,897,600 steps, including initial/final diagnostics. Full cohort: 25,984 episodes / 31,180,800 steps. BC: 1,280 optimizer steps per root. Four final checkpoints and 16 teacher-label archives per root remain server-only.
- Preflight: root410001, H=300, 12 episodes / 3,600 steps. Its small-budget artifacts never replace full-cohort teachers. Full launch follows native preflight qualification.
- This is prerequisite construction, not a performance test. Completion checks fixed-final training, frozen components, complete cohort and measured budgets, never a reward threshold. No historical teacher, forecaster, decoder or lower checkpoint is loaded.

Next: calibrate the decoder from these new labels without reward search; rebuild joint/lower donors; apply the unchanged staged lower-then-independent-noise-upper rule. Preserve zero-plan, U0, fixed-UJ and matched-independent controls and the four corrected primary endpoints at both periods. Do not pool this cohort with Stage94/95 or reopen frequency-superiority / Stage67 critic-credit claims.
