# Stage104: Paired Training-Roster Update Order

Stage103 did not support joint superiority at both periods. Its historical training rosters differed. Stage104 removes that difference with a new controlled training cohort; Stage103 remains archived separately.

- Four learners start from the same original Stage96 U0/L0 and frozen Stage97 decoder: joint independent/conditioned and staged independent/common. No trained Stage98-103 donor reuse, teacher selection or recalibration.
- Each actor consumes the same registered scenario and action-noise roster for each of eight updates across all four learners. Upper and lower have separate pools; A/B are disjoint. Only conditioned/common lower credit shares upper innovations within each pair; lower innovations remain independent. Preflight uses two updates, not performance admission.
- Joint collects upper and lower credit before either mean changes, then updates both. Staged completes all lower updates with original upper frozen, then all upper updates with the learned lower frozen. Per-step and phase-boundary checks retain fixed stds, values, source Adam and forecaster.
- Per method/period:512 gradient paths per actor,1024 native training episodes, eight updates per actor, nominal upper KL0.004 and lower KL0.00792/0.00796 at50/100. Common lower replay6144/3072 extra upper forwards matches across orders. Joint has8 update operations; staged16. Realized KL is reported rather than assumed equal.
- Evaluate six final complete policies with fresh independent paired noise,32 paths/policy/period, horizon1200. Keep all eight roots, both periods, original base and zero. All18 comparisons use equal-root bootstrap65,536 / Bonferroni18, seed104/104104. Both joint-conditioned-minus-staged-common primary lower CI bounds must exceed zero; conditioning and teacher contrasts cannot substitute.
- Total full cost:68,608 native episodes /82,329,600 steps,1024 actor-mean updates,147,456 extra upper replay forwards and64 final server-only checkpoints. Preflight:304 episodes /91,200 steps,32 mean updates,144 replay forwards, zero checkpoints.
- Scheduler placement is dynamic node001-006, no node binding. Preflight3 CPU /3 GB, full9 CPU /8 GB, qualifier1 CPU /2 GB. Retrieve completion markers and compact JSON only. Full follows mechanical qualification and the unchanged registration, not preflight reward.

## Scope
This isolates the specified simultaneous versus lower-then-upper MC training recipes under matched random rosters and nominal budgets. It remains same-task, teacher-initialized mean learning; it does not establish full actor-critic, equal realized trajectory KL or frequency superiority. Stage67 critic HOLD and all earlier negative results remain unchanged.
