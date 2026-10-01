# Stage68: Actor-Credit Reliability Diagnosis

Stage67 improves value fit in32/32 cases but fails the frozen credit gate. The same-batch MC-minus-value gradient changes its fitted baseline when the critic changes, so its cosine is not an independent true-gradient oracle. Diagnose sampling reliability and reference movement before selecting any credit intervention.

Freeze the same eight roots, periods50/100, arms and eight first-training episodes; restore the exact Stage64 ordinary-MC and Stage67 factored critics. Reproduce their probe values and full-batch score-gradient metrics. Derive per-episode raw GAE, raw MC-minus-value, constant-score and entropy gradients; require pre-update ratios to lie inside the PPO clip interval. Center/scale each fold separately, matching actual PPO normalization. Use all35 unordered4/4 partitions (preflight has one1/1 partition), never select a favorable split.

Measure within-GAE and within-MC disjoint-half cosine, cross-half GAE/MC cosine, and both critics' GAE against each of the same two MC references. Report all/mean/log_std separately, including undefined zero-gradient counts. All splits share episodes, so35 is not35 independent observations; no split-based confidence interval or significance claim. Actors, critics and all Adam states stay bitexact. New sampling, optimization, fitting and checkpoint writing are zero. Scheduler:3CPU/3GB per root, dynamic node001-node006; only completion markers and compact JSON pulled.

Decision: read-only diagnosis only. Stage67 HOLD stays unchanged regardless of reliability metrics. No lambda/LR/seed/threshold sweep, no model selection and no new reward/frequency claim. The next intervention must follow the observed reliability structure, not a relabeled Stage67 pass.
