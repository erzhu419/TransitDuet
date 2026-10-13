# Stage220: learned decisions without an oracle inference menu

- Original240 development states only. Four matched views, two complementary complete-prefix folds each:60 train/60 held states, balanced regimes and mixed-root counts.
- Both future partitions and both directions: four training draws only; evaluate the other four. Keep every state, including future-half disagreements. Train-input normalization only.
- Cost: exact paired-ridge control versus regularized linear safe-best-action cross-entropy over all six inference actions. Scores are decision logits, not calibrated cost forecasts.
- Wait: shared train-only GLM; linear, nonlinear and state-conditioned residuals. Each new branch uses600 AdamW steps with the same learning rate, decay and seed. Compare nonlinear versus conditioned to isolate conditioning from capacity.
- Evaluate same/other futures for both training and held prefixes, actual selected cost/wait, physical wait-mass loss and disputed-state outcomes. No oracle menu at inference; retain an unfiltered decision control.

Scheduler: eight1-CPU/3072-MB fit tasks, unbound node001-node006, then one report task.32 cells,19,200 decision and57,600 wait-residual updates; no new native episodes or RL actor updates. Models/predictions stay remote.

Next decision follows held-other-future cost and wait jointly, not training loss. The data remain development counterfactuals under the original greedy suffix; future sensitivity repeats are not independent seed CI or full HRL validation.
