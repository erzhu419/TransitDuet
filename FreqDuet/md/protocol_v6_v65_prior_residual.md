# V65: causal prior plus low-capacity residual value

Freeze all four V59 post-64 actors, dual and upper/B200 deployment. Fit only on
V64's 32 training demand days; do not reuse its consumed validation days.
Predict remaining cost as training-day mean terminal CV-squared plus a learned
terminal residual, minus exact completed-prefix CV-squared. Keep signed returns.

Compare a linear residual and a two-hidden-layer residual MLP with V59's hidden
width. Both retain the full observable state/action, adding completed-prefix
CV-squared and its product with observed completion fraction. No future labels
enter features. Standardization uses fitting days only, with equal day weights.
Both output layers start at zero, hence the identical causal prior. Use Adam,
original LR, 2,048 steps, batch 256, and V64's paired minibatch seed. No tuning.
Report full-training MSE against the prior and V64 neural fits to distinguish
fitting failure from held-out failure; this is not a convergence claim.

Confirm once on 16 fresh demand days (750001-750016) per actor, with private
sampler seeds `6500000 + case_index * 1000 + day_index`: 64 unique full-day
rollouts. Preserve all nine V64 control channels on these same new trajectories.
Primary contrast: linear minus 32-day prior. Also report linear minus V64
large direct/factored, MLP minus prior, and MLP minus linear. Average four actor
contexts within each demand day before bootstrapping 16 day clusters (10,000
resamples). Intervals are conditional on fixed fitting sets and actors.

Keep the strict gate: each candidate must beat all nine old controls in both
MAE/RMSE on every held-out day, for all four contexts. This can admit a later
action-effect audit, never a control change here. Calibration alone does not
identify action effects. Main policy, actor and dual remain frozen regardless.

Tests plus paired-fit/full-day smoke run through scheduler before the matrix.
Freeze all fitted weights before the formal collection; reuse the smoke day.
Use node001-node006 only. Raw states and models stay on server; pull logs,
completion markers and compact summary JSON only. Record outcome and next step
here without retuning on the 16 confirmation days.
