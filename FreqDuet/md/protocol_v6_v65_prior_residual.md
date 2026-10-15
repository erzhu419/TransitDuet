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

## Outcome / Next Step

33 server tests passed; eight residual fits and all 64 fresh rollouts completed
(332,800 held-out decisions). Source: `6d2cfa64eb`. Scheduler records:
t165373-t165376 (smoke/fits), t165378-t165382 (data), t165383-t165387 (evaluation/summary).

| Predictor | MAE | RMSE |
| --- | --- | --- |
| 32-day causal prior | 0.009055 | 0.009055 |
| Linear residual, 237 parameters | 0.007619 | 0.010301 |
| Nonlinear residual, 19,393 parameters | 0.007369 | 0.008531 |
| V64 large direct | 0.011279 | 0.013885 |
| V64 large factored | 0.010102 | 0.011204 |

Linear-minus-prior MAE: -0.001436, 95% CI [-0.003099, 0.000215].
MLP-minus-prior MAE: -0.001685 [-0.003360, 0.000221].
Linear-minus-V64 direct/factored MAE: -0.003660 [-0.005522, -0.001806] /
-0.002483 [-0.003895, -0.001065]. MLP-minus-linear RMSE:
-0.001771 [-0.002472, -0.001047]. Both residuals worsen mean MAE versus the
prior on the same four of 16 demand days (averaging actors within each day).
Training MSE: prior 0.0002421, linear 0.0001879, MLP 0.0001388;
V64 direct/factored 0.0002656/0.0002648. Training gains exist, but reliable
superiority over the prior is not established. Both strict gates remain 0/4;
no action audit/control admission, actor/dual updates or main-policy change.

V64 comparisons change centering, prefix features and normalization together;
only linear-versus-MLP isolates capacity. Smaller capacity is not the supported
remedy here. Next estimate residual shrinkage with whole-day cross-fitting on
V64 training days, freeze that calibration, then confirm on new demand days.
These 16 V65 days are consumed; do not fit coefficients or gates on them.
Compact result: `results_freqduet/protocol_v6_v65_prior_residual/summary/status.json`.
