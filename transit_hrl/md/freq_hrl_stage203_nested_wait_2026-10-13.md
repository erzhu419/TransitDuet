# Stage203: training-regime-selected interaction shrinkage

Reuse all three Stage202 controls and frozen cost heads. The primary keeps198
wait features: base102 ridge0.1, last96 interaction ridge in0.1/1/10/100.
Within each outer training set, leave one complete regime out. Fit all training
candidate pairs with mean8 weight8/singleton weight1, using each inner training
population's own scales. Score only inner-held mean8 labels in minute-squared
pair MSE; minimize worst regime error, then equal-regime mean, then prefer the
stronger penalty on exact ties. Refit the selected penalty on outer training.
No outer evaluation labels choose the penalty; zero margin and strict gate stay.
Feature extraction and normal equations are shared across each four-model path.

Run: `native_transit_nested_wait_stage203_development_20261013_r1`.
47 fits plus one report;920 inner and47 final solves,94 reused wait models and
47 frozen cost models,0 optimizer/actor/native updates. Unbound node001-node006,
1CPU/3072MB per task. Raw data, models and inner ledgers remain server-only.
Tests:23 passed, including block normal equations, feature reuse, default-path
reproduction, inner/outer label isolation and exact967-fit accounting.
Next: compare transfer gates and selected penalties against all frozen controls.
Limits: nested development selection is not independent confirmation, cross-root
training diversity, CI-supported improvement or full learned-HRL validation.
