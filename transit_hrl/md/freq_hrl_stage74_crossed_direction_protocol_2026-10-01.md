# Stage74: cross historical fitting with native execution

Stage73 demonstrated two localized zero-residual gains, but all normal-execution endpoints remained inconclusive. It changed historical direction-fitting data and native execution together. Stage74 separates those factors before joint training resumes.

For each period 50/100, reconstruct all three Stage73 directions from each historical fitting arm, reproducing the saved geometry and reward frames exactly. Retain the same Fisher radius 0.001, entropy treatment and source policy. Evaluate each of the twelve signed cloned actors under both zero-residual and normal execution; each execution has one shared base. Parameters and steps are identical across execution arms, with no radius retuning or direction selection.

Use fresh 74M seeds, both actors sampled, matched environment seeds, policy RNG, stepwise lower noise and upper proposals across all13 variants and both execution arms. Full: eight roots, 32 evaluation seeds per period; preflight: one root, four seeds, H=300. Only numerical JSON and completion markers are written; source networks/Adam remain unchanged.

Report 72 cell contrasts (plus-minus, plus-base, minus-base), 12 execution differences of plus-base (normal minus zero-residual), 12 fitting differences of plus-base (joint history minus zero history), and six differences-in-differences. All 102 endpoints use equal-root means, 65536 bootstrap draws and two-sided Bonferroni-102 intervals. Attribution requires the relevant adjusted contrast, not a difference between significant and nonsignificant cells. Stage67 HOLD and the no-adoption decision remain unchanged.

Full budget: 4096 historical episodes / 4,915,200 reconstructed lower calls; 13,312 native episodes / 15,974,400 native steps; 192 cloned-actor perturbations; 96 Stage73 direction reproductions. Native plan solver costs are recorded separately. Dynamic scheduler placement uses node001-node006, 9CPU/8GiB per full root, separate preflight/full resource histories and a 1CPU/2GiB qualifier. Pull compact JSON/logs/markers only.

## Limitations

Execution comparisons hold parameters fixed, not target-state KL. Fitting effects compare the entire historical-data/critic/direction bundle. This is a finite-radius stochastic diagnosis on teacher-initialized development roots, not full training, OOD validation or frequency-superiority evidence. Eight-root percentile-bootstrap uncertainty is limited.
