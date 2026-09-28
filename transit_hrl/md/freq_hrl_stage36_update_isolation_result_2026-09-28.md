# Stage-36 Update Isolation Result

Tasks `t102166`-`t102205`: all 40 native cells and 2560 paired final/selected
evaluation episodes complete. Forty raw-trajectory audits, eighty checkpoint
replays and an independent root-count bootstrap pass. Source `4b7af62fd9`;
full registration `5c573fc783`. Final128 weights are the primary cohort.

| Final-weight contrast | Mean | Seven-endpoint adjusted CI |
|---|---:|---:|
| Gate-only return vs frozen | -29.8709 | [-61.2733, -3.9655] |
| Controller-only return vs frozen | -19.2165 | [-29.4496, -8.5208] |
| Joint return vs frozen | -44.2377 | [-76.3933, -10.2895] |
| Return interaction | 4.8497 | [-22.6260, 21.2426] |
| Joint return vs fixed50 | -37.7494 | [-54.4626, -20.2432] |
| Joint ISE reduction vs fixed50 | -0.74605 | [-0.95952, -0.50541] |
| Joint call savings vs fixed50 | -0.4102 | [-3.4720, 1.9570] |

**Both gate-only and upper/lower-only updates reduce return versus frozen.**
Their interaction is inconclusive; the failure cannot be assigned only to
the gate or to joint interference. Final joint return856.75 versus fixed50
894.50; ISE2.39357 versus1.64752 (+45.28%). Planning-call savings are not
supported. Selected joint return889.28 versus selected fixed50 914.71 is
secondary and does not replace the registered final-weight diagnosis.

Method: 54192000 steps, 1149933 upper calls, 1525849 gate calls, no previews.
Verification: 96000 steps, 2171 upper calls, 2649 gate calls. Twenty-seven
tests pass. Raw arrays/checkpoints stay remote; only compact JSON is pulled.

## Limitations And Next

Conditional development on eight reused controller roots, not independent
confirmation or algorithm superiority. Stage-35's source builder already
seeded NumPy; Stage-36 makes shuffle seeds explicit per iteration to control
update-topology RNG advancement, not to correct an unseeded old result.
Next isolate upper versus lower adaptation and diagnose stochastic gate
training versus deterministic deployment before changing the credit objective.
Earlier negative results remain unchanged.
