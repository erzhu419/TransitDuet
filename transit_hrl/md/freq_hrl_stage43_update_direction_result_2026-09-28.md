# Stage-43 First-Update Objective Direction Result

Implementation `509b78e8ab`; pre-outcome full freeze `a2beb8886d`.
Focused scheduler tests `t103726`:17 pass (117.534s). Native preflight
`t103728`/`t103733`:7200 steps/24 trace audits, zero optimizer steps.
Full `t103741`-`t103748`: all eight original roots complete, no duplicates;
32 source first-learning batches and before/after policy drifts match exactly.
All1536 native trace audits pass. Aggregation `t103760` exits zero and passes
the independent sixteen-endpoint bootstrap. Controls/seeds/budgets retain the
pre-outcome freeze; dispatch changes only the documentation source revision.

Sixteen-endpoint adjusted intervals, mean [CI]:

| Arm | Training Clipped Gain | Task Direction | Deterministic Return Change | Sampled Return Change |
|---|---:|---:|---:|---:|
| Intrinsic-sham | 0.003828 [-0.003777, 0.008132] | 40.8542 [6.4795, 89.9444] | -0.8336 [-3.0745, 0.9978] | -0.0533 [-0.9989, 0.7910] |
| Intrinsic-clock | 0.004692 [0.00001022, 0.008539] | 44.6876 [-1.3299, 99.3603] | -0.6492 [-2.2239, 0.9884] | -0.2428 [-1.3797, 1.0674] |
| Task-sham | 0.001010 [-0.003875, 0.005589] | 12.6516 [-37.1388, 61.0127] | -0.1554 [-1.6059, 1.2904] | -0.7617 [-1.9622, 0.4642] |
| Task-clock | 0.003775 [-0.000650, 0.007430] | 8.6168 [-66.1779, 73.8023] | -0.3110 [-3.1466, 0.9482] | 0.2842 [-0.8887, 1.6537] |

Two effects are positive, fourteen inconclusive; none negative. Neither the
registered local-direction conflict nor finite-update utility conflict is
supported. All eight actual return intervals cross zero.

Descriptive optimizer evidence:8/32 source updates decrease both their original
full-batch clipped surrogate and entropy-inclusive actor objective (arm counts
1/1/4/2). Mean clip fractions17.30%-20.29%; per-step Gaussian KL0.01390-0.01641,
or16.68-19.69 summed per1200-step episode on the reconstructed before-policy data.
These observations motivate finite-step/optimizer-fidelity diagnosis.

New diagnostic cost:1843200 steps/45301 upper/64737 gate calls; no optimizer or
extra verification environment steps. Reconstruction307200, evaluation1536000.
Only [the76KB compact summary](../results/pointmaze_update_direction_stage43_v1_full_20260928_r1/qualification_summary.json)
is local; source weights and all raw traces remain remote.

## Limitations And Next

Eight reused development roots are not independent algorithm confirmation.
The task score is a potentially high-variance full-episode local estimate, not a
finite-update effect. It differs from the option-cut normalized GAE objective.
Neither a repair/no-harm claim nor the cause of cumulative loss is established.
Next freeze a signed microstep versus full-displacement native probe, preserving
direction/reward/critic inputs, to separate local-direction calibration from
finite-step response. No step fraction is selected using held-out reward.
No new training, seed extension or post-outcome retuning launched.
