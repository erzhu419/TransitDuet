# Stage78: actual nonlinear response constraint

Freeze the Stage77 BC-derived target: command-change RMS <= sqrt(BC command
MSE), averaged over every historical BC state. Reproduce Stage77's one-shot
alpha and response, then halve alpha until the actual nonlinear response first
satisfies this target. Keep reference and velocity on the same blended curve.
No reward-selected scales, monotonicity assumption, or model updates.

Four curves: zero, original, Stage77 ratio, bounded. Preflight uses the first
full trained root and all its 1,200-step BC labels, with four fresh native seeds
at H300 (32 episodes). Full: eight roots, 32 fresh seeds/root, periods 50/100,
H1200; 2,048 episodes / 2,457,600 steps. Evaluate every pairwise reward/tracking
contrast: 24 endpoints, equal-root bootstrap, 65,536 draws, Bonferroni24.
Record every solver evaluation and derive actual offline work from its trace.

Scheduler dispatches dynamically across node001-006; 3 CPU/3 GiB preflight,
9 CPU/8 GiB full. Pull compact JSON/markers only. Existing execution equivalence
is retained, not repeated. Stage67 HOLD and Stage77 negative evidence remain.

## Limitations / Next Step

This constrains aggregate historical conditional mean-command response, not
individual states, native reward, or learned joint HRL. Compare bounded against
both ratio and zero before deciding whether input compatibility is repaired.
Preserve residual harm; no post-hoc reward threshold or scale retuning.
