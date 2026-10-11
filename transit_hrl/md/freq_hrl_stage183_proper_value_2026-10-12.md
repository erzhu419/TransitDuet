# Stage183: Same-Step Predictive Value Objective

Stage182 chooses 100 steps in all 28 fits and reduces optimism, but still
fails cost/wait transfer. Test the gradient pressure of expected regret,
which sharpens decision preferences rather than solely fitting cost values.

Freeze all Stage182 selected steps and outer training/test groups. Reuse
its predictions as control. Keep data, causal96 features, 96-16-3 floor
network, seed178, optimizer, training-only scales and both selectors.
Change only the loss: absolute fleet MSE + three-head paired MSE; expected
regret stays diagnostic and contributes no gradient. Primary is fixed to
proper_mse:wait_feasible. Same both-root/three-split cost-and-wait gate.

One unbound scheduler CPU task, node001-006 allowed: 28 fits/2,800 updates.
No new nested fitting, control/actor updates or native episodes. Results pending.

Thirty-seven focused tests pass, including regret-gradient exclusion,
unchanged scales, frozen scene/step pairing, primary gate, inference label
isolation, scheduler scope and previous floor/scene-selection behavior.

## Limitations

Stopping is frozen from the regret-trained development model, not optimized
for the new objective. Squared prediction loss alone does not fix sparse
scene coverage or incomplete state representation. This is an objective
ablation, not native learned-policy confirmation; third/fourth action-label
scenes stay unused unless a separately registered transfer gate is passed.
