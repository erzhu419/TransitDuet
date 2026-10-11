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
No new nested fitting, control/actor updates or native episodes.

Thirty-seven focused tests pass, including regret-gradient exclusion,
unchanged scales, frozen scene/step pairing, primary gate, inference label
isolation, scheduler scope and previous floor/scene-selection behavior.

## Result and Next Step

t143224 completed exit0 on node004: all 28 fits/2,800 updates qualified.
The result JSON was already local; no extra pull was needed. Primary:

| Split | root397 cost | root401 cost | Overall wait, min |
|---|---:|---:|---:|
| Scene | +0.0007371 | -0.0055275 | +0.0129500 |
| Regime | +0.0220015 | -0.0051297 | +0.0184167 |
| Lower root | +0.0151114 | -0.0061543 | +0.0140750 |

Scene overall cost is -0.0023952, improving on the frozen-step regret
control by -0.0036878; regime cost worsens by +0.0066561. No variant
passes the both-root cost/wait gate. Neither early stopping nor removing
regret alone establishes transferable cost/wait improvement.

Representation audit finds one exact alias: root401/900035901/probe23,
candidates3/4 have identical causal96 features but cost deltas0/+0.000719
and wait deltas0/+0.003. Full-tail wait/headway proxies differ and are
omitted by the current feature map. Identical encoded actions cannot be
distinguished by any fitted critic. This one example does not explain
all transfer error. Prioritize faithful full upper-action/tentative-tail
encoding and scene coverage; do not claim a performance fix or collect
third-scene confirmation on the basis of these failed gates.

## Limitations

Stopping is frozen from the regret-trained development model, not optimized
for the new objective. Squared prediction loss alone does not fix sparse
scene coverage or incomplete state representation. This is an objective
ablation, not native learned-policy confirmation; third/fourth action-label
scenes stay unused unless a separately registered transfer gate is passed.
