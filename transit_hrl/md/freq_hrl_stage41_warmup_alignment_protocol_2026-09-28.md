# Stage-41 Critic Warmup Execution Alignment

Stage-40 has no supported return gain. Isolate whether critic pretraining
execution affects subsequent lower learning; retain both rewards and frozen.
Same Stage-33 controller, Stage-35 initial gate, inherited critics/fresh Adam,
unchanged PPO, sixteen critic-only plus sixteen actor/critic iterations.

Five arms: frozen, intrinsic-sampled/aligned, task-sampled/aligned. Suffixes
describe warmup only: sampled uses stochastic upper/gate, aligned deterministic.
Every arm uses deterministic upper/gate in the learning phase. Lower is always
stochastic/on-policy during training; upper/gate networks never update.
Within each reward pair, post-warmup actors and frozen values match exactly;
lower critic weights/Adam may differ. First native learning episode state,
action, reward, duration, done and log-prob match; old_value may differ.
All initial/warmup deployment outcomes match frozen before actor updates.

Eight episodes x1200 steps per iteration. Keep0/16/17/20/32 without selection.
Sixteen paired evaluation paths per snapshot, deterministic and lower-only
sampled deployment; upper/gate deterministic in both. One common initial
all-level stochastic probe per cell, separate from training. Probe MSE/KL and
paired critic/value distances are diagnostics, not outcome gates.

Reuse roots310011/310023/310037/310049/310061/310073/310089/310101.
Base10600000 + root-index x10000: train+1..256, probe+2001, eval+3001..3016.
Training Torch seed=environment+root; shuffle SeedSequence(41,root,iteration).
Lower stream(41,root,environment,41019)+primitive-step, coupled across arms;
sampled gate stream(41,root,environment,41029)+gate-step. Evaluation Torch
seed(41,root,environment,41017). Warmup differs in execution, not noise streams.

Eight primary deterministic return endpoints: four final arms vs frozen,
two final warmup-alignment effects and two first-update warmup-alignment effects.
Equal root means after path averaging;65536 paired root bootstrap draws,
seed(41,41041), two-sided percentile Bonferroni8. No exclusion/extension.
Full40 cells:20064000 method steps plus624000 native verification steps.
Preflight root310001/base10590000:2+2 iterations,300 horizon, five cells,
33000 method steps plus16500 verification steps. Freeze full matrix before
preflight outcomes; operational qualification never selects an arm or budget.
Dynamic node001-node006, cpu9/ram12GiB full; raw/weights stay remote.

## Limitations

Reused roots support conditional development, not independent confirmation.
Upper/gate warmup execution changes together. Task reward also removes intrinsic
action penalty. Initial stochastic probe is not aligned-policy value truth.
No improvement or noninferiority follows from native execution qualification.
