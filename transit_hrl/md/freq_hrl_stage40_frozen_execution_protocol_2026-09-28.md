# Stage-40 Frozen-Level Training Execution

Stage-39 critic calibration improved fixed-probe fit without supported return
gain. Isolate upper/gate training execution from lower learning, retaining
the same Stage-33 controller, initial gate, PPO, rewards and16+16 iterations.

Five arms: frozen, intrinsic-sampled/aligned, task-sampled/aligned. All learned
arms first receive identical sixteen-iteration critic-only warmup with all
levels sampled. During sixteen lower actor/critic learning iterations,
aligned arms execute frozen upper/gate deterministically; sampled controls
keep sampling them. Lower remains stochastic/on-policy in every training arm.
Within each reward pair, warmup network weights and critic Adam state must
match exactly. Upper/gate actors/critics never update; no outcome-selected arm.

Eight episodes x1200 steps per iteration. Keep0/16/17/20/32 without selection.
Sixteen fresh paired evaluation paths per snapshot, deterministic and lower-only
sampled deployment; upper/gate deterministic in both. One separate common
initial stochastic probe per cell, never trained on. MC MSE/KL and sampled
curves are diagnostic, not performance gates.

Reuse roots310011/310023/310037/310049/310061/310073/310089/310101.
Base10500000 + root-index x10000: train+1..256, probe+2001, eval+3001..3016.
Training Torch seed=environment+root; shuffle SeedSequence(40,root,iteration).
Lower stream SeedSequence(40,root,environment,40019)+primitive-step, coupled
across treatments; sampled gate stream(40,root,environment,40029)+gate-step.
Evaluation Torch seed(40,root,environment,40017). Re-seeding holds lower noise
fixed when deterministic upper/gate stop consuming random draws.

Eight primary deterministic return endpoints: four final arms vs frozen,
two final alignment effects and two first-update alignment effects. Equal
root means after path averaging;65536 paired root draws, seed(40,40040),
two-sided percentile Bonferroni8. No seed exclusions or sequential extension.
Full40 cells:20064000 method steps plus624000 native verification steps.
Preflight root310001/base10490000:2+2 iterations,300 horizon, five cells,
33000 method steps plus16500 verification steps. Dynamic node001-node006,
cpu9/ram12GiB full; raw trajectories/checkpoints stay remote.

## Limitations

Conditional development on reused roots, not independent confirmation.
Both frozen levels change execution together, so this does not isolate upper
from gate. Shared warmup does not test deployment-aligned critic pretraining.
Task reward also removes the intrinsic action penalty. No superiority or
noninferiority follows from a successful execution audit or a noisy probe fit.
