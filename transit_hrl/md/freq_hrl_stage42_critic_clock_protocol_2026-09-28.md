# Stage-42 Critic-Only Control Clock Information

Stage-41 warmup alignment has no supported return gain. Test whether explicit
control clocks help the lower reward critic while leaving actor information,
reward credit, PPO, controller source and training budget unchanged.

Five arms: frozen, intrinsic-sham/clock, task-sham/clock. Every lower critic
has two extra inputs; sham/frozen receive zeros, clock receives current option
age/100 and remaining episode fraction. Age is measured after any renewal and
before the current action; no future observation enters either channel.
Inherited MLP input weights are preserved and the two extra columns start at
zero. Construct/copy the original Stage-35 gate before expanding the critic,
so initialization RNG consumption cannot change the gate. Adam starts fresh.
Actor inputs and all upper/gate networks remain unchanged in every arm.

Same Stage-33 controller, Stage-35 initial gate and16 critic-only plus16 lower
actor/critic iterations. Upper/gate deterministic throughout both phases;
lower stochastic with coupled per-step noise. Intrinsic/task option-terminal
credit and its GAE masks are unchanged. Warmup actor data/frozen networks match
within each reward pair; critic context, weights, values and Adam may differ.
First native learning episode state/action/reward/duration/done/logp matches.
Initial/warmup deployed actor outcomes equal frozen. Persist only two context
channels per raw step; verify clocks against actual renewal times and verify
their presence in training batches. Raw trajectories/weights stay remote.

Eight episodes x1200 steps per iteration; retain0/16/17/20/32 without selection.
Sixteen paired evaluation paths per snapshot in deterministic and lower-only
sampled modes. One separate common initial all-level stochastic probe per cell.
Critic error, weight norms and policy drift are diagnostics, not outcome gates.
Reuse roots310011/310023/310037/310049/310061/310073/310089/310101.
Base10700000 + root-index x10000: train+1..256, probe+2001, eval+3001..3016.
Training Torch seed=environment+root; shuffle SeedSequence(42,root,iteration).
Lower stream(42,root,environment,42019)+primitive-step; probe gate stream
(42,root,environment,42029)+gate-step; evaluation Torch(42,root,environment,42017).

Eight primary deterministic return contrasts: four final arms vs frozen,
two final clock-minus-sham effects and two first-update clock-minus-sham effects.
Equal root means after path averaging;65536 paired root draws, seed(42,42042),
two-sided percentile Bonferroni8. No exclusion/extension or arm selection.
Full40 cells:20064000 method steps plus624000 native verification steps.
Preflight root310001/base10690000:2+2 iterations,300 horizon, five cells,
33000 method steps plus16500 verification steps. Freeze full before preflight
outcomes. Dynamic node001-node006, cpu9/ram12GiB full; no node binding.

## Limitations

Reused roots support conditional development, not independent confirmation.
The two clocks change together. Task reward also removes intrinsic action cost;
the initial stochastic probe is not aligned-policy value truth. Clock absence
is a hypothesis, not an established cause. Operational qualification alone
establishes neither return improvement nor noninferiority.
