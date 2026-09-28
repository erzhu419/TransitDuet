# Stage-38 Lower Reward And Credit Boundary

Stage-37 lower-only updates harmed task return; gate sampling did not rescue
learning. Test lower credit before another joint-training change.

Five arms: frozen, intrinsic-option, intrinsic-episode, task-option,
task-episode. Four learned arms form reward-source x credit-boundary 2x2.
Intrinsic is the existing subgoal progress minus action-cost reward. Task is
the unscaled native dense reward, with no added call cost or action penalty.
Option credit terminates at the step before each executed replan; episode
credit terminates only at the native horizon. Actual actions, upper reward,
gate reward and cadence are otherwise unchanged. Only lower actor/critic
update; upper and gate actor/critic remain exactly frozen.

Same Stage-33 controllers/Stage-35 initial gates and inherited lower critics;
no critic reset, reward rescaling, learning-rate or network change. All levels
still sample on-policy in training; deterministic deployment. Final128 is
primary, selected weights secondary. 128 iterations x8 episodes x1200 steps;
selection8 paths at0/32/64/96/128, evaluate32 untouched paths per cohort.
Eight reused roots310011/310023/310037/310049/310061/310073/310089/310101.
New base9400000 + root-index x10000: train+1..1024, selection+2001..2008,
evaluation+3001..3032. Shuffle SeedSequence(38,root,iteration).

Seven primary task-return contrasts: four learned arms versus frozen,
reward-source main effect, credit-boundary main effect and interaction.
Equal-weight root means; 65536 paired bootstrap draws, seed(38,38039),
two-sided Bonferroni percentile intervals over all seven endpoints.
This is a component diagnosis, not an algorithm-superiority success gate.

Forty full cells: 54192000 method steps, 144000 verification steps separately.
Record actual lower batch reward/cuts/value/GAE statistics. Verify final and
selected native policies plus each cell's initial stochastic training credit.
Preflight root310001, base9390000: five cells, 19500 method steps plus4500
verification steps. Preflight checks execution, never selects a treatment.
Scheduler dynamic node001-node006; full cpu9/ram12GiB, pre cpu2/ram4GiB.
Source-only staging; raw arrays/checkpoints remain on the server.

## Limitations

Conditional development on reused roots, not independent confirmation.
Task versus intrinsic changes the entire lower reward, including its action
penalty; inherited critic scale can affect adaptation. No post-outcome root,
threshold, reward-scale or budget changes; retain earlier negative results.
