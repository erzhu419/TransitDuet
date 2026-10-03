# Stage103: Matched-Budget Joint Versus Staged

Stage102 supports joint conditioning over its independent control, with a lower-driven gain. Compare joint and staged recipes at matched method-path training budgets using their registered final policies, not additional training or extra-compute Stage101 UJ refinement.

- Joint: both final Stage102 means trained from original Stage96 U0/L0. Staged: final Stage99 U0-trained lower and final Stage100 U0-start upper. Reuse all roots, both periods and both independent/common arms; no donor selection or recalibration.
- Per controller/period, each actor used512 gradient paths and eight mean updates, totalling1024 native training episodes. Cumulative nominal upper KL0.004 and lower KL0.00792/0.00796 at50/100 match, as does common-lower replay6144/3072 extra upper forwards. Joint has8 simultaneous updates; staged has16 single-actor updates.
- Load and verify all six final donors per period; staged checkpoint lower must equal its registered installed Stage99 lower. Both stds, values and source parameters remain fixed. No training models, gradients, new checkpoints, upper replay or raw trajectories in Stage103.
- Evaluate six complete policies: both joint arms, both staged arms, original base and zero. Fresh paired scenario/noise roles; full32 evaluation paths/policy/period, preflight4. All18 contrasts use equal-root bootstrap65,536 / Bonferroni18. Both joint-conditioned-minus-staged-common corrected CI lower bounds must be positive for joint-recipe superiority; retain independent controls and all negatives.
- Native evaluation budget: preflight48 episodes /14,400 steps and12 donor loads; full eight-root cohort3072 episodes /3,686,400 steps and96 donor loads. Dynamic node001-006: preflight3 CPU /3 GB, full9 CPU /8 GB, qualifier1 CPU /2 GB. Retrieve completion/compact JSON only. Full follows mechanical preflight, not its reward.

## Scope
Budgets match method-specific training paths, not the historical campaigns or shared teacher/decoder preparation. Training rosters differ; this fresh paired evaluation compares recipes, not an isolated causal effect of update order, full actor-critic, frequency superiority or unseen-task generalization. Stage102's negative upper-swap results and the earlier HOLD boundaries remain recorded.
