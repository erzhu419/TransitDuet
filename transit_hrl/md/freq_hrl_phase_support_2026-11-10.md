# W Sampling And Clock Support

Freeze a 2x2 development comparison: bin-start versus uniformly sampled optional
checks actually visited by original history, crossed with plain 39 features versus
39+offset_fraction+remaining_fraction (41). Both use slope DF8, the fixed 0.75
training-path-OOF cut, W target and original-history continuation. No new neural
model, clock interactions, threshold search or controller training is introduced.

Keep the same 32 training prefixes, three paired bins/path and one fitting future
per state: 96 fitting state-future pairs/arm. Reuse previous start A1/B4 labels;
start/plain must reproduce prior wide DF8. Within each of the SAME sampled bins,
draw one reference-visited check at offsets 0/5/10/15/20, excluding forced25 and
incomplete200-step tails. Check selection precedes branch outcomes. Reuse start
labels when the selected check is0. Plain/clock heads share exactly the same
states and futures within each sampling arm.

New holdout: root+9000 through+9007, disjoint from prior exposed roles and resolved
primary training seeds. Save all four rules before accessing it. On each held
path label paired starts and reachable checks with independent4/4 futures; shared
query states reuse their trajectories. Report start/check/later-check errors,
phase histograms, both episode halves and paired factorial effects. Whole-episode
controls retain history, L, threshold, termination and prior wide DF2; start/plain
is the prior wide DF8 control. Deployment query/renewal phase counts are retained.

Four native prechecks verify cache reproduction, original history, a real later
check's factual/conditional replay, and phase-head call budgets. Only training
prefixes are touched. Main sampling saves paid rows per prefix server-side.
One CPU/2GiB per job, dynamic node001-node006 placement, no Slurm; retrieve only
small public outputs. Preserve HOLD, unknown prior lifecycle costs and failed
termination competence. No automatic candidate promotion follows this experiment.

Limits: reference-visited support is not every new policy's on-policy support.
Report actual phase counts rather than assuming all strata were sampled. The
clock augmentation is additive; sampling and clock tests do not settle whether
phase-dependent interactions or continuation-policy updates are needed. Effective
DF and fitting-label counts match, not coefficient counts or total CPU. Two roots
per task provide development evidence; no confirmation roots are accessed.

Native prechecks t214238-t214241 all passed on node004/node005. All roots exposed
all five optional phases; the later probe was at offset5 and the new phase head
retained 24 upper calls/episode. Cost: 88.125 CPU seconds, 91,200 simulator steps,
264 ridge calls; only 14,287 public bytes retrieved. Main task IDs: t214242-t214245.
All four main tasks started on node004 with unpinned placement across the six-node
pool; no automatic promotion or confirmation was released.
