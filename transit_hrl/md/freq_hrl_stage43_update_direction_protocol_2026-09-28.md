# Stage-43 First-Update Objective Direction Diagnosis

Stage-42 has no supported clock repair. Diagnose its actual first lower update
without training, changing an objective, or selecting an arm/checkpoint.
All four intrinsic/task x sham/clock arms remain. Load their fixed16/17
checkpoints (preflight2/3). Exact pre-update actor/frozen-level equality permits
one shared before-update deployment reference per root.

Reconstruct all eight first-learning episodes using original Stage-42 seeds,
credit, sampling, critic inputs and values. Match recorded rewards, GAE stats,
sampling and before/after policy drift. Measure full-batch clipped PPO gain with
the original normalized GAE; entropy-inclusive objective gain is descriptive.

Fresh paths assess the undiscounted full-episode task objective. On before-update
lower-sampled trajectories, estimate its score gradient and dot it with the
actual parameter displacement. Use full task return-to-go, not option cuts or
intrinsic credit. The time-indexed baseline is the leave-one-episode-out mean
return-to-go, independent of the scored episode; no reward normalization.
This is a local directional estimate, not a finite-update return estimate.
Evaluate before and all four after policies on the same16 fresh paths in both
deterministic/lower-sampled modes; upper/gate always deterministic. Coupled
per-step lower noise, unchanged feedback and no input privilege.

Reuse roots310011/310023/310037/310049/310061/310073/310089/310101.
Fresh environment base10800000 + root-index x10000; eval+3001..3016.
Policy seedSeedSequence(43,root,environment,43017); lower noise
SeedSequence(43,root,environment,43019)+primitive-step. Reconstruction retains
Stage-42 original streams. Preflight root310001/new base10790000,2 fresh paths.

Sixteen primary endpoints: each arm's clipped training gain, held-out task
direction, deterministic return change and lower-sampled return change.
Equal root means after path averaging;65536 paired root draws, seed(43,43043),
two-sided percentile Bonferroni16. Positive surrogate plus negative task
direction supports local direction conflict; negative lower-sampled return
additionally supports finite-update utility conflict. Deterministic-only harm
is a distinct deployment diagnostic. Otherwise keep the boundary inconclusive.
No exclusions, extensions, checkpoint selection or outcome-driven changes.

Eight scheduler root tasks,9 CPU/12GiB each, dynamically node001-node006.
Full cost1843200 primitive steps:307200 reconstruction plus1536000 evaluation;
1536 offline native trace audits, zero extra verification steps/optimizer steps.
Preflight7200 primitive steps/24 trace audits,2 CPU/4GiB. Freeze full before
native preflight/fresh outcomes. Source training cost is historical, not rerun.
Only compact JSON returns locally; raw traces and all weights stay remote.

## Limitations

Existing development weights and eight reused roots do not provide independent
algorithm confirmation. Training surrogate and task objective differ in credit,
discount and normalization. The local score estimate can have high variance and
does not establish the effect of a finite displacement or of a remedy. Positive
surrogate alone does not show optimization correctness or algorithmic utility.
