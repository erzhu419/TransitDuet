# Stage88: frozen fresh-sample replication

Stage87 joint-call beat lower-only and joint-level under Bonferroni20. Replicate that result without changing the algorithm, decoder, teacher, allocation, updates or analysis; do not tune from the new results.
Same eight teacher/root IDs, periods50/100, Stage78 alpha/envelope, fixed std, independent MC core, three training methods and five evaluation variants. Initialize from original teachers, not Stage87 final weights.
8 fresh rounds of64 episodes per method/period; both joint actors use all64 before either changes. Joint-call upper .0005/lower .001-.0005/period; joint-level .0005 each; lower-only .001. Same nominal call-weighted budget matching and mechanical checks as Stage87.
Stage88 shifts every scenario, action-noise and evaluation seed by1,000,000 into a disjoint namespace; matching exogenous rosters across methods remains. Preflight: one teacher, horizon300, 2 rounds of8 episodes, four evaluation seeds; mechanical checks only.

## Frozen Analysis
Only update8 evaluated on32 fresh seeds per teacher/period. Same20 endpoint family, equal-root bootstrap65536, Bonferroni20 and bootstrap seed as Stage87. Stage88 CI is calculated independently; never pool the two stages as16 independent teachers.
Confirmation requires strictly positive CI lower bounds for joint-call minus lower-only and joint-call minus joint-level at both periods: all four primary endpoints. Preflight does not apply the reward gate.
If any primary fails, retain Stage87's positive result and Stage88's nonconfirmation together; no threshold, seed, allocation or horizon sweep. The confirmation gate is reported separately from mechanical validity.
Full budget unchanged: 27,136 episodes /32,563,200 steps, 640 actor mean updates, 48 final server checkpoints. Zero fitted critics/PPO/Adam/forecaster fits/raw traces; source and inactive actors frozen.
Scheduler dynamic node001-006 placement, 9 CPU/8192MiB per root, 8 rollout workers; completion-only sync and compact JSON pulls. Next after results: fixed-checkpoint actor swaps to attribute learned upper contribution.

## Limitations
Independent training/evaluation samples conditional on the same eight frozen teachers, not a new teacher population, domain or initialization method. Teacher-initialized fixed-std/decoder MC learning remains distinct from full actor-critic/frequency-superiority claims; lower-only retains a frozen active upper.
Stage67 critic-credit HOLD remains; the two-stage analyses do not support pooling duplicate teachers or equal compute claims. Positive joint reward alone does not isolate upper's direct effect from coadaptation.
