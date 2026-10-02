# Stage87: decision-call-weighted KL and full shared MC samples

Question: does matching decision-call exposure close the joint-versus-lower learning gap? Stage85/86 ordering remedy is closed; Stage84's positive direct upper effect remains.
Train joint-call, joint-level and lower-only afresh from the same frozen Stage78 teacher/decoder, at periods50/100. Same exogenous rosters, 8 fresh rounds of64 episodes per method/period; every active actor uses all64 episodes before either actor changes. Preflight: one root, 2 rounds of8 episodes, horizon300; no performance gate or CI.
Joint-level: upper/lower nominal KL .0005/.0005. Joint-call: upper .0005, lower .001-.0005/period. Lower-only: .001. No radius, allocation, seed or checkpoint selection. Lower-only retains the active frozen upper; it is not flat RL.

## Budget Definition
For horizon H divisible by p, lower makes H decisions and upper H/p. With fixed environment, decoder and policy std, the latent-history likelihood chain rule gives D_KL(P_old||P_new)/H = E_old[sum(lower conditional KL)/H + sum(upper conditional KL)/H]. Conditional Gaussian KL averaged over old-history samples therefore estimates K_lower+K_upper/p; it is not K_lower+K_upper. Fixed decoding is a shared measurable map and cannot increase trajectory KL.
Joint-call and lower-only match this nominal per-native-step budget at .001 per update; joint-level is .00051/.000505 at periods50/100. The mean-only directions are normalized within each actor, so no cross-level gradient magnitude allocation is used.
Record exact empirical conditional KL, both summed and call-weighted, actor gradient episode/decision counts and old-logp replay error. Existing local-curvature band [.5,2] times the nominal radius is a mechanical stop, not a performance-selection rule.

## Frozen Experiment
Eight roots310011/310023/310037/310049/310061/310073/310089/310101; fresh Stage87 training/evaluation roles, 32 final evaluation seeds per root. Only last registered update evaluated; final inference weights stay server-side. Source/std/critics/Adam/forecaster unchanged; no critic fit, PPO, raw traces or local native computation.
Five variants: source, zero, joint-call, joint-level, lower-only. All20 reward contrasts in one equal-root bootstrap65536/Bonferroni20 family. Primary: joint-call minus lower-only and joint-call minus joint-level at both periods. Missing roots or mechanical failures invalidate aggregation.
Scheduler dynamically places one root task on any node001-006, 8 rollout workers +1 parent CPU, 8192MiB/task; completion-only sync. Full budget: 27,136 native episodes /32,563,200 steps, 640 actor mean updates, 48 final server checkpoints.
If joint-call still loses to lower-only, retain the negative result and close this fixed-allocation remedy. Cross-zero CI is inconclusive, not equivalence. No performance-driven rerun or allocation sweep.

## Limitations
The chain-rule statement uses population expectations on old-policy histories; reported exact conditional KL is its finite-sample estimator. Nominal matching does not ensure equal empirical KL or global KL from initialization to the final policy; sums across updates are not final trajectory KL. This changes allocation together with its call-weighted total versus joint-level, not an optimal-allocation proof.
Teacher-initialized, fixed decoder/std MC mean learning; not full actor-critic HRL or frequency superiority. Stage67 critic-credit HOLD and the closed frequency-superiority claim remain unchanged.
