# Stage83 Iterative MC Learning Protocol

Question: does upper contribute sustained reward beyond same-budget lower-only?
Stage82 joint improved source but lost to lower-only at equal total conditional KL.
Preregister8 mean-learning rounds (preflight2), fresh on-policy native data each time.
Joint spends.0005 KL per level; lower-only spends.001 on lower. No allocation sweep.
Both std vectors, values/Adam, forecaster, sources and Stage78 decoder stay frozen.
This independent MC route does not reopen the rejected Stage67 critic-credit route.

Each round: two batches of16 exogenous scenarios,2 independent action-noise
replicates each. Methods use identical scenario/noise rosters, but their own current
policies; no teacher replay or stale likelihoods. Undiscounted native task MC,
other independent same-scenario return as baseline, mean-only raw policy gradient.
No entropy, shaping or advantage normalization. KL is a sum of per-level source-state
conditional Gaussian averages, not trajectory KL; each round has the same budget.

Evaluate only registered final round8: source,zero,joint-trained,lower-trained,
32 new paired seeds per root/period.8 roots,periods50/100. All12 reward contrasts
share equal-root bootstrap65536/Bonferroni12; no stopping or best-checkpoint selection.
Full budget:18,432 native episodes /22,118,400 steps;16,384 credit episodes,
40,960 score forwards,122,880 backwards,19,392 Fisher JVPs,38,784 exact-KL forwards.
Only32 final inference-weight files remain on servers; no intermediate/raw trace writes.
Scheduler dynamically places9CPU/8GB tasks on node001-006 (preflight3CPU/3GB).
Local pulls contain only completion markers and compact JSON.

Next decision uses final joint/lower-only/source/zero contrasts. No frequency-
superiority, full actor-critic, real-data or source-policy adoption claim from this test.
