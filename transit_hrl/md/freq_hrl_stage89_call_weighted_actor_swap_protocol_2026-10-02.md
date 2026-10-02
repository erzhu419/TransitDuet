# Stage89: attribution of confirmed call-weighted joint learning

Stage88 confirmed all four primary comparisons on fresh training/evaluation samples. Read its registered update8 joint-call and lower-only weights server-side; no training, selection or allocation changes.
Same eight frozen Stage78 teachers, periods50/100, alpha/envelope, std, values, source Adam and forecaster. U0/L0 denote source actors, UJ/LJ joint-call actors, LL lower-only learned lower.
Seven frozen variants: U0/L0 (base), zero, UJ/LJ, U0/LJ, U0/LL, UJ/LL, UJ/L0. Compose only actors; all other tensors stay exact. Only the two required donor methods are read, not joint-level.
Use32 fresh paired Stage89 environment/action-noise seeds per root/period, horizon1200, identical exogenous histories and standard-noise mapping. Preflight: one root, four fresh seeds, horizon300; mechanics only, no reward gate.

## Frozen Analysis
All22 reward/interaction endpoints use one equal-root bootstrap65536 / Bonferroni22 family, bootstrap seed(89,89089). No pooling Stage87/88/89 as independent teacher populations.
Direct upper attribution requires positive UJ/LJ minus U0/LJ CI lower bounds at both periods. Report UJ/LL minus U0/LL transfer and UJ/L0 minus U0/L0 separately; neither substitutes for direct attribution.
Interaction is (UJ/LJ minus U0/LJ) minus (UJ/LL minus U0/LL). Also report fixed-upper lower differences, joint-versus-lower-only, and zero controls. Inconclusive interaction does not establish equivalence.
If direct attribution fails, retain the Stage87/88 joint-performance evidence with that attribution limit; no new seeds, horizon, decoder, training or checkpoint search to rescue this gate.
Full budget: 3,584 episodes /4,300,800 steps, 32 checkpoint loads, 112 exact actor compositions; zero policy updates, critic/forecaster fits, training traces or checkpoint writes.
Scheduler dynamically places roots on node001-006, no pin; 9CPU/8192MiB with8 evaluation workers, preflight3CPU/3072MiB with2 workers. Sync completion only and pull compact JSON; checkpoint files stay on the server.

## Limitations
Conditional attribution for teacher-initialized fixed-std/decoder MC policies, not full actor-critic or frequency superiority. Lower-only retains an active frozen upper. Cross-training-run actor compositions do not constitute an equal-training-budget method comparison or a retraining counterfactual. Stage67 critic-credit HOLD remains unchanged.
