# Stage75: sampled upper reference / velocity paths

Stage74's normal-execution base reward was descriptively below zero-residual by about 155 (period50) and 71 (period100), much larger than its supported +0.185 lower-direction response. The residual curve simultaneously changes reference position and planned velocity. Stage75 separates these paths before any further training.

- Source: the same frozen Stage55 teacher clones and causal ridge forecaster; Stage74 completion required for each root. No historical-direction fitting, parameter perturbation, optimizer step, critic fit or forecaster refit.
- Four interventions: R0V0 = base reference/base velocity; R1V0 = residual reference/base velocity; R0V1 = base reference/residual velocity; R1V1 = residual reference/residual velocity. Velocity is selected consistently for actor and value context. The existing anchored Bernstein scale and world clipping are unchanged.
- Paired native stochastic episodes share environment seeds, initial policy RNG, stepwise lower noise, upper proposals and fixed decision times. Target-only forecast inputs preserve the same causal plan prefix across modes. No raw trace or checkpoint writes.
- Preflight: root310001, horizon300, four new 75,095,001..004 seeds. 32 factorial episodes plus 16 original-production replays; aligned modes must match original reward, proposals, noise and calls exactly. Total48 episodes /14,400 steps. Local tests also compare reference/context arrays through every option age.
- Full: eight frozen roots, periods50/100, 32 fresh seeds per root (75,105,001..032 + root-index *10,000), horizon1200. 2048 episodes /2,457,600 steps. Production replays are preflight-only; no repeated check in the full matrix.
- Registered outcomes: native episode return (higher is better) and tracking squared-error integral (lower is better). For each period/outcome report reference effects at both velocities, velocity effects at both references, normal-minus-zero and interaction. All24 endpoints use equal-root paired means and 65,536 root-bootstrap draws with two-sided Bonferroni24; no period/path selection, scale sweep or endpoint removal.
- Reference-target error, reference/velocity residual energy and proposal RMS are descriptive mechanism telemetry, not additional significance claims.
- Scheduler: node001..006, dynamic placement, no node pinning; each full root has eight workers plus coordinator (9CPU/8192MiB), preflight two workers plus coordinator (3CPU/3072MiB). Qualification waits for completion markers. Pull only compact JSON and small markers.

Preflight `pointmaze_upper_paths_stage75_preflight_20261001_r1`: t118800/t118801 both done0 on node006; all16 original-production comparisons passed, 48 episodes /14,400 steps /216 upper calls, source and Adam unchanged. Root computation took16.85 seconds. Six new tests and three original-plan regression tests passed locally. This releases the frozen full matrix without a performance-based selection.

Full `pointmaze_upper_paths_stage75_full_20261001_r1`: t118808..t118815 launched dynamically on node004/001/005/006 (two roots per selected node); t118816 waits for all completion markers. All eight roots completed their period50 factorial block at startup inspection; no native raw traces or checkpoints were pulled.

## Limits

Mixed paths are interventions, not coherent deployment plans. These teacher-initialized development roots do not establish joint-HRL training, OOD generalization or frequency superiority. Stage67 HOLD remains unchanged; this experiment diagnoses the upper execution interface and does not adopt an actor or choose a winning path.
