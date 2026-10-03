# Stage97: Fresh Teacher Decoders

Stage96 rebuilt eight new teachers. Stage97 reconstructs their decoders using the existing Stage76/78 command-response rule, not a new reward-tuned rule.

- Source: full Stage96 roots 410011, 410023, 410037, 410049, 410061, 410073, 410089, 410101. Preflight uses root410011's full BC labels and checkpoints, never the short root410001 artifacts.
- At each period (50/100), reconstruct all 8 x 1200 causal BC states and reproduce saved tanh-command MSE. Use the same-label velocity q99 and axis envelope. Counterfactual zero-mean Gaussian proposals have predeclared root/label/period seeds and the new teacher's std.
- Start at min(1, BC RMSE / original joint-command-change RMS), then halve until the actual nonlinear lower-command change is at most BC RMSE. Retain the first feasible scale. Both period decoders freeze before any native probe; no native reward selects scale or roots.
- Native probes: zero/bounded, four fresh stochastic episodes per mode/period, no learning. Preflight H=300: 16 episodes / 4,800 steps. Full H=1200: 16 episodes / 19,200 steps per root, 128 / 153,600 total. No CI or performance gate at this prerequisite stage.
- Fixed offline budget per root: 16 label archives, 19,200 reconstructed state rows, 288 curve decodes, 2 clone loads, 1 forecaster load. Nonlinear contraction has variable passes; exact actor row/forward costs are retained in the realized budget.
- Scheduler: dynamic node001-006, 3 CPU / 3GB preflight, 9 CPU / 8GB full; compact JSON and completion markers only return locally. Full launch follows native preflight qualification. No checkpoint, raw trajectory or new forecaster writes.

Next: rebuild native joint and matched independent/common lower donors, then apply the unchanged staged-upper rule and its four corrected endpoints. Preserve zero-plan/U0/fixed-UJ controls, all roots and both periods; do not pool with Stage94/95.
