# Stage70: fixed causal state baselines

Use all Stage69 independent archives, unchanged roots/periods/execution/batches and fixed Stage64/67 critics. No new native steps, fits, optimizer updates or checkpoint writes. Preserve Stage67 HOLD.

Compare raw MC-minus-common-time-baseline, MC-minus-control-state-baseline, MC-minus-factored-state-baseline and both GAE estimators. Primary noise estimates use raw independent episode gradients, not centered/scaled contributions. Report separately normalized batch directions as a PPO diagnostic. All baselines are fixed before these archives; coefficient is 1 with no coefficient fitting or baseline selection.

For g_state = g_common - h, report Var(g_state) = Var(g_common) + Var(h) - 2 Cov(g_common,h), covariance traces, raw SNR and independent-batch cosines. Preserve negative unbiased signal-power estimates. Reproduce Stage69 value probes, raw common-MC/GAE noise and normalized GAE repeatability. Baseline inputs are the physical/history/plan context and clock constructed before the current lower action; predictions are detached. Thus fixed state subtraction has zero expected current-action score contribution under the native latent Gaussian policy, but empirical means can differ.

Full replay: 1,024 episodes, 1,228,800 reconstructed lower calls, 18,432 upper calls; 2,048 score forwards / 14,336 backwards. Scheduler dynamically selects node001-node006 with 5 CPU / 4 GiB per full root, 3 CPU / 3 GiB preflight. Pull only markers and compact JSON; archives/weights remain server-only.

## Limitations
Same development policies and paths as Stage69, not another independent confirmation or a reward trial. MC expectation refers to the uniform-time discounted surrogate, not undiscounted native reward. Batch pairs are dependent descriptive observations, not extra roots/CI samples. Variance reduction cannot itself release HOLD or prove a useful policy update; any actor intervention requires a separate frozen protocol.
