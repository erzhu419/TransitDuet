# Stage-27 History Information Diagnostic

Freeze the Stage-26 cache, roots 209011/209061, 16 fit/eight evaluation paths,
320/160 opportunities per root. Preflight reuses 208001's four/four rows.
Read raw NPZ only on the server; stage source JSON and code, not raw caches.
No controller reconstruction, controller update or new environment transition.

Probe two objectives on identical observations: future target displacement
rate at 10/25/50 steps, and cached wait-minus-now ISE rates at those horizons.
Regenerate target labels with the source controller's exogenous driver options;
first match all 64 cached measured frames. Future targets enter labels only.
Charge regenerated tape points separately from environment transitions.

Use the 23 current features plus eight target-velocity estimates at fixed
lags 1/10/25/50. Current-repeat and shuffled-history controls use the same
31-column interface; current-repeat slopes are zero. All regressions use
training-only standardization, unit L2 on summed squared error and an
unpenalized intercept. No coefficient, lag or horizon selection. Report
zero and lag-one constant-velocity forecasting controls as well.

Each root uses six multivariate linear fits/27 scalar solves. Report all
horizons, eight evaluation paths, and timing decisions versus both matched
controls and constant actions. Forecast improvement without timing benefit
motivates revisiting decision supervision; improvement in both motivates a
predictive-state implementation; neither is a reason to add more seeds.
One CPU/1.5 GB per task, dynamically placed on node001-node006 via scheduler.
Fourteen focused tests passed, covering observed lags, future-label isolation,
source-prefix matching, training-only fits, units and scheduler/cache scope.

Execution frozen at `c1e5e72a44`: preflight `t101656` completed on node004,
with four training/four evaluation rows, 27 scalar solves and 1,204 generated
tape points. No new environment steps or controller updates; retrieved
62,447 bytes of JSON. The preflight's nearly constant lag-one speed has
training std 2.47e-6; normalization amplifies its float32 variation, producing
history forecast MSE 1.83154e8. Retain this negative result; full settings
remained unchanged. Full tasks `t101659/101660` completed on node004/node006
for 209011/209061, each with 320 training/160 evaluation rows, six linear
fits, 27 scalar solves and 28,824 generated tape points. No new environment
steps or controller updates. Full-run slope std ranges from 0.322 to 0.863;
the preflight scale amplification is absent. See the
[Stage-27 result](freq_hrl_stage27_history_information_result_2026-09-28.md).

## Limitations

This retrospective diagnostic cannot reopen Stage-26 or qualify a policy.
Linear-probe failure does not prove observational sufficiency. Exogenous
forecast skill does not establish an action-conditioned physical belief or
deployed control benefit.
