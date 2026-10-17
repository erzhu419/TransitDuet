# Stage289: waiting-constrained cross-future joint gains

Use only Stage288's 60 training prefixes, 360 states, 24 futures, six actions
and 16 fixed raw-reference profiles. No held rows/labels, fitting, simulation
or actor updates. Raw feasibility and native original fallback are unchanged.

- Primary: select on futures 0-15, freeze actions, then open/evaluate 16-23.
  Controls are saved raw and unchanged Stage247 choices. Information probes
  minimize finite-MC mean total cost, either unconstrained or subject to mean
  wait no higher than own raw. No wait margin/grid. Ties prefer raw, then
  original, then candidate index. Every probe includes the raw fallback.
- Two fixed 16-to-8 reverse checks hold out futures 0-7 or 8-15. Selection
  and evaluation never overlap within a query. These checks are not blinded
  and do not select a winner or replace the primary direction.
- Report selection16 versus independent8 cost/fleet/service/wait, both
  evaluation halves, and paired contrasts versus raw, cost-only and Stage247.
  Average profiles and six probes before a 60-prefix stratified bootstrap
  (six prefixes/root/regime, 5000 draws, seed289). Shared-future conditional
  MC standard errors are separate; profiles/directions are not seed replicas.
- Joint retention requires primary constrained-versus-raw cost and wait
  CI95 upper bounds below zero and negative cost/wait points in both four-draw
  evaluation halves. All reverse results are reported without selection.
- One scheduler analysis task, one CPU/3 GiB, any node001-node006, no pin.
  Read the 2160 train chunks server-side; keep proposals remote and pull only
  the compact scalar report. Zero additional native episodes or fits.

Limitations: these reused development futures are independent of each query's
selection bank, not fresh confirmation data. Label-informed choices measure
finite-MC information value, not a deployable controller or a guaranteed
oracle upper bound. New-prefix confirmation remains a separate next decision.

Run: `native_transit_wait_constraint_stage289_20261017_r1`.

Six focused tests pass, including independent-bank wait harm correctly failing
the joint gate, primary-ledger ordering, raw fallback/ties, no model-duplicate
uncertainty reduction, train-only loading, parent completion and unpinned
scheduler resources. Both CLI entry points start.
