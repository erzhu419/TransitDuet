# Stage9 Versus A Frozen Timing Portfolio

[Registered plan](../protocols/plan_validity_stage9_timing_portfolio_2026-11-12.json).
Reuse all eight completed PointMaze builds. In each root, select the lower
saved timing-selection ISE of learned termination and best constant clock;
ties choose the clock. Seven roots choose learned termination and root
1981440000 chooses the clock. Stage9 and the selected distance threshold are
unchanged. Freeze all choices before any new episode. The original learned-only
HOLD and its test roster remain unchanged.

Evaluate all three rules on 64 disjoint paths per root, 512 paired paths
(1,536 episodes), 24 calls each, 1,843,200 native steps and no fits.
For each comparator require a positive lower nominal paired root-t 95% bound
on absolute ISE gain, at least 2% pooled improvement, and positive first
and last 32-path half gains. Both contrasts and all resource admissions must
pass. Tail ISE and sustained failures are descriptive.

Charge the controller once per method, the learned branch plus the separately
paid clock search for the portfolio, and the preceding distance diagnostic
as shared development cost for all methods. The caps remain 10,000 CPU seconds
at 100 deployments, 6.5 million offline steps and 10ms control p99. Verify
admission again using measured evaluation control latency; evaluation replay
is separately accounted. This tests fresh paths under existing trained roots
against a selection-based portfolio, not a fresh training recipe or a
learned-only, cross-task, or end-to-end algorithm claim.
