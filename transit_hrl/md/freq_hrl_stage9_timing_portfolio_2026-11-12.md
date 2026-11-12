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

## Result

[Summary](../results/pointmaze_stage9_timing_portfolio_20261112/summary.json) and
[exit records](../results/pointmaze_stage9_timing_portfolio_20261112/completion.json).
All eight tasks t214829-t214836 exited 0. The budget closed at 1,843,200
evaluation steps, 36,864 upper calls, 1,687.26 CPU seconds and zero fits.
Only 48,513 bytes of final summaries were returned.

| Reference | Reference ISE | Absolute gain [nominal root 95% CI] | Relative gain | First / last half | Favorable roots |
| --- | ---: | --- | ---: | ---: | ---: |
| Selected distance threshold | 1.06097 | 0.06163 [0.01980, 0.10345] | 5.81% | 5.57% / 6.05% | 7/8 |
| Timing portfolio | 1.27402 | 0.27467 [0.19309, 0.35625] | 21.56% | 21.02% / 22.09% | 8/8 |

Stage9 mean ISE is 0.99934. It trails the distance threshold by 2.46% on
root 1981420000. The seven roots selecting learned termination show 19.86%
pooled improvement descriptively; the eighth uses a clock and is retained.
The joint gate passes. The tighter comparison is the selected distance
threshold, not the portfolio.

| Method | Max charged CPU seconds | Max offline steps | Max root p99 (ms) | 10ms misses |
| --- | ---: | ---: | ---: | ---: |
| Stage9 | 6,779 | 5,646,391 | 1.64 | 1 |
| Distance threshold | 6,815 | 5,611,200 | 1.19 | 0 |
| Timing portfolio | 7,585 | 6,420,000 | 1.78 | 0 |

All pass the shared caps, which are not equal realized cost. Mean root
upper-decile episode ISE is 1.549/1.639/1.981 in that order; sustained
failure counts are 0/0/1. Those are descriptive, not risk guarantees.
The original learned-only HOLD and its unused test roster remain unchanged.
