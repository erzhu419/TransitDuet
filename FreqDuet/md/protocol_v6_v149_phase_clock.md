# V149: independent demand-phase service and physical ETA clocks

Preregister before opening new demand-day outcomes. V148 is development evidence.
No actor, critic, dual, control, V146 calibration, interval support, window, or
quantile changes. No fitted parameters and no additional RL training.

## Fixed candidates

Four shadows on each unchanged old/handoff control path: frozen V141, service
only, clock only, and joint. No candidate chooses or modifies a holding action.

Service: forecast each downstream arrival sequentially using existing running,
holding and execution means. Select completed same-stop mandatory service means
by arrival before/after the published demand cutoff. Historical phase uses actual
arrival, not completion; forecast phase uses predicted arrival, not query time.
Every label must complete strictly before the query. The same 3600s window and
empty-window all-past fallback operate within each phase. No same-phase samples:
retain the original mean and count unsupported stops. Clearing service is not
assumed zero: remaining queues and onboard passengers are represented by observed
completed clearing services. Current in-progress service/ready/known-command
residuals stay unchanged. No future queues, loads or actions are inputs.

Clock: infer progress from observed spatial gap and target-stop geometry for all
AVL sources. Use the same strict moving/pending-stop prefix and V140/V141 current
residence logic, then the same physical running/service/holding/execution clock.
Unsupported segments, boundaries or components retain the frozen point, explicitly
counted. Already structured forecasts must reproduce the original total exactly.
Service-only changes original structured points; joint also treats newly supported
fallbacks. These are separate interventions, not a posthoc total-ETA bias offset.

## Independent experiment

12 archived B200 actors, checkpoint199; 32 new demand days 2651001--2651032;
samplers 1490000000--1490000031 shared across actors/modes by day. Check prior
archive admission and script day ranges before collection. 384 paired cases,
768 full-day frozen-control paths. No fitting/calibration on these days; one readout.
V146 remains sealed at lower3s/upper0s; native intervals are produced only by the
unchanged collector/evaluator, not by the candidate. Tails .1/.1 unchanged.

Primary fixed joint candidate must satisfy all of:
- Both modes: crossed95 upper<0 for clearance full-ETA MAE change and change in
  absolute cell-level signed bias, each relative to frozen V141.
- Both modes: crossed95 upper<=0 for all-day and hour13 full-ETA MAE changes.
- Crossed95 upper<=0 for same-identity source-switch gap-change MAE burden change.

The intersection gate tests one fixed joint claim; no winning-arm selection.
Service clearance improvement and clock switch-burden improvement are separately
reported mechanism checks, not alternate routes to admission. All three arms'
levels and paired differences are retained, including failures.

Average queries within follower/target-stop episodes, then equal actor/day cells;
use established 5000-draw crossed actor/day bootstrap, seed20260803. Report actor
and day margins. Switch conditional means remain explicit, with no CI when the
grid is incomplete. Whole-grid switch burden divides switch errors by every
valid same-identity matched decision; no observed switch contributes zero burden,
not a fabricated conditional observation. Missing required labels block the gate;
incomplete crossed support is inconclusive, not zero-filled.

Preflight: synthetic strict-prefix, phase/support and clock tests, plus one
consumed V146 pair to verify frozen structured-point reconstruction. The unchanged
collector reuses its established exact action/endpoint/clock/interval proof.
No new-day outcomes are used in preflight. All runs/tests/aggregation use scheduler
on node001--006; raw paths and checkpoints stay there. Only small summary/log pulls.

Passing establishes prediction evidence, not a controller replacement. Binary
published demand/clearance phase does not model arbitrary future demand profiles.
V146 intervals still concern the current stage, not full-ETA coverage.
