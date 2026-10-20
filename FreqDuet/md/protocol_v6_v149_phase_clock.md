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

## Execution

Registered source c0658db77a; final execution source d7996d60b4 adds server-only
per-query shadow archives and JSON-safe counters, with no candidate/gate changes.
Final-source preflight t199050 on node006: 11 tests and one consumed old/handoff
pair passed, native exit0; no new demand days read. Formal scheduler shards
t199051--t199053 plus t199068/t199069 cover all384 pairs, planned40 workers,
allowed node001--006 only. Queued t199054 was cancelled without running and split
into two7-worker ranges247:316 and316:384 to fit dispersed free CPU slots; no
case, sampler, candidate or gate changed. Formal collection and paired computation
have started; predictor effects have not been read out.

## Independent readout

384 crossed cells/768 paths, five shards exit0, required labels complete. Original
joint gate FAIL; service mechanism PASS; clock mechanism FAIL. Seconds below;
CI95 uses the registered5000 crossed actor/day draws. Joint equals service.

| Metric | old: V141 -> service | handoff: V141 -> service |
| --- | --- | --- |
| Clearance signed bias | 21.721 -> 15.305 | 21.665 -> 15.259 |
| Clearance MAE | 40.791 -> 38.016 | 40.820 -> 38.052 |
| Clearance MAE delta CI95 | -2.775 [-3.362,-2.248] | -2.768 [-3.367,-2.233] |
| All-day MAE delta CI95 | -0.0740 [-0.0906,-0.0593] | -0.0738 [-0.0907,-0.0591] |
| Hour13 MAE delta CI95 | -2.410 [-2.924,-1.949] | -2.404 [-2.929,-1.938] |

Both modes improve MAE margins for all12 actors/all32 days in these three scopes.
All-day absolute-cell-bias nevertheless increases0.091/0.105s. Source-switch burden
delta=-0.0000867s/valid matched decision, CI95[-0.0003549,+0.0001631]: non-harm
unconfirmed, not proven harm. Conditional switch MAE24.605->24.601 has no CI
(346/384 contributing cells).

Read-only audit t199125/node006/source6c6aee1c13 passed three tests/384 cells/exit0,
reproduced endpoints and burden, no new rollout/fitting. Clock adds zero structured
queries (max delta5.684e-13s). Retained fallback:2329/34209 switches, burden0.182892
of0.321451s/decision, blocked by strict segment evidence or no service survivors.
Remaining downstream-service bias11.905/11.930s;13958/13909 clearance queries lack
same-phase samples at one or more downstream stops.

Next: preserve service as a prediction candidate, not a promoted controller. Read
residual service error by phase support and causal inputs at retained fallbacks;
verify nonzero target coverage on consumed paths before a new clock registration.
Keep the joint rejection, frozen settings and V149 single-use confirmation intact.
Small summaries: freqduet/results_freqduet/protocol_v6_v149_phase_clock/summary/status.json
and readout_6c6aee1c13/status.json; raw paths remain on the server.
