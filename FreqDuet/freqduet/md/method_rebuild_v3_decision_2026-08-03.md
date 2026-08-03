# FreqDuet Method Rebuild Decision (2026-08-03)

## Decision

Do not freeze the paper yet. Continue with a bottom-up method repair before
the confirmatory 200-episode paper matrix.

## Evidence

- The compact continuous upper planner is promising: service cost `1.21970`
  versus `1.23690` for the previous four-dimensional main, but its paired CI
  still crosses zero.
- The original exact categorical upper planner is significantly worse and can
  collapse to the zero curve, so discrete action libraries are not promoted.
- A physical lower-state encoding improves the weak discrete-history reference
  from `1.32828` to `1.25413`, primarily by removing unstable holding. The
  near-zero lower action shows that state scaling is necessary but not yet a
  complete controller.
- Correct trip reset worsens the weak reference by `+0.01253`, CI
  `[+0.00039,+0.02678]`. The old score therefore partly depends on invalid
  cross-trip action/state persistence and cannot be retained for publication.
- Trip-end holding finalization alone has exactly zero effect because the old
  upper state reads stale bus fields instead of lifecycle-owned feedback.

## Structural Defects Being Repaired

1. The continuous actor emits a command, EMA changes it, and replay stores the
   changed action while policy optimization evaluates the raw command.
2. EMA dynamics depend on the previous plan, but that plan is absent from the
   compact upper state.
3. Promotion creates variable upper decision intervals while SAC uses one
   fixed discount per transition.
4. Upper holding state slots use reused-bus residue rather than completed-trip
   lifecycle data.
5. The lower controller carries pending actions across trips and bootstraps
   through its final observable state despite having no subsequent controlled
   transition.
6. Critic L1 uses an unnormalized parameter sum. At initialization its default
   contribution is about `10.5`, far larger than the observed upper Q MSE.
7. Upper `beta_ood` and `weight_reg` exist in the trainer but were not wired
   from configuration.

## August 3 Contract Audit Update

The first repair pass is complete and covered by the full unit suite:

- forecast horizons are now measured in demand bins rather than simulator
  seconds, and the harmonic forecast advances phase causally;
- every harmonic alias uses the historical OD prior when requested;
- strict lower runs emit a physical terminal transition, reset per-bus state
  at a trip boundary, record zero holding decisions, and never reuse an
  unobserved stale action;
- upper replay is separated by planner stream, so up- and down-direction SMDP
  transitions no longer alternate inside one invalid replay chain;
- additive interval credit partitions directional passenger queue exposure,
  headway deviation, and fleet excess over the exact transition interval;
- every shard records source and resolved-config fingerprints, and aggregation
  rejects mixed-source runs.

The implementation clarification from this audit is that `uppercompact` is not
a one-dimensional flat shift.  With its resolved configuration it is a
direction-local two-coefficient linear headway curve (`action_dim=2`), with a
separate active plan and replay stream for each direction.  The new curve
matrix therefore treats the two-coefficient row as a repeatability control and
tests three/four coefficients plus a joint six-coefficient bidirectional plan
as the genuinely richer alternatives.

## Submission Blocker In The Previous Paper Main

The promoted `cfaction_domainbest_v1` aliases are not one deployable learned
policy with fixed method hyperparameters.  The terminal, high-noise, and OD
aliases override the upper actor with a fixed `-20 s` command, while the
rush-shift alias enables a configuration rule that selects the fixed-headway
expert under a known peak shift.  The four-domain aggregate is therefore a
domain-informed policy portfolio and cannot support a claim that one FreqDuet
controller learned to outperform or match fixed headway across domains.

These runs remain useful as an oracle/engineering upper bound, but they must be
removed from the canonical paper-main claim.  A publishable main must use one
algorithmic configuration across held-out domains, with only exogenous
environment parameters changing.

## Physical-Credit And Objective Repair

The lower leakage implementation now has an explicit `trip_cumulative` mode.
Its penalty and Lagrangian cost use cumulative executed holding for the same
physical trip; they no longer mix action events from concurrent buses.
`DriftFB` in this mode uses only completed-trip totals from the causal
directional history.  The event-pooled `rolling_action_window` mode remains the
default for exact reproduction of old configurations.

Passenger removal during boarding and alighting now uses order-preserving
boolean masks instead of set differences, so FIFO station queues and onboard
ordering are deterministic.

The lower controller's forward-headway state had a second physical defect.  It
looked up the preceding bus in `trajectory_dict`, but all writes to that
dictionary had been commented out.  The policy therefore received a
distance/average-speed approximation at almost every stop while evaluation used
same-stop arrival events.  The state now queries the shared event recorder
before recording the current arrival, giving an exact causal same-stop headway.
The first arrival for each stop and direction uses a guarded spatial/default
fallback.  Episode diagnostics expose event, spatial-fallback, and default
counts plus the event-use rate; the integration test requires the event-state
count to equal the evaluation headway sample count.

The physical lower-state encoder also inferred the upper target as
`forward_headway / (1 + deviation)`.  This is undefined for a zero same-stop
headway and silently substituted `600 s`, violating the manual's requirement
that the lower controller explicitly receive `H_U(t, dir)`.  Protocol v3 uses
an `explicit_target_v2` input schema: the raw routing identifier remains
available to the runner, while the policy encoder receives the exact target and
computes deviation itself.  The inferred-target schema remains as a named
ablation.  Event and guarded-spatial headway modes are also configurable, so
the physical state repair can be tested rather than assumed.

The audit also found that the environment measurement vector used restricted
horizon wait while the scalar `service_cost` used boarded-only observed wait.
Both scalar costs are now emitted explicitly.  Old configs retain `observed`
as their default; the new versioned paper protocol must declare
`objective.wait_metric: restricted`, and matrix/external-baseline manifests
reject mixed protocol versions.

## Active Clean-Protocol Screens

- `protocol_v25_causal_contract_interval_ep80_s16_v1`: 9 configurations, 16
  train seeds, 12 frozen CRN evaluation seeds; complete with 1,728 frozen
  rollouts.  Additive interval credit reduces restricted wait by `0.0427 min`
  relative to the legacy compact planner, hierarchical 95% CI
  `[-0.0655, -0.0220]`.  Its observed-wait composite delta is `-0.0199`, CI
  `[-0.0443, +0.0054]`, so the composite claim is not yet confirmatory.
  Forecast and no-boundary-prior harmonic variants do not pass promotion.
- `protocol_v26_repaired_timetable_curve_ep80_s16_v1`: 8 configurations under
  the same seed protocol, complete with 1,536 frozen rollouts and verified
  source/config/tape identity.  The joint three-coefficient-per-direction plan
  has service cost `1.1988`, and the direction-local four-coefficient plan has
  `1.2006`.  Relative to the legacy compact reference their paired deltas are
  respectively `-0.0287` (95% CI `[-0.0529, -0.0076]`) and `-0.0269`
  (`[-0.0477, -0.0098]`).  Relative to the strict two-coefficient repair,
  however, the deltas are only `-0.0088` and `-0.0070`, and both intervals cross
  zero.  Global-three versus local-four is also indistinguishable
  (`-0.0018`, CI `[-0.0141, +0.0109]`).  The defensible conclusion is that the
  strict lifecycle/causal credit repair drives the confirmed gain; richer curve
  parameterization remains a secondary candidate.  Removing upper wait credit
  or EMA does not improve the strict reference.

Neither screen uses the old domain-specific paper-main aliases.

Protocol v3 must retrain from scratch with the event-headway state,
`trip_cumulative` drift, and explicit restricted-wait objective.  It will carry
both strict compact and global-three timetable variants until the new physical
contract identifies a winner; v2 checkpoints cannot be evaluated as v3.

The initial v3 selection matrix contains 13 source-identical configurations:
compact/global-three/local-four planners; 15/30/60/90-second trip drift budgets;
and isolated rolling-drift, no-lower-drift, no-DriftFB, spatial-headway,
inferred-target, boarded-only-training, and legacy-credit ablations.  A config
validator rejects any resolved difference outside each ablation's declared
fields.  Mixed observed/restricted training objectives are never compared via
the ambiguous generic `service_cost`; aggregation requires and reports both
explicit cost views.

## Promotion Gate

The rebuilt method is promoted only if it passes paired frozen-policy tests on
service cost, wait, headway CV, fleet overshoot, completion, action stability,
and seed failures. Mean improvement alone is insufficient. The winning
structure then receives multi-domain 200-episode confirmation against the
external fixed-headway baseline.

## Locked Selection And Confirmation Protocol

The protocol-v3 structure-selection run is fixed before reading any v3 result:

- run: `protocol_v3_physical_contract_selection_ep80_s20_v1`;
- 80 training episodes;
- train seeds: `7,11,17,23,31,37,42,53,61,71,83,97,101,113,127,139,149,157,167,179`;
- frozen CRN evaluation seeds:
  `20101,20107,20113,20117,20123,20129,20143,20149,20161,20173,20177,20183`;
- primary endpoint: `service_cost_restricted`;
- uncertainty unit: hierarchical bootstrap over training seeds and paired
  evaluation seeds;
- paired test: two-sided sign-flip test over per-training-seed mean deltas,
  with Holm correction across candidate-reference comparisons for each metric.

The deployable candidate set is compact b30, global-three b30, local-four b30,
compact b15/b60/b90, and rolling-drift b30.  No-lower-drift, no-DriftFB,
spatial-headway, inferred-target, observed-training, and legacy-credit rows are
mechanism or contract controls and cannot become the main method from this
screen.  A candidate is eligible only with complete finite runs and no paired
95% interval showing degradation in unserved rate, fleet overshoot, or trip
completion.  If no candidate has both a restricted-cost interval below zero
and Holm-adjusted `p <= 0.05` versus compact b30, compact b30 is retained.  If
several pass, the lowest restricted-cost mean is selected; differences below
`0.005` retain the lower-dimensional planner.

Only one selected configuration proceeds to the independent confirmation set:

- 200 training episodes;
- train seeds:
  `211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317`;
- frozen CRN evaluation seeds:
  `30103,30109,30113,30119,30133,30137,30139,30161,30169,30181,30187,30197,30203,30211,30223,30241`.

Selection seeds may not be reused for confirmatory claims.  Multi-domain and
external-baseline comparisons use the locked winner and confirmation seeds;
any later tuning creates a new method version and a new untouched seed set.
