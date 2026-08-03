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

## Remaining Physical-Credit Defect

The current lower leakage state is a fixed-length deque of action events pooled
across all buses in a direction.  It therefore measures recent aggregate
holding volume, not the physical downstream delay accumulated by a particular
trip.  The next rebuild will make the lower penalty depend on each trip's
cumulative executed holding and make `DriftFB` depend on completed-trip totals
over a causal time/trip window.  The legacy event-pooled mode will remain only
for reproducibility.

## Active Clean-Protocol Screens

- `protocol_v25_causal_contract_interval_ep80_s16_v1`: 9 configurations, 16
  train seeds, 12 frozen CRN evaluation seeds; isolates lifecycle, forecast,
  decomposer-prior, and exact interval-credit choices.
- `protocol_v26_repaired_timetable_curve_ep80_s16_v1`: 8 configurations under
  the same seed protocol; compares the two-coefficient repeatability control
  with three/four-coefficient local curves, a joint bidirectional curve, EMA,
  and pure interval credit.

Neither screen uses the old domain-specific paper-main aliases.

## Promotion Gate

The rebuilt method is promoted only if it passes paired frozen-policy tests on
service cost, wait, headway CV, fleet overshoot, completion, action stability,
and seed failures. Mean improvement alone is insufficient. The winning
structure then receives multi-domain 200-episode confirmation against the
external fixed-headway baseline.
