# FreqDuet Submission Re-audit and Protocol v4 Rebuild

Date: 2026-08-03

## Decision

Do not freeze the manuscript from the legacy paper package or from protocol v3.
The project has a defensible research question and several repaired causal
contracts, but the active controller still does not implement the complete
method claimed in `GPT.md` and `dev_manual.md`.  The next step is a protocol-v4
method rebuild followed by untouched-seed confirmation.  Paper assembly starts
only after that gate.

Protocol v3 remains useful as a source-identical screen of the earlier fixes.
It cannot become the final paper main without resolving the blockers below.

## Submission-blocking Findings

### 1. Planned and executable headways diverge

The upper actor outputs a headway-adjustment curve, but terminal launch times
are formed by recursively adding those target headways and then clipping every
trip to its original launch time plus or minus 45 seconds.  A constant `-30 s`
headway command produces executable gaps `330, 345, 360, 360, ...`; a constant
`+20 s` command produces `380, 380, 365, 360, ...`.  The target remains 330 or
380 seconds for all affected trips even after the executable schedule has
returned to 360 seconds.

Consequences:

- the lower policy is goal-conditioned on a target the terminal scheduler does
  not execute;
- a 45-minute upper plan changes only the first few launch gaps;
- negative target shifts are only weakly reachable by non-negative holding;
- the method cannot yet claim that the upper action is a smooth executable
  timetable/headway plan.

Protocol v4 must use one physical action definition.  The preferred design is
a bounded smooth launch-shift curve, with executable launch time
`T_i = T_i^base + shift_i` and lower target equal to the resulting consecutive
launch gap.  A constrained headway-curve projection is an admissible competing
implementation only if projected targets and scheduled gaps agree exactly.

### 2. Backward headway has no valid measurement contract

The forward headway is now the exact causal interval since the preceding
same-stop arrival.  The backward value is copied from the scheduled following
bus's previous forward headway, observed at another stop and earlier time.

In an 80-trip zero-action audit with 1,352 matched same-stop follower events:

- mean absolute error was 52.2 seconds;
- the 95th percentile absolute error was 153.5 seconds;
- only 67.3% of estimates were within 60 seconds;
- 5.8% of scheduled following buses were not physically behind the current bus.

This stale proxy currently enters the lower observation and symmetric headway
reward.  Protocol v4 must remove it from the base reward.  It may expose a
named causal AVL spatial-gap or ETA feature, with source counts and held-out ETA
error, but the legacy proxy is ablation-only.

### 3. Credit is not actually frequency separated

The upper interval reward receives total queue exposure, then also receives a
separate low-frequency passenger-wait credit.  The lower receives a
high-frequency wait shaping term whose configured high-share cap and weight
limit it to roughly one percent of the base headway reward.  Thus the current
implementation separates demand features more strongly than it separates
control credit.  It does not yet implement the manual's claimed allocation
`upper -> W^L` and `lower -> Delta W^H_local`.

Protocol v4 must label generated passenger mass with a causal LF/HF share from
the pre-update harmonic prediction.  Queue exposure and boarded waiting can
then be partitioned without double counting:

- upper interval credit uses LF-equivalent queue exposure;
- lower delayed credit uses HF-equivalent boarded waiting;
- LF plus HF source-data totals must reconstruct total waiting within numerical
  tolerance;
- promotion changes subsequent attribution through the causal LF state, not by
  relabeling past passengers.

The current implementation computes the share at boarding time from the then
current residual.  This is not a source-data attribution: a passenger who was
generated before a promotion event can be relabeled after the event.  Protocol
v4 therefore freezes `lf_share` and `hf_share` on each passenger at generation.
For every passenger and every aggregate, tests must assert
`lf_share + hf_share = 1` and `LF wait + HF wait = total wait`.

Upper wait credit must also follow the executable plan that owns the service
trip, not merely the wall-clock interval in which the eventual queue outcome is
observed.  A new plan often starts before buses controlled by the preceding plan
have reached downstream stops, so interval-only assignment shifts old-plan
outcomes onto the new action.  Protocol v4 uses plan-owned LF passenger wait as
the primary upper credit.  Wall-clock interval credit is retained only as a
named ablation and may score headway/fleet terms without duplicating LF wait.

### 4. Upper fleet utilization is mis-scaled

The upper state divides on-route buses by `env.max_agent_num`.  With the full
262-trip timetable, `max_agent_num` is 262 even though the fleet target is 12.
A full fleet therefore appears as approximately 0.046 rather than 1.0.  The
state must use the active fleet budget, and tests must assert scale invariance
to timetable row count.

### 5. Holding externality is not measured

The paper objective reports station waiting, headway CV, service completion and
concurrent-fleet overshoot, but not passenger in-vehicle or total journey time.
Holding delays every onboard passenger, so an unloaded and a full bus currently
have the same action penalty.  A transportation-journal submission needs at
least a no-harm outcome for this externality.

Protocol v4 must report observed and fixed-horizon restricted in-vehicle time,
total passenger journey time, and load-weighted holding person-seconds.  The
lower reward should include a configurable load-weighted action cost.  Waiting
remains the pre-registered primary service endpoint; journey and onboard delay
are safety/trade-off endpoints, not silently folded into a new scalar after
results are seen.

### 6. Fleet is a concurrency cap, not yet a fixed vehicle circulation model

When no terminal vehicle of the required direction exists, the simulator can
create a new bus as long as concurrent on-route buses remain below the target
plus buffer.  Peak concurrency is therefore not the same as physical fleet
inventory.  Protocol v4 must either enforce a fixed vehicle pool with terminal
readiness/block circulation or rename the resource everywhere as concurrent
service capacity.  A paper claim of executable terminal dispatch requires the
fixed-pool version as the main environment.

The mismatch is observable even under zero holding.  With a nominal fleet of
12, the legacy simulator instantiated 14 vehicles with zero buffer and 15 with
a buffer of one or three; peak concurrency was 12 or 13, respectively.  The v4
main environment must never instantiate more than `N_fleet` physical vehicles.
When no vehicle is ready at the requested terminal, dispatch must wait and log
the readiness delay instead of creating another bus.

### 7. Algorithm and reproducibility contracts need tightening

- Planner variants with different action dimensions consume different random
  initialization streams before the lower network is created.  Component
  initialization seeds must be independent and recorded.
- The upper bounded-policy log probability uses a normalized-action entropy
  convention while the lower continuous policy includes physical action scale.
  Protocol v4 must choose and test one explicit convention.
- The local trainer is an adapted robust ensemble SAC, not an exact drop-in
  implementation of every RE-SAC equation.  The manuscript must either name it
  accurately and state the adaptations or realign the implementation and add
  a standard SAC/DSAC ablation.
- The Lagrange multiplier is updated from immediate sampled cost while the actor
  is penalized by discounted cumulative cost Q.  The constraint unit and limit
  must be made consistent and tested on a synthetic constrained MDP.

### 8. The lower observation must be causal by construction

Removing the stale backward-headway term only from the reward is insufficient
if it remains in the policy observation.  Protocol v4 replaces the legacy slot
with an explicit validity indicator for the exact same-stop forward-arrival
headway.  Decisions without a preceding arrival event are masked from the main
controller.  If follower information is used, it must be a separately named
same-time AVL spatial-gap feature with a source flag; scheduled-trip identity
and a follower's old forward headway are not admissible measurements.

### 9. Terminal actions need an auditable projection contract

The v4 upper action is a bounded smooth launch-shift curve.  For every affected
trip the simulator must retain the unmodified base launch, desired shift,
projected scheduled launch, actual launch, predecessor scheduled launch and
derived target headway.  The following invariants are submission gates:

- scheduled launches are strictly ordered within direction;
- target headway equals the difference of consecutive executable scheduled
  launches to numerical tolerance;
- the action cannot silently revert to the base timetable inside its declared
  horizon;
- any fleet-readiness delay is reported separately from planner projection;
- lower goal reachability and projection saturation rates are reported.

## Protocol-v4 Build Order

1. Add invariant tests for executable target/schedule agreement, fleet-state
   scaling, follower measurement sources, LF/HF wait conservation, passenger
   journey metrics, and fixed vehicle inventory.
2. Replace the planner action with an executable bounded launch-shift curve and
   derive the lower target from scheduled consecutive launches.
3. Replace stale backward reward with forward-event reward plus optional causal
   AVL spacing context.
4. Add per-passenger causal frequency attribution and non-overlapping LF/HF
   wait credit.
5. Add load-weighted holding and journey-time safety outcomes.
6. Enforce fixed-pool terminal circulation and expose denied/late dispatch
   diagnostics.
7. Decouple RNG streams and formalize SAC entropy/Lagrangian units.
8. Add plan-owner LF credit and retain interval assignment only as an explicit
   ablation; verify that one episode's plan credits reconstruct its LF served
   waiting contribution.
9. Run small deterministic contract tests, then a multi-variant 20-seed screen
   on `node001-node006`.
10. Select one method using a pre-registered restricted-wait gate with no-harm
   constraints on journey time, unserved rate, completion and fleet use.
11. Run a fresh 200-episode multi-domain confirmation against fixed headway,
    rule holding, MPC, the closest valid TransitDuet control, and standard SAC.

## Result Status

All legacy paper-main V1 results that select different fixed actions or experts
by known evaluation domain remain oracle engineering bounds, not deployable
method evidence.  Legacy training-tail comparisons remain exploratory.  The
source-fingerprinted frozen-policy v2/v3 matrices are valid diagnostic evidence,
but only a source-identical v4 confirmation on untouched seeds can become the
headline table.

No algorithm source is changed while the 260 protocol-v3 shards are active.
Their shared source fingerprint must remain intact through strict aggregation.
