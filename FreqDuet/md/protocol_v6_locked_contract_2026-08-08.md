# FreqDuet Protocol V6 Locked Contract

Date: 2026-08-08

## Status

Protocol V6 (`freqduet-eval-v6`) is the post-V5 engineering candidate. It is
**pending engineering screening**. The source and protocol may be tested, but
no current V6 observation is eligible for an effectiveness, frequency-
allocation, or baseline-superiority claim.

Submission remains fail-closed. A development decision can nominate one
candidate for untouched-seed testing; it can never make the paper
submission-ready. Only a successful independent confirmation decision may do
that, after the paper manifest is separately and explicitly updated.

The frozen implementation source is commit
`95f406a49fabd075014ee6eeb33d3309d2f0c475`. The preregistered development
screen is `protocol_v6_engineering_ep40_s4_e4_v2`: 12 locked configs, four
training seeds (`503,521,541,557`), four direct evaluation seeds
(`41011,41017,41023,41039`), and 40 training episodes.

The `protocol_v6_engineering_ep40_s4_e4_v1` scheduler attempt (`t72782` to
`t72829`) is invalid transport evidence: 30 tasks failed and 18 were cancelled
before training because the isolated remote environment had no `git` binary.
No artifact from that run name may enter a V6 matrix.

## Locked implementation repairs

### 1. Executable timetable and phase conservation

- A new upper decision may materialize future terminal departures once.
- Reuse of the same plan must call the read-only cached-plan path and must not
  rewrite any future trip.
- `rolling_zero_sum_delta_v6` closes each budget block at the first departure
  at or after the configured replan window. The projected headway deltas in
  every closed block must sum to zero within numerical tolerance.
- A new receding horizon must not inherit hidden endpoint phase from a previous
  horizon. Projection diagnostics are recorded only for actual upper decisions.
- Vehicle readiness delay is execution error, not a new lower target or a
  silent rewrite of the commanded timetable.

### 2. Causal demand and APC observations

- Passenger counts sampled for a demand bin are assigned within-bin arrival
  times and become visible only after their own arrival time.
- A controller at the start of a bin cannot observe passengers arriving later
  in that bin.
- APC boardings are accumulated for every simulation second and exposed to the
  frequency tracker only when the complete observation bin closes.
- Historical demand priors remain permitted, but no future realized demand,
  future bus event, or incomplete-bin count may enter a decision.

### 3. Lower holding evidence and execution metrics

- The V6 causal holding guard uses the matched predecessor's departure event
  available before the action (`pre_action_departure_v6`), rather than an
  arrival gap that ignores service and prior holding.
- Commanded holding and realized holding are recorded separately. Paper service
  cost and passenger exposure must use realized seconds; horizon truncation
  cannot be counted as executed control.
- Actual terminal dispatch gaps and planned-versus-executed departure errors
  are explicit diagnostics.
- Fleet-denial retries are reported as retry trip-seconds as well as any unique
  denied-trip statistic; the two quantities must not be conflated.

### 4. Episode-global upper credit

- Additive interval wait, onboard, backlog, headway, and fleet components use
  episode-level denominators.
- Component clipping occurs after all interval components for the episode are
  summed, then the same scale is returned to the contributing intervals.
- Splitting one physical outcome into more bookkeeping intervals must not
  change total credit.
- LF and HF ownership remains disjoint: low-frequency planning cost belongs to
  the upper layer, while local high-frequency correction and holding effects
  belong to the lower layer.

## Alignment with the design route

The V6 repair preserves the scientific direction in `md/GPT.md`:

| Design requirement | V6 contract alignment |
| --- | --- |
| LF demand drives high-level planning | The upper plan remains a smooth executable headway/timetable object with a rolling phase budget. |
| HF residual drives low-level control | The lower controller receives causal local evidence and acts through station holding. |
| No future leakage | Passenger release and APC frequency updates are delayed until observations physically exist. |
| Low-level action must not rewrite the timetable | Departure-based holding evidence, leakage diagnostics, realized holding, and upper phase conservation make the boundary observable. |
| Passenger waiting is the primary service objective | `restricted_total_journey_horizon_min` is primary, with wait, in-vehicle time, unserved demand, fleet feasibility, and holding as safety outcomes. |
| Frequency allocation must beat feature addition | NoFreq, RawHistory, AllFreq, layer-only, Swapped, and mechanism controls remain required evidence. |

Relative to `md/dev_manual.md`, V6 retains the Phase 0 logging and Phase 1
causal decomposition evidence, the Phase 2 LF/HF state allocation, and the
Phase 3 executable target-headway/timetable semantics. The repairs also make
the terminal execution boundary measurable. They do not by themselves finish
the strongest Phase 4 claim of field-validated actual terminal launch and
first-stop dispatch control.

## Fail-closed confirmation chain

A package is submission-ready only when every condition below holds:

1. paper status is exactly `ready_protocol_v6_confirmed`;
2. active protocol is exactly `freqduet-eval-v6` and the paper manifest is
   non-historical;
3. confirmation status and stage are both confirmation, with a full source
   commit bound to the active source commit;
4. the decision file SHA256 matches the paper manifest;
5. the decision uses
   `freqduet-protocol-v6-staged-decision-v1`, records stage `confirmation`,
   status `confirmation_supported`, and a non-empty `selected_config`;
6. a top-level `candidate_config` is absent: development/candidate decisions
   are never submission decisions;
7. the decision's matrix-manifest SHA256 equals the SHA256 of the confirmation
   source manifest named by the paper manifest;
8. the matrix manifest is `freqduet-matrix-manifest-v2`, protocol V6, stage
   confirmation, and records `strict_complete=true`,
   `run_manifests_verified=true`, `common_random_numbers_verified=true`, and
   `independent_confirmation=true`;
9. `frozen_per_eval.csv`, `frozen_summary.csv`, and
   `frozen_paired_deltas.csv` all exist and match their matrix SHA256 records.

The historical override remains limited to an explicitly historical manifest
with the locked historical hold status. It permits reproduction of an old
package only; it cannot turn a hold, V5 decision, development decision, or
candidate decision into a ready package.

## Evidence gates still outstanding

V6 must pass these gates in order:

1. **Engineering invariants.** Unit tests plus deterministic episodes must show
   cached-plan immutability, rolling phase conservation, causal passenger/APC
   timing, departure-evidence correctness, realized-holding accounting, and
   partition-invariant episode credit on every checked rollout.
2. **Development screen.** Run the locked V6 matrix on development seeds. A
   failure of physical, causal, or provenance invariants invalidates the run;
   it is not a tunable performance loss.
3. **Frequency and allocation evidence.** The selected method must survive
   NoFreq, RawHistory, AllFreq, LF/HF layer-only, Swapped, optimizer, budget,
   guard, load-cost, and credit controls under the locked primary and safety
   metrics. A simpler or non-frequency control winning triggers redesign or a
   narrower claim.
4. **Selection freeze.** Select at most one candidate, then freeze source,
   config, train length, checkpoint, metric family, and decision thresholds
   before accessing confirmation seeds.
5. **Independent confirmation.** Use disjoint untouched train and evaluation
   seeds with complete crossed rollouts, common random numbers, paired deltas,
   crossed bootstrap intervals, and the declared multiple-testing treatment.
6. **External baselines and generalization.** Rerun physically matched
   fixed-headway, rule holding, rule MPC, and closest TransitDuet lineage
   baselines under V6 scenario semantics; then test held-out demand noise, OD,
   rush, route/day, dwell, fleet, and service perturbations.
7. **Mechanism evidence.** Produce the decomposer, upper-HF power, lower-LF
   drift, HF-residual-to-holding lag, promotion/leakage, actual-dispatch error,
   and commanded-versus-realized holding diagnostics required by the manual.
8. **Paper promotion.** Only after gates 1-7 pass may a confirmation decision
   be bound into a non-historical paper manifest and the submission gate be
   exercised positively.

## Claim boundary

At contract lock, the only defensible statement is:

> V6 repairs the identified V5 execution and evidence-integrity defects and is
> ready for engineering screening under a locked causal protocol.

It is currently forbidden to state that V6 is effective, validates the
frequency-separation hypothesis, matches fixed-headway, or outperforms any
baseline. Those are empirical conclusions reserved for the locked development
and independent confirmation sequence above.
