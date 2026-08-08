# FreqDuet Paper Effect Status

Last updated: 2026-08-08 CST

## Submission Status

**HOLD.** The June paper package is historical evidence from the legacy
service-cost objective. It is not the active submission result.

The corrected V4 crossed experiment showed that the old learned controller can
look competitive under the service-cost composite while producing excessive
holding, readiness delay, and denied trips. That protocol therefore failed the
passenger-journey claim. The locked result and its evidence path are documented
in `protocol_v4_selection_outcome_2026-08-08.md`.

Protocol V5 is the only active candidate. Its primary endpoint is
`restricted_total_journey_horizon_min`; fixed fleet, exact timetable execution,
causal holding feasibility, passenger-weighted holding cost, and normalized
safety endpoints are mandatory parts of the comparison.

## Current Evidence State

### Historical evidence, not submission evidence

- The old paper-main was statistically close to fixed-headway under the legacy
  composite and better than rule-holding/rule-MPC on that same composite.
- The old mechanism and decomposer packages remain useful diagnostics.
- The MBTA/MTA/Halifax caches remain valid external realism evidence within
  their documented data boundaries.

These facts do not establish passenger-journey benefit under the corrected
physical protocol.

### Active V5 evidence

- Source commit: `f2a5ae8e183c48ca2e15295e854913736ca88857`.
- Frozen tag: `freqduet-v5-dev-screen-v1`.
- Development screen: 11 configs x 8 training seeds x 4 frozen evaluation
  seeds, 80 training episodes.
- External comparator screen: fixed-headway, rule-holding, and rule-MPC on the
  same four frozen scenario tapes.
- Untouched confirmation seeds remain sealed until the preregistered V5
  decision is made.

No V5 effect claim is valid until the development matrix is complete, strictly
aggregated, and passed through `decide_freqduet_protocol_v5_screen.py`.

## Claim Gate

The paper can move off hold only if all of the following are true:

1. The V5 source and scenario manifests pass strict provenance checks.
2. The main policy satisfies the physical and safety invariants.
3. The frequency controls and layer-allocation controls meet the locked effect
   and confidence-interval rules, or the paper claim is narrowed accordingly.
4. The selected policy is confirmed on untouched 200-episode seeds and the
   held-out generalization matrix.
5. Learned-versus-external comparisons use the same scenario tapes and the V5
   passenger-journey endpoint.

## Currently Safe Wording

FreqDuet V5 is a preregistered, journey-feasible frequency-separated
hierarchical controller under evaluation. Earlier composite-based results are
reported as historical diagnostics and do not support a current performance
claim.

## Currently Unsafe Wording

Do not state that FreqDuet matches or exceeds fixed-headway under the corrected
passenger-journey protocol, that every frequency module is effective, or that
field benefit has been demonstrated. Those claims require the pending V5
confirmation package.
