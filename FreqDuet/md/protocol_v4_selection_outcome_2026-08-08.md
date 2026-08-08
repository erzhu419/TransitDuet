# FreqDuet Protocol V4 Selection Outcome

Date: 2026-08-08

## Locked status

Protocol V4 is closed as a negative development result. It must not be used as
the paper main method and its old wait/CV improvements must not be described as
evidence that frequency separation is effective.

The strict crossed matrix contains 11 configurations, 20 independent training
seeds, and 8 common evaluation seeds: 220 trained policies and 1,760 frozen
rollouts. The run manifests, source fingerprints, checkpoint identity, scenario
tapes, crossed bootstrap, train-seed sign-flip tests, and Holm correction were
validated before the decision rule was applied.

Frozen aggregate:

`results_freqduet/protocol_v4_selection_ep40_s20_e8_v1/combined_summary_crossed_v2`

## Decision

The locked decision is `frequency_claim_failed`. No V4 configuration was
selected. The no-frequency control had the lowest mean restricted service cost:
its difference from V4 main was -0.64577 with a 95% crossed-bootstrap interval
of [-0.93783, -0.35115].

| Policy | Restricted service cost | Restricted wait (min) | Restricted in-vehicle (min) | Restricted journey (min) | Holding passenger-seconds | Denied trips | Readiness delay (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| V4 main | 1.9597 | 11.8269 | 10.8326 | 22.6595 | 432,710 | 121.98 | 124.10 |
| V4 no-frequency | 1.3140 | 8.0543 | 12.3840 | 20.4383 | 1,859,900 | 219.98 | 1,603.96 |
| External fixed-headway | 0.8739 | 4.5813 | 10.8876 | 15.4689 | 0 | 55.75 | 35.86 |

## Failure diagnosis

The former primary scalar under-priced passenger time on board and did not
price overdue dispatch backlog. It therefore rewarded two non-deployable or
undesirable behaviors:

1. The lower controller could reduce observed waiting and headway CV through
   excessive holding while moving delay into in-vehicle time.
2. The upper controller could compress the timetable globally, while the fixed
   vehicle pool displaced the shortage into denied dispatches and readiness
   delay.

This is an objective and executability failure, not evidence that the
no-frequency controller is a valid paper method. Protocol V5 replaces the
faulty objective/action contracts rather than tuning V4 further.

## Claim boundary

- Retain V4 only as a documented negative result and design motivation.
- Do not merge V4 results with V5 or earlier FreqHRL/TransitDuet results.
- Do not claim that harmonic frequency separation was validated by V4.
- Any new efficacy claim requires the locked V5 development screen and an
  untouched V5 confirmation matrix.
