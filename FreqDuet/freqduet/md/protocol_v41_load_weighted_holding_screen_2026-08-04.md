# Protocol V4.1 Load-weighted Holding Screen

Date: 2026-08-04

## Question

Does explicitly charging the lower controller for causal APC occupancy-weighted
holding reduce onboard passenger externality without sacrificing service?

## Locked variants

The reference is a fresh rerun of `F_freqduet_protocol_v4_main_hiro` under the
same source snapshot. Four candidates change only
`lower.load_weighted_holding.reward_weight`: `0.01`, `0.03`, `0.06`, and `0.10`.
The normalized penalty is
`weight * (holding_s / 45 s) * clipped_occupancy_ratio`, with occupancy clipped
to `[0,1]` and read from the action-time deployable APC observation.

The screen reuses the V4 *selection* train/eval seeds and 40-episode budget. It
does not inspect or consume untouched confirmation seeds. A source-identical
zero-weight reference is mandatory because adding the implementation changes
the source fingerprint even when the feature is disabled.

## Promotion rule

A positive weight is eligible only when all protocol-V4 physical/causal/tape
checks pass and its candidate-minus-zero-weight crossed-bootstrap intervals
satisfy:

- restricted service-cost CI upper <= `+0.01`;
- unserved-rate CI upper <= `+0.005`;
- launch/completion-rate CI lower >= `-0.005`;
- headway-CV CI upper <= `+0.02` and fleet-overshoot CI upper <= `0`;
- restricted in-vehicle and total-journey CI upper <= `+0.5 min`;
- denied-trip CI upper <= `+1` and mean readiness-delay CI upper <= `+15 s`.

Among eligible positive weights, choose the one with the lowest mean restricted
total journey time; break a difference smaller than `0.05 min` by lower
passenger holding seconds, then by the smaller reward weight. If no positive
weight is eligible, retain the zero-weight controller and report the unresolved
onboard-externality limitation rather than silently weakening a gate.

This screen is run only after the original V4 frequency-allocation screen has
not rejected the parent architecture. If that screen selects a different
single-axis architecture, recreate these four variants from that selected
parent before promotion.
