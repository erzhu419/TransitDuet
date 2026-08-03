# Protocol v4 External Baseline Contract

Date: 2026-08-03

## Reason for replacement

Historical external-baseline runs are not valid protocol-v4 comparisons. They
expanded a nominal seed as `seed * 1000003 + episode`, averaged many episodes,
and applied a target headway only after a row in the original timetable became
eligible. A frozen learned policy instead receives each evaluation seed
directly, and v4 defines the upper action through an executable launch curve.

## Locked comparison contract

1. `--direct-scenario-seeds` maps every requested seed directly to
   `env.scenario_seed` and requires exactly one episode per job.
2. The external baseline and every learned training seed must report the same
   `scenario_tape_id` for an evaluation seed.
3. Constant-headway and rule schedules are written as recursive executable
   launch sequences before simulation. Consecutive projected launches equal
   the requested target; the controller cannot rely on the old timetable row
   as an unreported lower bound.
4. Every policy uses the same fixed physical inventory (`N_fleet=12` in the
   frozen reference), deployable APC/AVL observation contract, forward-event
   headway reward, demand model, objective weights, and horizon accounting.
5. Each run records the resolved config lineage hash, the common core-source
   hash, a separate external-evaluator hash, physical-fleet checks, exact
   projection mode, and immutable scenario tape identifier.
6. Aggregation rejects missing manifests, mixed source hashes, non-exact
   schedules, non-conserved fleets, observation-contract drift, or LF/HF share
   errors.

The comparison script reports learned-minus-baseline paired deltas, crossed
bootstrap confidence intervals over training and evaluation seeds, paired
effect sizes, train-seed sign-flip tests, and Holm correction across external
methods. Negative deltas are improvements for cost metrics.

Passenger safety endpoints are mandatory rather than optional columns:
observed/restricted in-vehicle time, observed/restricted total journey time,
vehicle holding seconds, passenger holding seconds, distinct denied trips, and
fleet-readiness delay. Each bootstrap draw shares one evaluation-seed resample
across all sampled training seeds. Raw waiting-only comparison tables are not
eligible for the paper package.

## Interpretation

`fixed_headway` is the strong operational baseline. `rule_holding` and
`rule_mpc` are classical mechanism references and must not be described as
stronger than their measured tuning supports. Any MPC tuning must use a seed
partition disjoint from the frozen policy-selection and confirmation tapes.
