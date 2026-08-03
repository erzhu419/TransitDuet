# FreqDuet Protocol v4 Method Contract

Date: 2026-08-04

## Decision

Protocol v3 remains a historical diagnostic run. It cannot be the final paper
protocol because its learned timetable curve was not always the executed curve,
its fleet budget was not a conserved vehicle inventory, and parts of the lower
observation/reward path used information unavailable to a deployable controller.

The submission candidate is rebuilt under
`F_freqduet_protocol_v4_main_hiro.yaml`. No v3 numerical result may be relabeled
as v4 evidence. The v4 method must be selected and confirmed on fresh, disjoint
seeds.

## Non-negotiable invariants

1. **Executable upper action.** `exact_headway_curve` recursively writes the
   future terminal launches whose consecutive gaps equal the commanded headway
   curve. Independent per-trip clipping, terminal bias, and headway-floor
   overrides are not part of this action semantics.
2. **Conserved fleet.** Exactly `N_fleet` physical vehicles exist. A dispatch
   waits when the required terminal has no vehicle; the simulator cannot create
   an extra bus to satisfy a timetable row.
3. **Markov fleet state.** The upper policy observes causal terminal readiness,
   opposite-terminal readiness, inbound vehicle mass, and overdue-dispatch
   pressure from AVL and the executable timetable.
4. **Deployable sensing.** The lower policy uses exact forward arrival events,
   current APC boardings/left-behind counts, current dwell/load/capacity, and
   causal historical demand priors. Latent generated arrivals, a future bus,
   downstream queues, and stale backward headways are forbidden.
5. **Immutable frequency ownership.** Every passenger receives LF/HF shares at
   arrival, before the causal demand tracker consumes that arrival. Shares are
   immutable and sum to one through boarding and end-of-horizon accounting.
6. **Disjoint credit.** The upper interval receives frozen LF queue exposure;
   the lower receives frozen HF wait/holding effects. The old extra upper LF
   wait shaping is zero in v4, so LF is not counted twice and HF does not leak
   through an all-frequency interval cost.
7. **Explicit optimization units.** Continuous entropy is measured in a
   dimensionless unit-action coordinate. Entropy temperature starts inside its
   configured bounds and the optimized log parameter itself is bounded. The
   Lagrange threshold is a per-decision cost-rate constraint under the replay
   approximation to normalized discounted occupancy; the actor uses the
   corresponding cost continuation value.
8. **Independent stochastic components.** Network initialization, upper/lower
   exploration, upper/lower replay, selectors, TPC, reachability, and fleet
   sampling use named streams derived from the training seed. Adding an
   ablation to one component cannot consume another component's RNG stream.
9. **Exact training resume.** A v4 resume restores online and target networks,
   all optimizers, replay contents and samplers, entropy/Lagrange state,
   adaptive deployment state, and named/global RNG states. Deployment-only
   weights are rejected as a training resume.
10. **Method naming.** The implemented optimizer is called pessimistic ensemble
    SAC (with a Lagrangian lower constraint). Historical `RESAC*` Python class
    names remain for checkpoint compatibility, but the paper must not claim an
    equation-by-equation reproduction of a named RE-SAC publication.

## Machine-enforced configuration

Run:

```bash
PYTHONPATH=. python3 scripts/validate_freqduet_protocol_v4_configs.py
```

The validator rejects legacy fleet, observation, reward, frequency ownership,
temperature, randomness, TPC-density, timetable, and checkpoint contracts.
It also compares each resolved ablation with the canonical main configuration
and rejects undeclared secondary changes, so a mechanism row cannot silently
change learning rates, network size, or another control component.
Diagnostics and frozen-evaluation manifests record the resolved-config hash,
randomness fingerprint, observation-ledger hash, scenario tape, policy digest,
physical fleet statistics, and projection mode.

## Locked selection axes

The initial v4 screen uses single-axis variants only:

| Config suffix | Scientific question |
| --- | --- |
| `main` | Full causal LF/HF ownership with discrete holding |
| `csac` | Does the 10-critic pessimistic ensemble improve on twin-min constrained SAC? |
| `nofreq` | Does any frequency-aware control help? |
| `rawhistory` | Does harmonic decomposition help beyond the same causal raw history? |
| `allfreq` | Does assigning all frequency state to both levels hurt specialization? |
| `nopromotion` | Is persistent HF-to-LF promotion useful? |
| `noleakage` | Is the explicit cross-timescale leakage regularizer useful? |
| `nodriftfb` | Is causal lower drift feedback to the upper controller useful? |
| `noprior` | Does the historical harmonic prior reduce single-day variance? |
| `continuous_holding` | Is the discrete holding action set important? |
| `nolowercontext` | Does the causal load/queue/shock/slack context improve lower control? |

All rows retain the exact timetable, fixed vehicle inventory, deployable
APC/AVL observation, independent RNG streams, frozen scenario-tape evaluation,
and exact checkpoint contracts. `nofreq` and `rawhistory` cannot use LF/HF
passenger ownership by definition; their upper interval therefore receives the
same all-passenger wait quantity without a second reward-attribution term.

## Evidence sequence

1. Contract tests and deterministic short episodes.
2. Fresh-seed v4 selection screen: optimizer contract, action representation,
   promotion/leakage, and frequency-allocation ablations.
3. Lock one configuration before looking at confirmation seeds.
4. Paired frozen evaluation against the same physical fixed-headway, rule
   holding, rule-MPC, and closest TransitDuet lineage baselines.
5. Untouched-seed 200-episode confirmation, broad held-out domains, route/day
   profiles, mechanism audits, and negative-results appendix.

No paper table is updated until stages 2-4 establish that the rebuilt method is
effective under the corrected physical and causal protocol.
