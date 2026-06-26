# FreqDuet Paper Claim Scope

Last updated: 2026-06-26 CST

## Terminology Ledger

| Canonical term | Use | Do not use for |
| --- | --- | --- |
| FreqDuet | The frequency-separated hierarchical bus holding controller in this repo. | The separate FreqHRL or TransitHRL projects. |
| paper-main V1 | The current deterministic paper-main aliases for the four canonical domains. | A field-deployed controller. |
| strong fixed-headway baseline | The tuned deterministic fixed-headway comparator in the 60-seed external baseline matrix. | A weak straw-man baseline. |
| closest preserved TransitDuet-family baseline | The no-frequency HIRO control rebuilt from the canonical 60-seed paper ablation matrix. | The unmodified original TransitDuet repository. |
| route/day held-out readiness | A data/protocol audit showing how to build route-family and service-day splits from MBTA/MTA caches. | A completed route/day policy matrix. |
| external realism audit | Public AFC/APC/OD/onboard-load/AVL source-data evidence and claim boundaries. | Same-day field calibration or deployment validation. |

## One-Sentence Argument

In OD-driven bus holding control, FreqDuet shows that frequency-separated
hierarchical control with leakage prevention is mechanistically traceable and
competitive with a strong fixed-headway policy, supported by 60-seed simulation
matrices, external classical baselines, mechanism traces, and public AFC/APC/AVL
realism audits, while same-day field calibration and route/day policy
generalization remain explicit future work.

## Safe Main Claims

1. FreqDuet is statistically tied with a strong fixed-headway baseline under
   the 60-seed, 200-episode external baseline protocol, and significantly
   outperforms rule-holding and rule-MPC.

2. Leakage prevention is the clearest internal mechanism result: the no-leakage
   ablation is consistently and substantially worse in the 60-seed, 200-episode
   ablation matrix.

3. The broad held-out matrix supports robustness to controlled demand-noise,
   OD-profile, and rush-timing perturbations. It does not by itself prove
   multi-route or multi-service-day field generalization.

4. The closest preserved TransitDuet-family baseline is now locked as a
   no-frequency control rebuilt from the canonical 60-seed paper matrix. The
   overall comparison is statistically close, with a rush-shift advantage for
   FreqDuet, so it should be treated as a lineage control rather than a headline
   dominance result.

5. Public MTA/MBTA/Halifax data support demand-profile, OD-estimate,
   onboard-load, route/stop, AVL snapshot, and route/day protocol readiness
   audits. These data strengthen realism evidence but do not establish observed
   real-world operating improvements.

## Unsafe Claims

Do not claim:

- FreqDuet robustly dominates fixed-headway in all regimes.
- Same-day AFC/APC/AVL/OD field calibration is complete.
- MTA Bus Time is APC/onboard-load data.
- The MTA API cache contains FreqHRL paper results.
- Route-family or service-day FreqDuet policy generalization has been run.
- Learned first-stop/terminal launch value control is validated.
- The closest preserved TransitDuet-family baseline is the unmodified original
  TransitDuet repository.

## Claim-Evidence Map

| Claim | Evidence | Status |
| --- | --- | --- |
| FreqDuet matches fixed-headway and beats weaker rule baselines. | `paper_external_classical_v1_ep200_60seed_paired_deltas.csv`. | Supported. |
| Leakage prevention is necessary. | `paper_ablation_v1_ep200_60seed_paired_deltas.csv`; noleakage CI is strongly worse. | Supported. |
| HF residuals align with lower holding/wait responses. | `mechanism_paper_v1_trace_alignment` source data and figures. | Supported as mechanism trace evidence, not field causality. |
| Demand perturbation robustness. | `paper_broad_generalization_v1_ep100_60seed_*`. | Supported for controlled perturbations. |
| Closest TransitDuet-family lineage control. | `transitduet_like_baseline_ep200` rebuilt from the canonical 60-seed matrix. | Supported with conservative interpretation. |
| Route-family and service-day validation. | `route_day_heldout_readiness_v1`. | Protocol supported; policy matrix not yet run. |
| Same-day field calibration/deployment. | No matched same-day AFC/APC/AVL/OD control-loop artifact. | Not supported. |

## Manuscript Wording

Use:

> FreqDuet matched the strong fixed-headway baseline while substantially
> improving over rule-holding and rule-MPC in the 60-seed external-baseline
> matrix. Its strongest internal evidence is the failure of the no-leakage
> ablation, indicating that preventing low-frequency/high-frequency credit
> leakage is necessary for stable performance.

Avoid:

> FreqDuet outperforms fixed-headway across all route and service-day settings.

That stronger sentence needs a completed route/day policy matrix or field
calibration experiment that does not currently exist.
