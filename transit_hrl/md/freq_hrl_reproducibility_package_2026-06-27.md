# Freq-HRL Reproducibility Package

Date: 2026-06-27

This package records the commands required to regenerate the current claim matrix, manuscript tables, figures, and carrier-upgrade artifacts. Raw third-party caches remain ignored; regeneration commands must download or rebuild them explicitly.

| stage | command | output | expected |
| --- | --- | --- | --- |
| unit_tests | PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m unittest discover -s transit_hrl/tests | all local tests | OK |
| claim_matrix | PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.top_journal_unified_matrix --output-dir transit_hrl/results/top_journal_unified_matrix_latest | claims.csv, report.md, summary.json | 9 conservative claims with explicit boundaries |
| external_truth_raw_cache | PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.transit.external_transit_truth_validation --download-missing --mta-od-total-rows 116279069 --output-dir transit_hrl/results/external_transit_truth_validation_latest | public MBTA/MTA derived summaries; raw caches remain ignored | supported public board/alight/load and estimated OD coverage |
| agency_coverage | PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.transit.agency_demand_onboard_coverage --output-dir transit_hrl/results/agency_demand_onboard_coverage_latest | source_coverage.csv, claim_boundaries.csv, deployment_data_gate.csv | field coverage and same-agency native-control boundary ledger |
| baseline_manifest | PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.baseline_ablation_matrix --output-dir transit_hrl/results/baseline_ablation_matrix_latest | paired baseline and ablation checks | frequency-responsibility baselines supported where registered |
| figures | PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.manuscript_figures --output-dir transit_hrl/results/manuscript_figures_latest | SVG/PDF/PNG/TIFF figures and source_data CSVs | five manuscript figures |
| submission_pack | PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.manuscript_submission_pack --output-dir transit_hrl/results/manuscript_submission_pack_latest | conservative submission package | claim tables, methods/SI, data availability |
| theory_appendix | PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.theory_appendix --output-dir transit_hrl/results/freq_hrl_theory_appendix_latest | formal theorem/proposition appendix | sufficient-condition and reporting-boundary statements |
| carrier_upgrade | PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.carrier_upgrade_package --output-dir transit_hrl/results/carrier_upgrade_package_latest | carrier upgrade md and manifests | frozen spec, audit, baseline, data, theory, manuscript, reproducibility package |

## Reproducibility Artifacts

| artifact | kind | status | count | reproduce_stage | commit_policy |
| --- | --- | --- | --- | --- | --- |
| claim_matrix | evidence_table | present |  | claim_matrix | commit compact summary and claim tables |
| figure_source_data | source_data | present | 11 | figures | commit source_data CSVs and lightweight figure drafts |
| scheduler_seed_manifest | seed_ledger | present | 560 | claim_matrix | commit compact seed ledger, not raw scheduler scratch |
| external_transit_raw_cache | ignored_raw_cache | regenerate_with_command |  | external_truth_raw_cache | do not commit raw third-party cache |
| carrier_validation_json | machine_check | present | 1 | carrier_upgrade | commit compact validation JSON |

## Scheduler Seed Ledger

| artifact | role | status | seed_count | first_seed | last_seed | seed_stride | boundary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native_promotion_v47_odshift_wait_first_512seed | C1/C7 native learned-promotion evidence | present | 512 | 31 | 5141 | 10 | scheduler seed ledger; raw shard logs are not committed |
| native_real_demand_service_response_v7_48pair | C2 native real-demand service-response evidence | present | 48 | 31 | 501 | 10 | scheduler seed ledger; raw shard logs are not committed |

## Artifact Policy

Commit compact summaries, claim tables, figure source data, and manuscript figures. Do not commit large raw third-party files, scheduler scratch shards, or generated TIFFs unless required by a journal submission portal.
