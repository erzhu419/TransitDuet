# Freq-HRL Reproducibility Package

Date: 2026-06-27

This package records the commands required to regenerate the current claim matrix, manuscript tables, figures, and carrier-upgrade artifacts. Raw third-party caches remain ignored; regeneration commands must download or rebuild them explicitly.

| stage | command | output | expected |
| --- | --- | --- | --- |
| unit_tests | PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m unittest discover -s transit_hrl/tests | all local tests | OK |
| claim_matrix | PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.top_journal_unified_matrix --output-dir transit_hrl/results/top_journal_unified_matrix_latest | claims.csv, report.md, summary.json | 9 conservative claims with explicit boundaries |
| baseline_manifest | PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.baseline_ablation_matrix --output-dir transit_hrl/results/baseline_ablation_matrix_latest | paired baseline and ablation checks | frequency-responsibility baselines supported where registered |
| figures | PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.manuscript_figures --output-dir transit_hrl/results/manuscript_figures_latest | SVG/PDF/PNG/TIFF figures and source_data CSVs | five manuscript figures |
| submission_pack | PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.manuscript_submission_pack --output-dir transit_hrl/results/manuscript_submission_pack_latest | conservative submission package | claim tables, methods/SI, data availability |
| carrier_upgrade | PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.carrier_upgrade_package --output-dir transit_hrl/results/carrier_upgrade_package_latest | carrier upgrade md and manifests | frozen spec, audit, baseline, data, theory, manuscript, reproducibility package |

## Artifact Policy

Commit compact summaries, claim tables, figure source data, and manuscript figures. Do not commit large raw third-party files, scheduler scratch shards, or generated TIFFs unless required by a journal submission portal.
