# FreqDuet Data Availability And Reproducibility Note

Last updated: 2026-06-26 CST

This note is written as a paper-facing Data Availability and reproducibility
draft. It separates generated FreqDuet artifacts, reused public data, local
external caches, and unsupported field-deployment claims.

## Data Availability

The seed-level simulation outputs, paired-delta tables, figure source data,
configuration snapshots, scripts, manifest files, and negative-result appendix
supporting the current FreqDuet paper package are assembled under
`results_freqduet/paper_package/current`. Before submission, this package
should be deposited in a durable repository such as Zenodo, Dryad, Figshare, or
an institutional repository, and the placeholder `[repository DOI]` should be
replaced with the assigned persistent identifier.

Public datasets reused in the realism audits were obtained from the following
sources. MTA hourly station-entry AFC profiles were downloaded from the New
York State Open Data endpoint `https://data.ny.gov/resource/wujg-7c2s.json`.
MTA subway origin-destination ridership estimates were downloaded from
`https://data.ny.gov/resource/jsu2-fbtj.json` and its metadata endpoint
`https://data.ny.gov/api/views/jsu2-fbtj`. Halifax Transit APC route boarding
profiles were downloaded from its public ArcGIS FeatureServer endpoint. MBTA
Fall 2025 bus stop/trip board-alight-load data were obtained from the public
MassGIS/ArcGIS dataset `8daf4a33925a4df59183f860826d29ee`; the large raw file is
kept as a local external cache, while derived calibration-target and
source-coverage tables are packaged in FreqDuet. MBTA static GTFS and live
GTFS-RT VehiclePositions/occupancy snapshots are used only for route/stop and
AVL realism audits.

The MTA Bus Time API cache used by FreqDuet is stored offline under
`data/external_truth_sources/mta_bus_time_api/offline_cache/20260626T144132Z`.
The parsed CSV files contain MTA route, stop, route-stop sequence, and
route-filtered SIRI VehicleMonitoring snapshots. The API key used for download
is not stored in the repository or paper package; manifests record request
parameters with the key redacted. The cache is an external-data audit input for
FreqDuet only. It is not MTA APC/onboard-load data, not a full-day historical
AVL archive, and not imported from FreqHRL results.

No same-day AFC/APC/AVL/OD field-calibration dataset and no observed field
deployment outcome are currently available for FreqDuet. The route/day
held-out readiness tables document how such experiments should be constructed
from the available MTA/MBTA data, but they do not replace a completed policy
matrix or field validation.

## Code Availability

The paper-facing code paths are listed in `paper_manifest.yaml` and copied into
`results_freqduet/paper_package/current/scripts`. The canonical entry points are:

- `scripts/run_freqduet_ablation.py`
- `scripts/run_freqduet_external_baselines.py`
- `scripts/summarize_freqduet_paper_matrix.py`
- `scripts/summarize_freqduet_broad_generalization.py`
- `scripts/compare_freqduet_external_baseline.py`
- `scripts/make_freqduet_decomposer_figures.py`
- `scripts/make_freqduet_mechanism_figures.py`
- `scripts/audit_external_afc_apc_profiles.py`
- `scripts/audit_external_od_onboard_truth.py`
- `scripts/audit_mbta_same_network_calibration.py`
- `scripts/audit_route_day_heldout_readiness.py`
- `scripts/curate_freqduet_paper_panels.py`
- `scripts/build_freqduet_paper_package.py`

The paper package also includes exact config snapshots for the canonical
ablation, broad generalization, paper-main, and trace diagnostics.

## Repository Actions Before Submission

1. Deposit `results_freqduet/paper_package/current` as a versioned release with
   a persistent DOI.

2. Include a README and file manifest from the paper package. The package
   already contains `package_manifest.json`, `paper_manifest.yaml`, curated
   panel manifests, source-data CSVs, scripts, and config snapshots.

3. If the target journal requires raw external data redistribution, verify the
   licence for each public source. Otherwise cite the original public sources
   and deposit only the derived FreqDuet audit tables plus processing scripts.

4. Keep MTA API credentials out of all archives. Only the redacted offline
   cache manifests and parsed CSVs should be deposited.

5. Record software/environment details for the final run environment if the
   journal requests a full computational reproducibility package.

## FAIR And Risk Audit

| Item | Status | Action |
| --- | --- | --- |
| Persistent identifier | Pending | Deposit paper package and replace `[repository DOI]`. |
| File manifest | Present | `package_manifest.json` and curation manifests exist. |
| Figure source data | Present | Copied under package figure `source_data/` directories. |
| Exact configs | Present | Copied under package `configs/`. |
| Scripts | Present | Copied under package `scripts/`. |
| External public data provenance | Present | README files and source coverage CSVs document public endpoints. |
| MTA API key exposure | Controlled | Key is redacted and not written to disk. |
| Large third-party raw MBTA file | Local external cache | Deposit derived tables or cite source; do not silently redistribute if licence is unclear. |
| Same-day field calibration | Missing | State as future work or limitation. |
| Route/day policy matrix | Missing | Readiness protocol exists; do not claim completed validation. |

## Author Check

- Confirm the target journal and preferred repository before submission.
- Confirm whether derived FreqDuet tables may be deposited openly under the
  intended licence.
- Confirm whether the large MBTA raw CSV should be redistributed, cited only,
  or represented by derived tables plus processing scripts.
- Confirm whether a separate Code Availability section is required.
