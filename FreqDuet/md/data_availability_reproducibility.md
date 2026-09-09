# FreqDuet Data Availability And Reproducibility Note

Last updated: 2026-09-10 CST

This note is written as a paper-facing Data Availability and reproducibility
draft. It separates generated FreqDuet artifacts, reused public data, local
external caches, and unsupported field-deployment claims.

## Data Availability

The small immutable V8 confirmation, V9 long-training, and V9 external-baseline
artifacts supporting the current result are tracked under
`freqduet/paper_evidence/protocol_v6/current_best`. Running
`scripts/build_freqduet_protocol_v6_evidence_package.py` validates their
protocol, source/scenario identity, common-random-number status, artifact
bindings, and config fingerprints, then assembles the paper-facing draft under
`results_freqduet/paper_package/protocol_v6_current_best`.

That draft package records `submission_ready: false`: V8 independently confirms
the 40-episode headway-regularity effect with passenger-journey no-harm, while
V9 fails the registered 200-episode long-training headway gate. The older
`results_freqduet/paper_package/current` directory is the historical V1
composite package and is not current effect evidence. A final release should be
deposited in a durable repository and assigned a persistent identifier only
after the manuscript scope and final figure package are fixed.

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

The paper-facing code paths and source commits are listed in
`paper_manifest.yaml`. The canonical Protocol V6 evidence entry points are:

- `scripts/run_freqduet_protocol_v2_matrix.py`
- `scripts/decide_freqduet_protocol_v6_screen.py`
- `scripts/run_freqduet_external_baselines.py`
- `scripts/compare_freqduet_external_frozen.py`
- `scripts/build_freqduet_protocol_v6_evidence_package.py`

The current builder includes the exact verified config inheritance chains for
the confirmed controller and matched no-guard comparator. Historical
mechanism, figure, realism-audit, and composite-package scripts remain listed
in the manifest, but their outputs are not silently promoted into the current
Protocol V6 effect tables.

## Repository Actions Before Submission

1. Resolve the manuscript-level decision created by the failed V9
   long-training gate, then freeze the final claims and package version.

2. Assemble the final figure panels from Protocol V6 source tables and add the
   final environment specification; the current evidence bundle already
   contains a README, file manifest, source CSV/JSON files, and exact configs.

3. Deposit the frozen release in a durable repository and replace
   `[repository DOI]` with its persistent identifier.

4. If the target journal requires raw external data redistribution, verify the
   licence for each public source. Otherwise cite the original public sources
   and deposit only the derived FreqDuet audit tables plus processing scripts.

5. Keep MTA API credentials out of all archives. Only the redacted offline
   cache manifests and parsed CSVs should be deposited.

## FAIR And Risk Audit

| Item | Status | Action |
| --- | --- | --- |
| Persistent identifier | Pending | Deposit only after the final manuscript scope is frozen. |
| File manifest | Present | The Protocol V6 builder writes `package_manifest.json`. |
| Current result source data | Present | V8/V9 seed-level CSV and decision JSON artifacts are tracked and packaged. |
| Current result figure source data | Present | Figures 2 and 3 use the normalized V8/V9 tables and include editable exports and QA notes. |
| Complete final figure package | Pending | Rebind or redraw the method, mechanism, and realism panels to the final V6 scope. |
| Exact configs | Present | The builder verifies and copies the relevant inheritance chains. |
| Scripts | Source-bound | Exact source commits and canonical entry points are recorded; final release archive is pending. |
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
- Confirm whether the transparent V8-positive/V9-negative result scope is
  suitable for the selected journal before labelling any package final.
