# External AFC/APC Demand-Profile Data

This directory contains small public AFC/APC cache files used only as external
demand-profile evidence for the FreqDuet paper package.

The two original CSV files are bounded 1000-row API caches. They must not be
aggregated directly because both contain incomplete pagination fragments. The
paper figure uses `balanced_profile_cache_v1/`, produced offline by
`scripts/derive_freqduet_external_profile_balanced_cache.py`.

## Files

- `public_afc_mta/hourly_ridership.csv`
  - Source: MTA / New York State Open Data hourly ridership endpoint.
  - URL: `https://data.ny.gov/resource/wujg-7c2s.json`
  - Observation: station-complex hourly entries.
  - Boundary: AFC entries only; not OD geometry, onboard load, alighting, or
    agency field deployment outcomes.
  - Completeness: bounded cache, not a full network extract.

- `public_apc_halifax/route_boardings.csv`
  - Source: Halifax Transit public APC ArcGIS endpoint.
  - URL: `https://services2.arcgis.com/11XBiaBYA9Ep0yNJ/ArcGIS/rest/services/Transit_Automated_Passenger_Counts/FeatureServer/0/query`
  - Observation: route half-hour boardings.
  - Boundary: APC boardings only; not full OD geometry, onboard occupancy,
    alighting, or agency field deployment outcomes.
  - Completeness: bounded cache; Route 136 is incomplete.

- `balanced_profile_cache_v1/mta_complete_station_day_2024-10-01.csv`
  - Derived subset: 39 station complexes with exactly one row for every hour of
    2024-10-01.

- `balanced_profile_cache_v1/halifax_complete_route_days_2026-01-01_2026-01-07.csv`
  - Derived subset: 7 complete routes and 37 route-days; incomplete Route 136
    is excluded.

- `balanced_profile_cache_v1/derivation_manifest.json`
  - Records source files, inclusion rules, selected rows, and excluded rows.

## Use Boundary

These files were copied into FreqDuet as data evidence only. They must not be
used to import or reuse the separate `transit_hrl` algorithm implementation,
checkpoints, or result claims. Paper wording should say "public AFC/APC
demand-profile evidence" unless a future FreqDuet-specific real-data control
experiment is run.
