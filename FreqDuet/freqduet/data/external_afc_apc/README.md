# External AFC/APC Demand-Profile Data

This directory contains small public AFC/APC cache files used only as external
demand-profile evidence for the FreqDuet paper package.

## Files

- `public_afc_mta/hourly_ridership.csv`
  - Source: MTA / New York State Open Data hourly ridership endpoint.
  - URL: `https://data.ny.gov/resource/wujg-7c2s.json`
  - Observation: station-complex hourly entries.
  - Boundary: AFC entries only; not OD geometry, onboard load, alighting, or
    agency field deployment outcomes.

- `public_apc_halifax/route_boardings.csv`
  - Source: Halifax Transit public APC ArcGIS endpoint.
  - URL: `https://services2.arcgis.com/11XBiaBYA9Ep0yNJ/ArcGIS/rest/services/Transit_Automated_Passenger_Counts/FeatureServer/0/query`
  - Observation: route half-hour boardings.
  - Boundary: APC boardings only; not full OD geometry, onboard occupancy,
    alighting, or agency field deployment outcomes.

## Use Boundary

These files were copied into FreqDuet as data evidence only. They must not be
used to import or reuse the separate `transit_hrl` algorithm implementation,
checkpoints, or result claims. Paper wording should say "public AFC/APC
demand-profile evidence" unless a future FreqDuet-specific real-data control
experiment is run.
