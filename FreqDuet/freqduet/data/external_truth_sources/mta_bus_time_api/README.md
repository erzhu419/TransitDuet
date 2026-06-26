# MTA Bus Time API Offline Cache

This directory stores a FreqDuet-only offline cache downloaded from the MTA Bus
Time API. It is external data for FreqDuet realism and calibration audits; it is
not imported from, derived from, or mixed with any FreqHRL paper result,
checkpoint, or experiment output.

## Snapshot

- Snapshot id: `20260626T144132Z`
- Agencies: `MTA NYCT`, `MTABC`
- Static OBA discovery cache:
  - routes: 378
  - unique stops: 13,585
  - route-stop sequence rows: 22,730
  - failed route downloads: 0
- Route-filtered SIRI VehicleMonitoring snapshots:
  - route snapshots: 8
  - vehicle rows: 144

## Offline Files

The parsed CSV files under
`offline_cache/20260626T144132Z/parsed/` are the canonical offline inputs:

- `mta_bus_time_agencies.csv`
- `mta_bus_time_routes.csv`
- `mta_bus_time_stops.csv`
- `mta_bus_time_route_stop_sequences.csv`
- `mta_bus_time_vehicle_snapshot_meta.csv`
- `mta_bus_time_vehicle_snapshots.csv`
- `mta_bus_time_source_coverage.csv`
- `mta_bus_time_claim_boundaries.csv`

Raw JSON responses are kept locally under the same snapshot's `raw/` directory
for debugging and re-parsing, but are ignored by git to avoid unnecessary
repository bloat. The API key is not written to disk; manifests store request
parameters with `key=<redacted>`.

## Boundaries

Allowed claim: offline MTA Bus Time route, stop, route-stop sequence, and
route-filtered vehicle snapshot data are available for FreqDuet external-data
audits and offline replay/realism checks.

Forbidden claim: this is not MTA APC/onboard-load data, not a full-day
historical AVL archive, not a FreqDuet field deployment result, and not FreqHRL
paper result data.
