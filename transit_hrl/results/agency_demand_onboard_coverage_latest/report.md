# Agency Demand and Onboard-Load Coverage Ledger

This ledger separates observed agency demand from native simulator service metrics.  Claims about real OD/onboard-load/alighting ground truth require external files that expose those fields.

- overall scope: `real_afc_apc_demand_plus_native_service_response`
- supported boundary rows: `4`
- external-missing boundary rows: `3`

## Source Coverage

| source | status | rows | coverage | boundary |
|---|---|---:|---|---|
| public_afc_station_hour | supported | 1000 | stations=41 routes= time_bins=25 | observed station-hour entries; not onboard occupancy, alightings, or OD unless those fields are present |
| public_apc_route_boarding | supported | 1000 | stations= routes=8 time_bins=1000 | observed route boardings; not onboard occupancy, alightings, or OD unless those fields are present |
| gtfs_ride_external | external_missing | 0 | stations= routes= time_bins= | optional external GTFS-ride directory was not supplied |

## Claim Boundaries

| id | evidence item | status | allowed wording | forbidden wording | evidence |
|---|---|---|---|---|---|
| A1 | real_afc_station_hour_demand | supported | real AFC-style station-hour entry demand | real OD or onboard-load ground truth | rows=1000 stations=41 time_bins=25 |
| A2 | real_apc_route_boarding_demand | supported | real APC-style route boarding demand | real onboard occupancy, alighting, or OD ground truth unless columns exist | rows=1000 routes=8 route_time_bins=1000 |
| A3 | native_service_response_wait_alighting_throughput | supported | native public-demand service-response loop improves wait/alighting/throughput | external agency alighting or onboard-load ground-truth improvement | rows=96 seeds=48 board_wait=supported alighted=supported throughput=supported |
| A4 | native_onboard_load_loop | supported | native onboard-load metric is recorded and audited | native onboard-load improvement is supported if CI is inconclusive | onboard_improvement=inconclusive |
| A5 | real_gtfs_ride_board_alight | external_missing | real stop-level board/alight validation when GTFS-ride board_alight is supplied | real alighting ground truth for the current AFC/APC-only cache | board_alight_rows=0 has_alightings=False source_kind=unknown source_verified=False |
| A6 | real_gtfs_ride_onboard_load | external_missing | real onboard-load validation when GTFS-ride load_count/current_load is supplied | real onboard-load ground truth for the current AFC/APC-only cache | board_alight_rows=0 has_onboard_load=False source_kind=unknown source_verified=False |
| A7 | real_gtfs_ride_od | external_missing | real OD validation when rider_trip or origin/destination fields are supplied | real OD ground truth for the current AFC/APC-only cache | rider_trip_rows=0 has_od_fields=False source_kind=unknown source_verified=False |

## Native Service Metrics

- variant: `native_real_freqhrl`
- rows: `96`
- seeds: `48`
- service-response status: `supported`
- onboard improvement status: `inconclusive`
