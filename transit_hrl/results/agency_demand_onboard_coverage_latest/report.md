# Agency Demand and Onboard-Load Coverage Ledger

This ledger separates observed agency demand from native simulator service metrics and external truth-source coverage.  Public MBTA and MTA truth sources close data-availability boundaries only for the fields they expose; they do not by themselves prove native control improvement on those exact files.

- overall scope: `partial`
- supported boundary rows: `6`
- external-missing boundary rows: `3`

## Source Coverage

| source | status | rows | coverage | boundary |
|---|---|---:|---|---|
| public_afc_station_hour | supported | 1000 | stations=41 routes= stops= origins= destinations= time_bins=25 | observed station-hour entries; not onboard occupancy, alightings, or OD unless those fields are present |
| public_apc_route_boarding | supported | 1000 | stations= routes=8 stops= origins= destinations= time_bins=1000 | observed route boardings; not onboard occupancy, alightings, or OD unless those fields are present |
| gtfs_ride_external | external_missing | 0 | stations= routes= stops= origins= destinations= time_bins= | optional external GTFS-ride directory was not supplied |
| mbta_bus_stop_trip_ridership | supported | 1202491 | stations= routes=152 stops=6775 origins= destinations= time_bins= | observed MBTA stop/trip bus averages with boardings, alightings, and onboard load; not OD |
| mta_subway_od_estimate_2024 | supported | 5000 | stations= routes= stops= origins=422 destinations=422 time_bins= | agency-published subway OD estimate from fare data; not observed bus OD or onboard load |

## Claim Boundaries

| id | evidence item | status | allowed wording | forbidden wording | evidence |
|---|---|---|---|---|---|
| A1 | real_afc_station_hour_demand | supported | real AFC-style station-hour entry demand | real OD or onboard-load ground truth | rows=1000 stations=41 time_bins=25 |
| A2 | real_apc_route_boarding_demand | supported | real APC-style route boarding demand | real onboard occupancy, alighting, or OD ground truth unless columns exist | rows=1000 routes=8 route_time_bins=1000 |
| A3 | native_service_response_wait_alighting_throughput | not_supported | native public-demand service-response loop is implemented; strict raw wait/alighting/throughput improvement remains unresolved | external agency alighting or onboard-load ground-truth improvement | rows=96 seeds=48 board_wait=not_supported alighted=not_supported throughput=not_supported projection_contaminated=True |
| A4 | native_onboard_load_loop | supported | native onboard-load metric is recorded and audited | native onboard-load improvement is supported if CI is inconclusive | onboard_improvement=inconclusive |
| A5 | real_gtfs_ride_board_alight | external_missing | real stop-level board/alight validation when GTFS-ride board_alight is supplied | real alighting ground truth for the current AFC/APC-only cache | board_alight_rows=0 has_alightings=False source_kind=unknown source_verified=False |
| A6 | real_gtfs_ride_onboard_load | external_missing | real onboard-load validation when GTFS-ride load_count/current_load is supplied | real onboard-load ground truth for the current AFC/APC-only cache | board_alight_rows=0 has_onboard_load=False source_kind=unknown source_verified=False |
| A7 | real_gtfs_ride_od | external_missing | real OD validation when rider_trip or origin/destination fields are supplied | real OD ground truth for the current AFC/APC-only cache | rider_trip_rows=0 has_od_fields=False source_kind=unknown source_verified=False |
| A8 | real_public_bus_stop_board_alight | supported | real public bus stop/trip boardings and alightings from MBTA | GTFS-ride-native board_alight feed unless supplied separately | rows=1202491 routes=152 stops=6775 total_boardings=987905.5 total_alightings=985893.8 |
| A9 | real_public_bus_stop_onboard_load | supported | real public bus stop/trip onboard load averages from MBTA | onboard-load improvement under Freq-HRL unless linked to a control validation | rows=1202491 mean_load=10.0635 max_load=69.4980 |
| A10 | real_public_subway_od_estimate | supported | real public agency subway OD estimates from MTA | observed individual OD truth or bus OD/onboard load | sample_rows=5000 full_table_rows=116279069 origins=422 destinations=422 od_pairs=4860 |

## Deployment Data Gate

- same-agency native control status: `native_control_outcome_unresolved`
- field-complete data status: `partial_external_truth_source_union`

| gate | status | required for | evidence | boundary |
|---|---|---|---|---|
| afc_station_hour_entries | supported | public demand coverage | rows=1000 stations=41 | entry demand only, not OD/load |
| apc_route_boardings | supported | public route boarding coverage | rows=1000 routes=8 | boardings only unless alight/load/OD columns exist |
| external_truth_board_alight_load_od | supported | public truth-source field coverage | supported_boundaries=3 total=3 | may combine public sources; not necessarily one same-agency control feed |
| verified_gtfs_ride_board_alight_load_od | external_missing | same-agency field-complete Transit validation | agency= board_alight_rows=0 rider_trip_rows=0 | requires verified real agency GTFS-ride or equivalent AVL/APC export |
| native_service_response | not_supported | native performance loop | rows=96 seeds=48 | native simulator metrics, not external ground truth by itself |
| same_agency_field_union | partial_external_truth_source_union | deployment-grade Transit evidence | gtfs_full_fields=False external_truth_supported=True | supported only by one verified field-complete agency feed |
| native_control_linkage | native_control_outcome_unresolved | full native real-demand control validation | native_supported=False gtfs_full_fields=False external_truth_supported=True | full deployment claim requires native control driven by the same field-complete feed |

## Native Service Metrics

- variant: `native_real_freqhrl`
- rows: `96`
- seeds: `48`
- service-response status: `not_supported`
- onboard improvement status: `inconclusive`
