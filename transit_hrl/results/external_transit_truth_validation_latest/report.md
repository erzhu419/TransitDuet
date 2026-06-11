# External Transit Truth Source Validation

This ledger validates public external truth sources for Transit data coverage. It does not by itself prove a native Freq-HRL control improvement on those exact files.

- evidence scope: `real_public_board_alight_load_and_estimated_od`
- supported boundaries: `3`

## Sources

| source | status | rows | coverage | boundary |
|---|---|---:|---|---|
| mbta_bus_stop_trip_ridership | supported | 1202491 | routes=152 stops=6775 origins= destinations= | observed MBTA stop/trip bus averages with boardings, alightings, and onboard load; not OD |
| mta_subway_od_estimate_2024 | supported | 5000 | routes= stops= origins=422 destinations=422 | agency-published subway OD estimate from fare data; not observed bus OD or onboard load |

## Claim Boundaries

| id | evidence item | status | allowed wording | forbidden wording | evidence |
|---|---|---|---|---|---|
| E1 | real_public_bus_stop_board_alight | supported | real public bus stop/trip boardings and alightings from MBTA | GTFS-ride-native board_alight feed unless supplied separately | rows=1202491 routes=152 stops=6775 total_boardings=987905.5 total_alightings=985893.8 |
| E2 | real_public_bus_stop_onboard_load | supported | real public bus stop/trip onboard load averages from MBTA | onboard-load improvement under Freq-HRL unless linked to a control validation | rows=1202491 mean_load=10.0635 max_load=69.4980 |
| E3 | real_public_subway_od_estimate | supported | real public agency subway OD estimates from MTA | observed individual OD truth or bus OD/onboard load | sample_rows=5000 full_table_rows=116279069 origins=422 destinations=422 od_pairs=4860 |
