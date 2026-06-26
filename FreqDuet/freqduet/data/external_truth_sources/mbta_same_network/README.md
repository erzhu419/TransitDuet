# MBTA Same-Network Calibration Pointers

This directory records the small, redistributable pointer layer for the MBTA
same-network calibration audit. The raw public MBTA files are intentionally not
copied into the FreqDuet repository because they are large and already cached
outside the project tree.

## Local Inputs

- APC board/alight/load CSV:
  `/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/open_transit/mbta/ridership/MBTA_Bus_Ridership_by_Trip_Season_Route_Line_and_Stop/MBTA_Bus_Ridership_by_Trip_Season_Route_Line_and_Stop/MBTA_Bus_Ridership_by_Trip_Season_Route_Line_and_Stop_Fall_2025.csv`
- Static GTFS cache:
  `/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/open_transit/mbta/gtfs`
- Live MBTA GTFS-RT VehiclePositions/occupancy snapshot:
  `/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/open_transit/gtfsrt_occupancy_snapshots/mbta_live_vehicle_snapshots.csv`
- Derived full-day MBTA SUMO APC/AVL replay snapshots:
  `/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/sumo_apc_avl_benchmark/sumo_full_day/mbta_all/full/snapshots/avl_snapshots.csv`
  and
  `/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/sumo_apc_avl_benchmark/sumo_full_day/mbta_all/full/snapshots/apc_events.csv`

## Public Source Anchors

- MBTA bus ridership by trip, season, route, line, and stop:
  https://gis.data.mass.gov/datasets/8daf4a33925a4df59183f860826d29ee
- MBTA GTFS documentation and archive pointers:
  https://github.com/mbta/gtfs-documentation
- MBTA transit-performance GTFS/GTFS-RT processing reference:
  https://github.com/mbta/transit-performance
- GTFS Realtime reference:
  https://gtfs.org/documentation/realtime/reference/

## Boundary

The audit supports same-agency APC-to-GTFS route/stop structural matching, a
Route 111 APC load/boarding/alighting calibration target, local MBTA live
GTFS-RT AVL/occupancy snapshot evidence, and derived full-day MBTA SUMO APC/AVL
replay evidence. It still does not include a Fall 2025 same-day historical AVL
archive matched to the APC season, route-level OD, or observed FreqDuet field
deployment result.
