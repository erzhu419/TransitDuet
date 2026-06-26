# External OD And Onboard-Load Truth Sources

This directory stores small public truth-source metadata and samples used by
FreqDuet's realism audit.

## Included Data

- `mta_subway_od/mta_subway_od_2024_sample_5000.json`
  - Source: MTA Subway Origin-Destination Ridership Estimate: 2024.
  - API: `https://data.ny.gov/resource/jsu2-fbtj.json`
  - Observation: agency-published OD estimates aggregated by month, day of
    week, hour, origin station complex, and destination station complex.
  - Boundary: fare-derived subway OD estimate, not observed individual AFC OD
    truth, bus OD, onboard load, or FreqDuet field deployment evidence.

- `mta_subway_od/mta_subway_od_2024_count.json`
  - Source: same endpoint with `$select=count(*)`.
  - Purpose: records the full published table row count used by the audit.

- `mta_subway_od/mta_subway_od_2024_metadata.json`
  - Source: `https://data.ny.gov/api/views/jsu2-fbtj`.
  - Purpose: documents official dataset metadata and column schema.

- `mta_bus_time_api/offline_cache/20260626T144132Z/parsed/*.csv`
  - Source: MTA Bus Time OneBusAway discovery API and SIRI
    VehicleMonitoring API.
  - API roots:
    `https://bustime-classic.mta.info/api/where/` and
    `https://bustime-classic.mta.info/api/siri/vehicle-monitoring.json`.
  - Observation: offline MTA bus route, stop, route-stop sequence, and
    route-filtered vehicle snapshot data.
  - Boundary: MTA Bus Time API cache for FreqDuet external-data audit only.
    This is not MTA APC/onboard-load data, not a full-day historical AVL
    archive, not a FreqDuet field deployment result, and not FreqHRL paper
    result data.

## External Large Data

The MBTA bus stop/trip ridership file is intentionally not committed because it
is large. The audit script reads it from the local external cache by default:

`/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/open_transit/mbta/ridership/MBTA_Bus_Ridership_by_Trip_Season_Route_Line_and_Stop/MBTA_Bus_Ridership_by_Trip_Season_Route_Line_and_Stop/MBTA_Bus_Ridership_by_Trip_Season_Route_Line_and_Stop_Fall_2025.csv`

Source: `https://hub.arcgis.com/datasets/8daf4a33925a4df59183f860826d29ee`.

Observation: public MBTA bus stop/trip averages with boardings, alightings, and
onboard load.

Boundary: this supports onboard-load and board/alight calibration targets. It
does not supply OD geometry and is not a matched FreqDuet control deployment.
