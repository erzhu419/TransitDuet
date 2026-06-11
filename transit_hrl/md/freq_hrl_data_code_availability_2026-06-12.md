# Data And Code Availability Draft

Date: 2026-06-12

## Data Availability

The processed evidence tables, validation summaries, claim matrices, and source-coverage ledgers generated in this study are available in the repository under `transit_hrl/results/`. The public AFC/APC demand traces used by the native Transit validation are stored under `transit_hrl/data/public_afc_mta/` and `transit_hrl/data/public_apc_halifax/`. Public external Transit truth-source coverage was derived from the MBTA Bus Ridership by Trip, Season, Route, Line, and Stop dataset and the MTA Subway Origin-Destination Ridership Estimate 2024 dataset; the derived summaries are committed under `transit_hrl/results/external_transit_truth_validation_latest/`, while raw downloaded caches are ignored under `transit_hrl/data/public_mbta_bus_ridership_raw/` and `transit_hrl/data/public_mta_od_raw/` to avoid committing large third-party files. The LOBSTER/NASDAQ TotalView-ITCH sample-derived replay summaries are available under `transit_hrl/results/order_book_lobster_venue_grade_multisymbol/`; access to any full raw proprietary exchange feed remains governed by the original data provider.

## Code Availability

The code used to generate the Freq-HRL experiments, evidence matrices, external data-source ledgers, and manuscript submission package is available in the same repository under `transit_hrl/`. The external Transit source ledger can be regenerated with `python3 -m freq_hrl.experiments.transit.external_transit_truth_validation`, the agency coverage ledger with `python3 -m freq_hrl.experiments.transit.agency_demand_onboard_coverage`, and the unified claim matrix with `python3 -m freq_hrl.experiments.top_journal_unified_matrix`.

## Dataset Citations And Source URLs

- MBTA Bus Ridership by Trip, Season, Route, Line, and Stop: https://mbta-massdot.opendata.arcgis.com/datasets/8daf4a33925a4df59183f860826d29ee
- MTA Subway Origin-Destination Ridership Estimate 2024: https://data.ny.gov/Transportation/MTA-Subway-Origin-Destination-Ridership-Estimate-2/jsu2-fbtj
- GTFS-ride specification for optional native-format replication: https://gtfsride.org/specification
- LOBSTER sample data and NASDAQ TotalView-ITCH semantics should be cited according to the data provider's required citation terms in the final manuscript.

## Missing Information / Risk Flags

- Add final repository URL, release tag, and DOI if the submission requires archival code deposit.
- Confirm whether the final journal requires source-data files for each figure panel.
- Do not describe ignored raw MBTA/MTA caches as newly generated data; they are reused public third-party data.
- Do not imply public GTFS-ride native feed availability unless such a feed is added later.

## Chinese Check

- 需要投稿前补最终代码仓库链接、release tag、可能的 Zenodo DOI。
- MBTA/MTA 是公开第三方数据源；当前提交的是派生 summary，不提交大 raw cache。
- GTFS-ride 是可复现实验接口和标准钩子，不是当前已经拿到的真实 feed。
