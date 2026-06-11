import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from freq_hrl.experiments.transit.external_transit_truth_validation import (
    run_external_truth_validation,
)


class ExternalTransitTruthValidationTest(unittest.TestCase):
    def _write_mbta_zip(self, path: Path) -> None:
        rows = [
            "GTFS route_id,GTFS direction_id,trip start time,Stop Name,GTFS stop_id,stop sequence,Year,Day Type,Boardings,Alightings,Load,Route/Variant,# of Trip Samples ",
        ]
        for route in range(3):
            for stop in range(5):
                rows.append(
                    f"{route},0,08:{route}{stop}:00,Stop {stop},{100 + stop},{stop},"
                    f"Fall 2025,Wkdy,{1 + route + stop},{0.5 + stop},{3 + route + stop},"
                    f"{route},10"
                )
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(
                "MBTA_Bus_Ridership_by_Trip_Season_Route_Line_and_Stop/"
                "MBTA_Bus_Ridership_by_Trip_Season_Route_Line_and_Stop_Fall_2025.csv",
                "\n".join(rows) + "\n",
            )

    def _write_mta_od(self, path: Path) -> None:
        rows = []
        for origin in range(4):
            for dest in range(4):
                rows.append({
                    "year": "2024",
                    "month": "4",
                    "day_of_week": "Monday",
                    "hour_of_day": str(origin + dest),
                    "timestamp": "2024-04-08T08:00:00.000",
                    "origin_station_complex_id": str(origin),
                    "origin_station_complex_name": f"Origin {origin}",
                    "destination_station_complex_id": str(dest),
                    "destination_station_complex_name": f"Destination {dest}",
                    "estimated_average_ridership": str(1.0 + origin + dest),
                })
        path.write_text(json.dumps(rows), encoding="utf-8")

    def test_public_external_truth_sources_are_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mbta_zip = root / "mbta.zip"
            mta_od = root / "mta_od.json"
            self._write_mbta_zip(mbta_zip)
            self._write_mta_od(mta_od)

            payload = run_external_truth_validation(
                output_dir=root / "out",
                mbta_zip=mbta_zip,
                mta_od_json=mta_od,
                mta_od_total_rows=1000,
                min_mbta_rows=10,
                min_mbta_routes=3,
                min_mbta_stops=5,
                min_mta_od_rows=10,
                min_mta_od_origins=4,
                min_mta_od_destinations=4,
            )

            boundaries = {row["evidence_item"]: row for row in payload["claim_boundaries"]}
            self.assertEqual(payload["summary"]["evidence_scope"], "real_public_board_alight_load_and_estimated_od")
            self.assertEqual(boundaries["real_public_bus_stop_board_alight"]["status"], "supported")
            self.assertEqual(boundaries["real_public_bus_stop_onboard_load"]["status"], "supported")
            self.assertEqual(boundaries["real_public_subway_od_estimate"]["status"], "supported")
            self.assertEqual(payload["source_coverage"][0]["selected_member"].split("_")[-1], "2025.csv")


if __name__ == "__main__":
    unittest.main()
