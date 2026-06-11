import json
import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.transit.agency_demand_onboard_coverage import run_coverage


class AgencyDemandOnboardCoverageTest(unittest.TestCase):
    def _write_afc(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            f.write("transit_timestamp,station_complex_id,station_complex,ridership\n")
            for station in range(3):
                for hour in range(6):
                    f.write(
                        f"2024-10-01T{hour:02d}:00:00.000,"
                        f"{station},Station {station},{10 + station + hour}\n"
                    )

    def _write_apc(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            f.write("OBJECTID,Route_Number,Route_Name,Ridership_Total,Route_Hour,Route_Date\n")
            object_id = 1
            for route in range(3):
                for hour in range(6):
                    f.write(
                        f"{object_id},{route},Route {route},{5 + route + hour},"
                        f"{hour / 2.0},2026-01-01\n"
                    )
                    object_id += 1

    def _write_native_summary(self, path: Path) -> None:
        rows = []
        for seed in (1, 2, 3):
            rows.extend([
                {
                    "source": "afc",
                    "seed": seed,
                    "variant": "native_real_interval",
                    "control_score": 10.0,
                    "ep_reward": -5.0,
                    "native_avg_board_wait_min": 3.0,
                    "native_alighted_pax": 100.0,
                    "native_completed_throughput_pax": 100.0,
                    "native_avg_onboard_load": 0.50,
                    "native_peak_onboard_load": 1.0,
                    "LowerLFDrift": 0.20,
                },
                {
                    "source": "afc",
                    "seed": seed,
                    "variant": "native_real_freqhrl",
                    "control_score": 12.0,
                    "ep_reward": -3.0,
                    "native_avg_board_wait_min": 2.0,
                    "native_alighted_pax": 105.0,
                    "native_completed_throughput_pax": 105.0,
                    "native_avg_onboard_load": 0.50,
                    "native_peak_onboard_load": 1.0,
                    "LowerLFDrift": 0.10,
                },
            ])
        checks = [
            {"metric": "control_score", "check": "score", "status": "supported"},
            {"metric": "ep_reward", "check": "reward", "status": "supported"},
            {"metric": "native_avg_board_wait_min", "check": "wait", "status": "supported"},
            {"metric": "native_alighted_pax", "check": "alighted", "status": "supported"},
            {
                "metric": "native_completed_throughput_pax",
                "check": "throughput",
                "status": "supported",
            },
            {"metric": "native_avg_onboard_load", "check": "onboard", "status": "inconclusive"},
            {"metric": "LowerLFDrift", "check": "drift", "status": "supported"},
        ]
        path.write_text(json.dumps({"rows": rows, "paired_checks": checks}), encoding="utf-8")

    def _write_gtfs_ride(self, root: Path) -> None:
        root.mkdir(parents=True)
        (root / "board_alight.txt").write_text(
            "\n".join([
                "trip_id,stop_id,boardings,alightings,load_count",
                "t1,s1,4,0,4",
                "t1,s2,1,3,2",
                "t1,s3,0,2,0",
            ])
            + "\n",
            encoding="utf-8",
        )
        (root / "rider_trip.txt").write_text(
            "\n".join([
                "rider_id,origin_stop_id,destination_stop_id",
                "r1,s1,s2",
                "r2,s1,s3",
            ])
            + "\n",
            encoding="utf-8",
        )
        (root / "trip_capacity.txt").write_text(
            "trip_id,seated_capacity,standing_capacity\n"
            "t1,30,20\n",
            encoding="utf-8",
        )

    def test_current_afc_apc_cache_keeps_external_od_load_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            afc = root / "afc.csv"
            apc = root / "apc.csv"
            native = root / "summary.json"
            self._write_afc(afc)
            self._write_apc(apc)
            self._write_native_summary(native)

            payload = run_coverage(
                output_dir=root / "out",
                afc_csv=afc,
                apc_csv=apc,
                native_summary=native,
                min_afc_rows=10,
                min_afc_stations=3,
                min_afc_time_bins=6,
                min_apc_rows=10,
                min_apc_routes=3,
                min_apc_time_bins=10,
            )

            boundaries = {row["evidence_item"]: row for row in payload["claim_boundaries"]}
            self.assertEqual(boundaries["real_afc_station_hour_demand"]["status"], "supported")
            self.assertEqual(boundaries["real_apc_route_boarding_demand"]["status"], "supported")
            self.assertEqual(
                boundaries["native_service_response_wait_alighting_throughput"]["status"],
                "supported",
            )
            self.assertEqual(boundaries["real_gtfs_ride_onboard_load"]["status"], "external_missing")
            self.assertEqual(boundaries["real_gtfs_ride_od"]["status"], "external_missing")
            self.assertEqual(
                payload["summary"]["evidence_scope"],
                "real_afc_apc_demand_plus_native_service_response",
            )

    def test_optional_gtfs_ride_marks_real_od_and_onboard_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            afc = root / "afc.csv"
            apc = root / "apc.csv"
            native = root / "summary.json"
            gtfs_ride = root / "gtfs_ride"
            self._write_afc(afc)
            self._write_apc(apc)
            self._write_native_summary(native)
            self._write_gtfs_ride(gtfs_ride)

            payload = run_coverage(
                output_dir=root / "out",
                afc_csv=afc,
                apc_csv=apc,
                native_summary=native,
                gtfs_ride_dir=gtfs_ride,
                min_afc_rows=10,
                min_afc_stations=3,
                min_afc_time_bins=6,
                min_apc_rows=10,
                min_apc_routes=3,
                min_apc_time_bins=10,
            )

            boundaries = {row["evidence_item"]: row for row in payload["claim_boundaries"]}
            self.assertEqual(boundaries["real_gtfs_ride_board_alight"]["status"], "supported")
            self.assertEqual(boundaries["real_gtfs_ride_onboard_load"]["status"], "supported")
            self.assertEqual(boundaries["real_gtfs_ride_od"]["status"], "supported")
            self.assertEqual(
                payload["summary"]["evidence_scope"],
                "real_gtfs_ride_od_onboard_plus_native_service_response",
            )


if __name__ == "__main__":
    unittest.main()
