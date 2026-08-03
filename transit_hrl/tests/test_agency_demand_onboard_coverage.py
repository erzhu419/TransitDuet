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

    def _write_external_truth_summary(self, path: Path) -> None:
        payload = {
            "summary": {
                "evidence_scope": "real_public_board_alight_load_and_estimated_od",
                "supported_boundaries": 3,
            },
            "source_coverage": [
                {
                    "source": "mbta_bus_stop_trip_ridership",
                    "claim_status": "supported",
                    "rows": 100,
                    "unique_routes": 3,
                    "unique_stops": 5,
                    "boundary": "fixture board/alight/load",
                },
                {
                    "source": "mta_subway_od_estimate_2024",
                    "claim_status": "supported",
                    "sample_rows": 100,
                    "unique_origins": 4,
                    "unique_destinations": 4,
                    "boundary": "fixture estimated OD",
                },
            ],
            "claim_boundaries": [
                {
                    "evidence_item": "real_public_bus_stop_board_alight",
                    "status": "supported",
                    "allowed_wording": "real public bus stop/trip boardings and alightings",
                    "forbidden_wording": "GTFS-ride-native board_alight feed",
                    "evidence": "rows=100",
                },
                {
                    "evidence_item": "real_public_bus_stop_onboard_load",
                    "status": "supported",
                    "allowed_wording": "real public bus stop/trip onboard load",
                    "forbidden_wording": "onboard-load improvement",
                    "evidence": "rows=100",
                },
                {
                    "evidence_item": "real_public_subway_od_estimate",
                    "status": "supported",
                    "allowed_wording": "real public agency subway OD estimates",
                    "forbidden_wording": "observed individual OD truth",
                    "evidence": "sample_rows=100",
                },
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

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
            gate = {row["gate"]: row for row in payload["deployment_data_gate"]}
            self.assertEqual(gate["same_agency_field_union"]["status"], "external_missing")
            self.assertEqual(gate["native_control_linkage"]["status"], "external_missing")
            self.assertTrue((root / "out" / "deployment_data_gate.csv").exists())

    def test_legacy_projected_service_metrics_cannot_close_native_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            afc = root / "afc.csv"
            apc = root / "apc.csv"
            native = root / "summary.json"
            self._write_afc(afc)
            self._write_apc(apc)
            self._write_native_summary(native)
            payload = json.loads(native.read_text(encoding="utf-8"))
            for row in payload["rows"]:
                if row["variant"] == "native_real_freqhrl":
                    row["native_service_adjusted"] = 1.0
            payload["paired_checks"].extend([
                {
                    "metric": "native_raw_native_avg_board_wait_min",
                    "check": "raw_wait",
                    "status": "not_supported",
                },
                {
                    "metric": "native_raw_native_alighted_pax",
                    "check": "raw_alighted",
                    "status": "not_supported",
                },
                {
                    "metric": "native_raw_native_completed_throughput_pax",
                    "check": "raw_throughput",
                    "status": "not_supported",
                },
                {
                    "metric": "native_raw_LowerLFDrift",
                    "check": "raw_drift",
                    "status": "not_supported",
                },
            ])
            native.write_text(json.dumps(payload), encoding="utf-8")

            result = run_coverage(
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

            native_summary = result["native_service"]
            self.assertTrue(native_summary["legacy_projection_contaminated"])
            self.assertEqual(native_summary["native_service_response_status"], "not_supported")
            boundary = next(
                row for row in result["claim_boundaries"] if row["id"] == "A3"
            )
            self.assertIn("remains unresolved", boundary["allowed_wording"])

    def test_external_truth_summary_adds_public_board_load_od_without_relabeling_gtfs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            afc = root / "afc.csv"
            apc = root / "apc.csv"
            native = root / "summary.json"
            external_truth = root / "external_truth.json"
            self._write_afc(afc)
            self._write_apc(apc)
            self._write_native_summary(native)
            self._write_external_truth_summary(external_truth)

            payload = run_coverage(
                output_dir=root / "out",
                afc_csv=afc,
                apc_csv=apc,
                native_summary=native,
                external_truth_summary=external_truth,
                min_afc_rows=10,
                min_afc_stations=3,
                min_afc_time_bins=6,
                min_apc_rows=10,
                min_apc_routes=3,
                min_apc_time_bins=10,
            )

            boundaries = {row["evidence_item"]: row for row in payload["claim_boundaries"]}
            self.assertEqual(boundaries["real_public_bus_stop_board_alight"]["status"], "supported")
            self.assertEqual(boundaries["real_public_bus_stop_onboard_load"]["status"], "supported")
            self.assertEqual(boundaries["real_public_subway_od_estimate"]["status"], "supported")
            self.assertEqual(boundaries["real_gtfs_ride_board_alight"]["status"], "external_missing")
            self.assertEqual(boundaries["real_gtfs_ride_onboard_load"]["status"], "external_missing")
            self.assertEqual(boundaries["real_gtfs_ride_od"]["status"], "external_missing")
            self.assertEqual(
                payload["summary"]["evidence_scope"],
                "real_afc_apc_external_board_alight_load_od_plus_native_service_response",
            )
            gate = {row["gate"]: row for row in payload["deployment_data_gate"]}
            self.assertEqual(
                gate["same_agency_field_union"]["status"],
                "partial_external_truth_source_union",
            )
            self.assertEqual(
                gate["native_control_linkage"]["status"],
                "external_truth_not_control_linked",
            )
            self.assertEqual(
                payload["summary"]["same_agency_native_control_status"],
                "external_truth_not_control_linked",
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
                gtfs_ride_source_kind="real_agency",
                gtfs_ride_source_url="https://example.org/agency/gtfs-ride.zip",
                gtfs_ride_agency="Example Transit",
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
            gate = {row["gate"]: row for row in payload["deployment_data_gate"]}
            self.assertEqual(gate["same_agency_field_union"]["status"], "supported")
            self.assertEqual(
                gate["native_control_linkage"]["status"],
                "data_ready_not_control_linked",
            )
            self.assertEqual(
                payload["summary"]["field_complete_data_status"],
                "supported",
            )

    def test_gtfs_ride_without_real_source_provenance_is_not_claim_supported(self):
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
            self.assertEqual(
                boundaries["real_gtfs_ride_board_alight"]["status"],
                "schema_supported_unverified_source",
            )
            self.assertEqual(
                boundaries["real_gtfs_ride_onboard_load"]["status"],
                "schema_supported_unverified_source",
            )
            self.assertEqual(boundaries["real_gtfs_ride_od"]["status"], "schema_supported_unverified_source")
            self.assertEqual(
                payload["summary"]["evidence_scope"],
                "real_afc_apc_demand_plus_native_service_response",
            )
            gate = {row["gate"]: row for row in payload["deployment_data_gate"]}
            self.assertEqual(
                gate["same_agency_field_union"]["status"],
                "external_missing",
            )


if __name__ == "__main__":
    unittest.main()
