import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.build_freqduet_protocol_v6_evidence_package import (
    CONFIRMED_SOURCE_CONFIG,
    NOGUARD_REFERENCE,
    PAPER_CONTROLLER,
)
from scripts.make_freqduet_protocol_v6_supporting_figures import (
    ROOT,
    build_supporting_figures,
)


METRICS = (
    "holding_vehicle_seconds_per_launched_trip",
    "holding_passenger_min_per_generated",
    "fleet_denied_trip_rate",
    "terminal_dispatch_execution_error_abs_mean_s",
)


def write_pair(path: Path, candidate: str, n_pairs: int) -> None:
    row = {
        "candidate": candidate,
        "reference": NOGUARD_REFERENCE,
        "n_pairs": n_pairs,
    }
    for index, metric in enumerate(METRICS, start=1):
        row[f"delta_{metric}_mean"] = -0.1 * index
        row[f"delta_{metric}_ci_low"] = -0.2 * index
        row[f"delta_{metric}_ci_high"] = 0.01 if n_pairs == 24 else -0.01
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


class ProtocolV6SupportingFiguresTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.package = self.root / "package"
        self.package.mkdir()
        (self.package / "evidence_status.json").write_text(json.dumps({
            "protocol": "freqduet-eval-v6",
            "paper_controller": PAPER_CONTROLLER,
            "submission_ready": False,
        }) + "\n")
        write_pair(
            self.package / "source_artifacts" / "v8" / "frozen_paired_deltas.csv",
            CONFIRMED_SOURCE_CONFIG,
            24,
        )
        write_pair(
            self.package / "source_artifacts" / "v9" / "frozen_paired_deltas.csv",
            PAPER_CONTROLLER,
            64,
        )

        self.afc = self.root / "afc.csv"
        pd.DataFrame({
            "transit_timestamp": [
                "2026-01-01 06:00:00",
                "2026-01-01 07:00:00",
                "2026-01-01 17:00:00",
            ],
            "station_complex_id": [1, 1, 2],
            "station_complex": ["A", "A", "B"],
            "ridership": [10, 20, 30],
        }).to_csv(self.afc, index=False)

        self.apc = self.root / "apc.csv"
        pd.DataFrame({
            "Route_Number": ["1", "1", "2"],
            "Route_Name": ["A", "A", "B"],
            "Ridership_Total": [12, 18, 20],
            "Route_Hour": [6.0, 7.5, 17.0],
            "Route_Date": [1767225600000] * 3,
        }).to_csv(self.apc, index=False)

        self.od = self.root / "od.xlsx"
        pd.DataFrame({
            "time": ["06:00", "07:00", "17:00"],
            "origin": ["A", "A", "B"],
            "X1": [3, 6, 12],
            "X2": [2, 4, 8],
        }).to_excel(self.od, index=False)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_builds_method_mechanism_and_realism_exports(self) -> None:
        manifest = build_supporting_figures(
            self.package,
            ("svg", "png"),
            config_root=ROOT,
            afc_path=self.afc,
            apc_path=self.apc,
            od_path=self.od,
        )

        self.assertFalse(manifest["submission_ready"])
        figures = self.package / "figures"
        stems = (
            "fig1_protocol_v6_method",
            "fig4_protocol_v6_physical_outcomes",
            "fig5_protocol_v6_external_realism",
        )
        for stem in stems:
            self.assertGreater((figures / f"{stem}.png").stat().st_size, 1000)
            self.assertIn("<text", (figures / f"{stem}.svg").read_text())
        contract = json.loads(
            (figures / "source_data" / "figure1_method_contract.json").read_text()
        )
        self.assertFalse(contract["legacy_holding_guard_enabled"])
        self.assertEqual(contract["regularity_objective"],
                         "avl_two_sided_incremental_reward")
        self.assertEqual(contract["fleet_size"], 12)
        self.assertEqual(contract["harmonic_forgetting"], 0.9995)
        self.assertEqual(contract["upper_ensemble_size"], 10)
        self.assertEqual(contract["lower_ensemble_size"], 10)
        self.assertEqual(contract["regularity_cost_cap"], 0.25)
        self.assertEqual(contract["objective_wait_metric"], "restricted")
        self.assertEqual(contract["service_cost_weights"]["unserved"], 5.0)
        self.assertTrue(
            (figures / "source_data" / "figure4_physical_outcomes.csv").is_file()
        )
        self.assertTrue(
            (figures / "source_data" / "figure5_hourly_profiles.csv").is_file()
        )

    def test_rejects_missing_physical_metric(self) -> None:
        path = (
            self.package / "source_artifacts" / "v9" / "frozen_paired_deltas.csv"
        )
        with path.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        row.pop("delta_fleet_denied_trip_rate_mean")
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

        with self.assertRaisesRegex(ValueError, "lacks mechanism metric"):
            build_supporting_figures(
                self.package,
                ("svg",),
                config_root=ROOT,
                afc_path=self.afc,
                apc_path=self.apc,
                od_path=self.od,
            )


if __name__ == "__main__":
    unittest.main()
