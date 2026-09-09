from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.audit_external_afc_apc_profiles import load_afc, load_apc
from scripts.derive_freqduet_external_profile_balanced_cache import (
    DEFAULT_AFC,
    DEFAULT_APC,
    derive,
)
from scripts.make_freqduet_protocol_v6_supporting_figures import (
    read_balanced_cache_manifest,
)


class BalancedExternalProfileCacheTest(unittest.TestCase):
    def test_derives_frozen_complete_subsets(self) -> None:
        with TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = derive(DEFAULT_AFC, DEFAULT_APC, out)
            afc, afc_coverage = load_afc(
                out / "mta_complete_station_day_2024-10-01.csv"
            )
            apc, apc_coverage = load_apc(
                out
                / "halifax_complete_route_days_2026-01-01_2026-01-07.csv"
            )
            cache_manifest = read_balanced_cache_manifest(
                out / "mta_complete_station_day_2024-10-01.csv",
                out
                / "halifax_complete_route_days_2026-01-01_2026-01-07.csv",
            )

        self.assertEqual(
            manifest["sources"]["public_afc_mta"]["selected_station_complexes"],
            39,
        )
        self.assertEqual(
            manifest["sources"]["public_apc_halifax"]["selected_route_days"],
            37,
        )
        self.assertEqual(
            manifest["sources"]["public_apc_halifax"][
                "excluded_incomplete_routes"
            ],
            ["136"],
        )
        self.assertAlmostEqual(float(afc["share"].sum()), 1.0)
        self.assertAlmostEqual(float(apc["share"].sum()), 1.0)
        self.assertEqual(afc_coverage["rows"], 39 * 24)
        self.assertEqual(afc_coverage["series_count"], 39)
        self.assertEqual(apc_coverage["series_count"], 7)
        self.assertEqual(apc_coverage["profile_units"], 37)
        self.assertEqual(
            apc_coverage["coverage_basis"], "balanced derived cache"
        )
        self.assertIsNotNone(cache_manifest)


if __name__ == "__main__":
    unittest.main()
