import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.analysis_provenance import (
    csv_artifact_record,
    validate_csv_artifact,
)


class AnalysisProvenanceTest(unittest.TestCase):
    def test_locked_csv_round_trip_and_tamper_rejection(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.csv"
            frame = pd.DataFrame([
                {"config": "a", "seed": 1, "value": 2.0},
                {"config": "a", "seed": 2, "value": 3.0},
            ])
            frame.to_csv(path, index=False)
            record = csv_artifact_record(path, frame, ["config", "seed"])

            observed = validate_csv_artifact(
                path,
                record,
                expected_primary_key=["config", "seed"],
            )
            pd.testing.assert_frame_equal(observed, frame)

            path.write_text(path.read_text().replace("3.0", "4.0"))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                validate_csv_artifact(
                    path,
                    record,
                    expected_primary_key=["config", "seed"],
                )

    def test_duplicate_primary_key_is_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.csv"
            frame = pd.DataFrame([
                {"config": "a", "seed": 1},
                {"config": "a", "seed": 1},
            ])
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "not unique"):
                csv_artifact_record(path, frame, ["config", "seed"])

    def test_manifest_byte_size_is_verified(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.csv"
            frame = pd.DataFrame([{"config": "a", "seed": 1}])
            frame.to_csv(path, index=False)
            record = csv_artifact_record(path, frame, ["config", "seed"])
            record["size_bytes"] += 1

            with self.assertRaisesRegex(ValueError, "byte size"):
                validate_csv_artifact(
                    path,
                    record,
                    expected_primary_key=["config", "seed"],
                )


if __name__ == "__main__":
    unittest.main()
