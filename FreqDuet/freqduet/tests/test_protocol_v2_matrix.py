import unittest

import numpy as np
import pandas as pd

from scripts.run_freqduet_protocol_v2_matrix import (
    METRICS,
    PROTOCOL_VERSION,
    config_fingerprint,
    paired_sign_flip_p,
    validate_evaluation_frame,
)


def evaluation_frame(seeds):
    rows = []
    for seed in seeds:
        row = {
            "protocol_version": PROTOCOL_VERSION,
            "eval_seed": int(seed),
            "checkpoint_ep": 59,
            "policy_digest": "abc123",
            "scenario_tape_id": f"tape-{seed}",
        }
        row.update({metric: 1.0 for metric in METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


class ProtocolV2MatrixTest(unittest.TestCase):
    def test_evaluation_frame_requires_exact_unique_seed_rows(self):
        frame = evaluation_frame([101, 101])
        with self.assertRaisesRegex(ValueError, "one row per"):
            validate_evaluation_frame(frame, [101, 102], "synthetic")

    def test_evaluation_frame_rejects_nonfinite_metric(self):
        frame = evaluation_frame([101, 102])
        frame.loc[0, "service_cost"] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_evaluation_frame(frame, [101, 102], "synthetic")

    def test_exact_sign_flip_test_uses_training_seed_as_unit(self):
        self.assertAlmostEqual(
            paired_sign_flip_p(np.array([-1.0, -1.0, -1.0])),
            0.25,
        )

    def test_config_fingerprint_covers_inheritance_lineage(self):
        result = config_fingerprint(
            "F_freqduet_protocol_v2_upperdisc_hiro")
        self.assertEqual(len(result["sha256"]), 64)
        self.assertGreater(len(result["lineage"]), 1)
        self.assertTrue(
            result["lineage"][-1].endswith(
                "F_freqduet_protocol_v2_upperdisc_hiro.yaml"))


if __name__ == "__main__":
    unittest.main()
