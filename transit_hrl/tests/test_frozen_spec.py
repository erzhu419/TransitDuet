import unittest
from pathlib import Path

import numpy as np

from freq_hrl.core import (
    FrequencyRouter,
    default_spec,
    validate_claim_freeze,
    validate_frequency_features,
    validate_lower_policy_state,
    validate_shared_core_paths,
    validate_upper_policy_state,
)


class FrozenSpecTest(unittest.TestCase):
    def _features(self):
        return {
            "timestamp": 10.0,
            "x_raw": np.array([1.0]),
            "x_low": np.array([1.0]),
            "x_low_forecast": np.ones((4, 1)),
            "x_low_uncertainty": np.array([0.2]),
            "x_mid": np.array([0.1]),
            "x_high": np.array([0.4]),
            "x_high_energy": np.array([0.16]),
            "x_high_persistence": np.array([0.5]),
            "shock_age": np.array([0.25]),
            "metadata": {"max_observed_timestamp": 10.0},
        }

    def test_default_spec_freezes_c1_to_c9(self):
        spec = default_spec()
        self.assertEqual(spec.version, "freq_hrl_frozen_spec_2026_06_27")
        self.assertEqual(spec.required_claim_ids, tuple(f"C{i}" for i in range(1, 10)))
        self.assertIn("x_high", spec.upper_forbidden_keys)
        self.assertIn("x_low_forecast", spec.lower_forbidden_keys)

    def test_valid_frequency_features_pass(self):
        result = validate_frequency_features(self._features(), current_time=10.0)
        self.assertEqual(result["status"], "supported")

    def test_future_encoder_metadata_fails_causality(self):
        features = self._features()
        features["metadata"] = {"max_observed_timestamp": 11.0}
        with self.assertRaises(ValueError):
            validate_frequency_features(features, current_time=10.0)

    def test_upper_and_lower_policy_contracts(self):
        router = FrequencyRouter()
        features = self._features()
        upper = router.upper_view(features, z_upper={"fleet": 12}, promotion={"promote": False})
        lower = router.lower_view(features, z_lower={"load": 0.3}, current_plan={"target": 300.0})
        self.assertEqual(validate_upper_policy_state(upper)["status"], "supported")
        self.assertEqual(validate_lower_policy_state(lower)["status"], "supported")

        upper_bad = dict(upper)
        upper_bad["x_high"] = np.array([9.0])
        with self.assertRaises(ValueError):
            validate_upper_policy_state(upper_bad)

        lower_bad = dict(lower)
        lower_bad["x_low_forecast"] = np.ones((8, 1))
        with self.assertRaises(ValueError):
            validate_lower_policy_state(lower_bad)

    def test_claim_and_shared_core_validators(self):
        rows = [
            {"claim_id": f"C{i}", "status": "supported"}
            for i in range(1, 10)
        ]
        self.assertEqual(validate_claim_freeze(rows)["status"], "supported")
        with self.assertRaises(ValueError):
            validate_claim_freeze(rows[:-1])

        audit_rows = [
            {"path": "transit_hrl/freq_hrl/core/types.py"},
            {"path": "transit_hrl/freq_hrl/core/spec.py"},
        ]
        self.assertEqual(
            validate_shared_core_paths(audit_rows, source_root=Path("."))["status"],
            "supported",
        )


if __name__ == "__main__":
    unittest.main()
