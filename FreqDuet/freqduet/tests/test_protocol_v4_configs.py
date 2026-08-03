import copy
import unittest

from runner_v3 import load_config
from scripts.validate_freqduet_protocol_v4_configs import (
    REFERENCE,
    ROOT,
    _validate_single_axis,
    validate_all,
    validate_config,
)


class ProtocolV4ConfigTest(unittest.TestCase):
    def test_reference_satisfies_complete_contract(self):
        self.assertEqual(validate_all(), [])

    def test_validator_rejects_oracle_and_legacy_contracts(self):
        config = load_config(ROOT / "configs_freqduet" / REFERENCE)
        broken = copy.deepcopy(config)
        broken["frequency"]["observation_source"] = "latent_arrivals"
        broken["reward_attribution"]["upper_wait_weight"] = 0.15
        broken["upper"]["timetable_planner"][
            "terminal_schedule_mode"] = "bounded_shift_legacy"

        errors = validate_config(broken, name="broken")

        self.assertTrue(any("observation_source" in item for item in errors))
        self.assertTrue(any("upper_wait_weight" in item for item in errors))
        self.assertTrue(any("terminal_schedule_mode" in item for item in errors))

    def test_validator_rejects_confounded_ablation(self):
        reference = load_config(ROOT / "configs_freqduet" / REFERENCE)
        broken = copy.deepcopy(reference)
        broken["_name"] = "broken_nopromotion"
        broken["protocol"]["role"] = "ablation_no_promotion"
        broken["frequency"]["promotion"]["enable"] = False
        broken["lower"]["lr"] = 0.001
        errors = []

        _validate_single_axis(broken, reference, "broken", errors)

        self.assertTrue(any("lower.lr" in item for item in errors))

    def test_validator_rejects_action_contract_mismatch(self):
        config = load_config(ROOT / "configs_freqduet" / REFERENCE)
        broken = copy.deepcopy(config)
        broken["lower"]["action_bins"] = None

        errors = validate_config(broken, name="broken")

        self.assertTrue(any("action_bins" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
