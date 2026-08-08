import copy
from tempfile import TemporaryDirectory
import unittest

from frequency.demand_frequency import DemandFrequencyTracker
from runner_v3 import TransitDuetV2Runner
from runner_v3 import load_config
from scripts.validate_freqduet_protocol_v5_configs import (
    REFERENCE,
    ROOT,
    _validate_single_axis,
    validate_all,
    validate_config,
)


class ProtocolV5ConfigTest(unittest.TestCase):
    def test_engineering_matrix_satisfies_the_complete_contract(self):
        self.assertEqual(validate_all(), [])

    def test_validator_rejects_unbudgeted_main_plan(self):
        config = load_config(ROOT / "configs_freqduet" / REFERENCE)
        config["upper"]["timetable_planner"]["headway_budget_mode"] = "free"

        errors = validate_config(config, name="broken")

        self.assertTrue(any("headway_budget_mode" in item for item in errors))

    def test_validator_rejects_nonjourney_primary_endpoint(self):
        config = load_config(ROOT / "configs_freqduet" / REFERENCE)
        config["objective"]["primary_endpoint"] = "service_cost_restricted"

        errors = validate_config(config, name="broken")

        self.assertTrue(any("primary_endpoint" in item for item in errors))

    def test_validator_rejects_confounded_ablation(self):
        reference = load_config(ROOT / "configs_freqduet" / REFERENCE)
        broken = copy.deepcopy(reference)
        broken["_name"] = "broken_guard"
        broken["protocol"]["role"] = "ablation_no_causal_holding_guard_v5"
        broken["lower"]["causal_holding_guard"]["enable"] = False
        broken["lower"]["lr"] = 0.001
        errors = []

        _validate_single_axis(broken, reference, "broken", errors)

        self.assertTrue(any("lower.lr" in item for item in errors))

    def test_raw_history_feature_budget_is_dimension_matched(self):
        main = load_config(ROOT / "configs_freqduet" / REFERENCE)
        raw = load_config(
            ROOT / "configs_freqduet"
            / "F_freqduet_protocol_v5_rawhistory_hiro.yaml")
        main_tracker = DemandFrequencyTracker.from_config(main["frequency"])
        raw_tracker = DemandFrequencyTracker.from_config(raw["frequency"])

        self.assertEqual(
            main_tracker.upper_feature_dim, raw_tracker.upper_feature_dim)
        self.assertEqual(
            main_tracker.lower_feature_dim, raw_tracker.lower_feature_dim)

    def test_v5_nofrequency_state_dimension_is_derived_from_environment(self):
        config = load_config(
            ROOT / "configs_freqduet"
            / "F_freqduet_protocol_v5_nofreq_hiro.yaml")
        with TemporaryDirectory() as tmp:
            config.setdefault("logging", {})["logs_dir"] = tmp
            runner = TransitDuetV2Runner(config)

        self.assertEqual(
            runner.upper_state_dim,
            runner.env.upper_state_dim + runner.upper_plan_context_dim,
        )


if __name__ == "__main__":
    unittest.main()
