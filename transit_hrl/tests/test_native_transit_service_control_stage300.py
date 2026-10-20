import unittest

import numpy as np

from scripts import run_native_transit_service_control_stage300 as spec
from scripts import submit_native_transit_service_control_stage300_scheduleurm as submit


class ServiceOnlyControlTests(unittest.TestCase):
    def test_original_family_is_preserved_in_simultaneous_calibration(self):
        fleet = np.arange(20).reshape(2, 10)/10
        service = -fleet-.5; keep = np.ones_like(fleet, bool)
        original = spec.source.rule.family(fleet, service, keep)
        expanded = spec.family(fleet, service, keep)
        self.assertEqual(expanded[:len(original)], original)
        self.assertTrue(all(c["family"] == "service" for c in expanded[len(original):]))

    def test_service_only_choices_ignore_fleet_and_respect_base_eligibility(self):
        raw = np.zeros((1, 4), int); proposal = np.ones_like(raw)
        keep = np.array([[True, True, False, True]])
        service = np.array([[-2., -.5, -3., 1.]])
        candidate = [{"family": "service", "threshold": -1.}]
        a = spec.choices(raw, proposal, keep, np.zeros_like(service), service, candidate)
        b = spec.choices(raw, proposal, keep, np.full_like(service, 1e6), service, candidate)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[:, 0], [[1, 0, 0, 0]])

    def test_selection_does_not_pick_other_families_or_force_admission(self):
        base = {"family": "base", "selection_eligible": True, "cost_upper_objective": -1.,
            "mean": {"cost": -2.}, "index": 0, "changed_profile_states": 100}
        service = {**base, "family": "service", "cost_upper_objective": -1.2,
            "mean": {"cost": -2.1}, "index": 2, "changed_profile_states": 60}
        other = {**service, "family": "joint", "cost_upper_objective": -99., "index": 1}
        chosen, admitted = spec.select_service({"candidate_statistics": [base, other, service]})
        self.assertEqual(chosen, service); self.assertTrue(admitted)
        chosen, admitted = spec.select_service({"candidate_statistics": [base, other, {**service, "selection_eligible": False}]})
        self.assertEqual(chosen, base); self.assertFalse(admitted)

    def test_offline_scheduler_and_retired_test_contract(self):
        job = submit.task("unit_stage300")
        self.assertIsNone(job["require_node"])
        self.assertEqual(job["ram_mb"], 3072)
        self.assertIn("unit_stage300/inputs", job["stage_input_paths"][-1])
        self.assertFalse(spec.contract()["Stage298_E32_used_for_calibration"])
        self.assertEqual(spec.contract()["new_native_episodes"], 0)


if __name__ == "__main__": unittest.main()
