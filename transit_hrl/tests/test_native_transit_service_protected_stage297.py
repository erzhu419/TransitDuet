import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_service_protected_calibration as rule
from scripts import run_native_transit_service_protected_stage297 as runner
from scripts import submit_native_transit_service_protected_stage297_scheduleurm as submit


class ServiceProtectedCalibrationTests(unittest.TestCase):
    def fixture(self):
        candidates = [{"family": "base", "threshold": None},
            {"family": "risk", "threshold": 0.},
            {"family": "risk", "threshold": .1},
            {"family": "joint", "threshold": 0.}]
        effects = np.array([[-1., .2, -1.2, -2.], [-1.5, -1., -.5, -.4],
            [-1.1, -.2, -.9, -1.2], [-1.2, -.1, -1.1, -1.5]])
        actual = np.broadcast_to(effects, (6, 4, 4, 4)).copy()
        return actual, np.arange(6).reshape(2, 3, 1), candidates, [60, 12, 36, 45]

    def test_joint_score_keeps_service_that_offsets_positive_fleet_cost(self):
        raw = np.zeros((1, 2), int); proposal = np.ones_like(raw)
        fleet = np.array([[.6, -.2]]); service = np.array([[-1., -.01]])
        candidates = [{"family": "base", "threshold": None},
            {"family": "risk", "threshold": 0.}, {"family": "joint", "threshold": 0.}]
        selected = rule.choices(raw, proposal, np.ones_like(raw, bool), fleet, service, candidates)
        np.testing.assert_array_equal(selected[0], [[1, 1], [0, 1], [1, 1]])

    def test_only_base_or_own_raw_actions_and_original_eligibility(self):
        raw = np.array([[2, 3, 4]]); proposal = np.array([[1, 0, 5]])
        keep = np.array([[True, False, True]])
        scores = np.array([[-1., -2., 1.]])
        candidates = rule.family(scores, np.zeros_like(scores), keep)
        selected = rule.choices(raw, proposal, keep, scores, np.zeros_like(scores), candidates)
        self.assertTrue(np.all((selected == raw[:, None]) | (selected == proposal[:, None])))
        self.assertTrue(np.all(selected[:, :, 1] == raw[:, 1]))

    def test_same_fit_quantile_grid_and_zero_for_both_families(self):
        scores = np.arange(32.).reshape(2, 16)
        candidates = rule.family(scores, -scores/2, np.ones_like(scores, bool))
        for name, x in (("risk", scores), ("joint", scores/2)):
            expected = np.unique(np.r_[0., np.quantile(x, rule.QUANTILES)])
            np.testing.assert_array_equal([c["threshold"] for c in candidates if c["family"] == name], expected)

    def test_cheap_fleet_policy_fails_waiting_retention(self):
        result = rule.calibrate(*self.fixture())
        self.assertFalse(result["candidate_statistics"][1]["retention_constraint_passed"])
        self.assertFalse(result["candidate_statistics"][1]["selection_eligible"])
        self.assertEqual(result["selection"]["risk"]["index"], 2)
        self.assertEqual(result["selection"]["joint"]["index"], 3)
        self.assertTrue(result["fresh_confirmation_admitted"])

    def test_waiting_floor_must_hold_in_both_future_halves(self):
        actual, cube, candidates, counts = self.fixture()
        actual[:, :2, 3, 3] = -2.4
        actual[:, 2:, 3, 3] = -.4
        result = rule.calibrate(actual, cube, candidates, counts)
        self.assertGreater(result["candidate_statistics"][3]["waiting_gain_retention"], .5)
        self.assertFalse(result["candidate_statistics"][3]["retention_constraint_passed"])

    def test_all_raw_and_unchanged_base_cannot_admit_new_confirmation(self):
        actual, cube, candidates, counts = self.fixture()
        actual[:, :, 3] = 0
        result = rule.calibrate(actual, cube, candidates, [*counts[:3], 0])
        self.assertFalse(result["candidate_statistics"][3]["selection_eligible"])
        self.assertEqual(result["selection"]["joint"]["index"], 0)
        self.assertFalse(result["fresh_confirmation_admitted"])

    def test_cost_requires_both_calibration_halves(self):
        actual, cube, candidates, counts = self.fixture()
        actual[:, :2, 3, 0] = -3.
        actual[:, 2:, 3, 0] = .1
        result = rule.calibrate(actual, cube, candidates, counts)
        self.assertLess(result["candidate_statistics"][3]["mean"]["cost"], 0)
        self.assertFalse(result["candidate_statistics"][3]["selection_eligible"])

    def test_duplicate_profiles_do_not_create_new_uncertainty_units(self):
        rng = np.random.default_rng(297)
        panel = rng.normal(size=(6, 4, 6, 4))
        raw = np.zeros((1, 6), int)
        selected = np.ones((1, 2, 6), int); selected[:, 0] = 0
        one = rule.values(panel, selected, raw)
        duplicate = rule.values(panel, np.repeat(selected, 16, axis=0), np.repeat(raw, 16, axis=0))
        np.testing.assert_allclose(one, duplicate, atol=1e-14)

    def test_shared_future_noise_cancels_at_the_retention_boundary(self):
        actual, cube, candidates, counts = self.fixture()
        base_wait = np.random.default_rng(7).uniform(-3., -1., (6, 4))
        actual[:, :, 0, 3] = base_wait
        actual[:, :, 3, 3] = .5*base_wait
        candidate = rule.calibrate(actual, cube, candidates, counts)["candidate_statistics"][3]
        self.assertEqual(candidate["retention_difference_MC_bonferroni_upper"], 0.)
        self.assertTrue(candidate["retention_constraint_passed"])

    def test_source_test_role_is_rejected_before_any_read(self):
        with patch.object(runner.parent, "load_rows") as read:
            with self.assertRaisesRegex(ValueError, "only opens"):
                runner.source_inputs("test", {})
            read.assert_not_called()

    def test_family_is_saved_before_calibration_target_access(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [{} for _ in range(6)]
            raw = np.zeros((1, 6), int); proposal = np.ones_like(raw)
            keep = np.ones_like(raw, bool)
            fleet = np.linspace(-.2, .2, 6)[None]; service = np.full_like(fleet, -.5)
            targets = np.zeros((6, 4, 6, 4)); targets[:, :, 1] = [-1., .2, -1.2, -2.]
            roles = []
            def inputs(role, model):
                roles.append(role)
                if role == "calibration": self.assertTrue((root/"freeze_raw/family.json").exists())
                return rows, raw, proposal, keep, fleet, service
            def panel(source, role, bank):
                self.assertEqual((role, bank), ("calibration", "target"))
                self.assertTrue((root/"freeze_raw/family.json").exists())
                return targets, {"native_episodes": 0, "state_futures": 24}
            with patch.object(runner, "directory", return_value=root), patch.object(runner, "source_inputs", side_effect=inputs), \
                    patch.object(runner.parent, "read", return_value={}), patch.object(runner.parent, "load_panel", side_effect=panel), \
                    patch.object(runner.parent.native, "cube", return_value=np.arange(6).reshape(2, 3, 1)):
                runner.run_calibration("unit")
            self.assertEqual(roles, ["fit", "calibration"])
            result = json.loads((root/"report/result.json").read_text())
            self.assertFalse(result["Stage296_test_or_E32_read"])
            self.assertFalse(result["untouched_confirmation_performed"])

    def test_scheduler_is_unpinned_and_does_not_retrain_reference(self):
        job = submit.task("unit")
        self.assertIsNone(job["require_node"])
        self.assertEqual(job["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual((job["cpu"], job["ram_mb"], job["vram"]), (1, 3072, 0))
        self.assertEqual(runner.contract()["raw_reference_fits"], 0)
        self.assertEqual(runner.contract()["new_native_episodes"], 0)


if __name__ == "__main__": unittest.main()
