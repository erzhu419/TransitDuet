from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v79_policy_step_diagnosis as audit


class PolicyStepDiagnosisTest(unittest.TestCase):
    def days(self):
        rng = np.random.default_rng(79)
        return [dict(scenario_seed=i, times=np.arange(1., 4.), horizon_s=4.,
            remaining_passenger=np.array([9., 6., 3.]) * (i + 1), cv_squared=.1 * i,
            readiness_delay_s=2. * i, probabilities=np.full((3, 7), 1. / 7),
            features=rng.normal(size=(3, 7, 4)), actions=np.array([0, 1, 2])) for i in range(4)]

    def test_jackknife_recomputes_baselines_after_each_deleted_day(self):
        days = self.days()
        mean, pseudo = audit.jackknife_gradients(days)
        np.testing.assert_allclose(mean, audit.step.gradient_days(days).mean(0))
        for i in range(4):
            deleted = audit.step.gradient_days(days[:i] + days[i + 1:]).mean(0)
            np.testing.assert_allclose(pseudo[i], 4 * mean - 3 * deleted)

    def test_zero_tilt_has_identical_paths_and_full_effective_sample(self):
        result = audit.path_overlap(self.days(), np.zeros(4))
        self.assertAlmostEqual(result["mean_summed_state_kl"], 0., places=12)
        self.assertAlmostEqual(result["path_weight_effective_days"], 4.)

    def test_path_kl_sums_decisions_and_likelihood_uses_selected_actions(self):
        day = self.days()[0]
        day["features"][:] = 0.
        day["features"][:, 0, 0] = 1.
        theta = np.array([.2, 0., 0., 0.])
        row = audit.path_overlap([day], theta)["days"][0]
        log_z = np.log((np.exp(.2) + 6.) / 7.)
        self.assertAlmostEqual(row["summed_state_kl"], 3 * (log_z - .2 / 7.))
        self.assertAlmostEqual(row["path_log_likelihood_ratio"], .2 - 3 * log_z)

    def test_half_fits_use_only_their_own_days(self):
        calls = []

        def fit(days):
            calls.append([int(d["scenario_seed"]) for d in days])
            return dict(theta=[0.] * 4, update_identified=False, directional_cost_changes=[0.] * 3)

        days = self.days() + self.days()
        for i, day in enumerate(days):
            day["scenario_seed"] = i
        with patch.object(audit.step, "fit_step", side_effect=fit):
            rows = audit.split_directions(days)
        self.assertEqual(calls, [[0, 1, 2, 3], [4, 5, 6, 7]])
        for row in rows:
            self.assertFalse(set(row["fitting_days"]) & set(row["scoring_days"]))
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
