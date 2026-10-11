import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit.native_distributional_supply_plan import residual_draws
from freq_hrl.domains.transit.native_joint_duration import joint_residual_draws, marginal_reproduction
from scripts import analyze_native_transit_joint_duration_stage170 as spec
from tests import test_native_transit_distributional_supply as distributional_tests
from tests import test_native_transit_duration_calibration as duration_tests


class JointDurationTest(unittest.TestCase):
    def test_coupling_preserves_each_marginal_trip_identity_and_native_rng(self):
        model = distributional_tests.DistributionalSupplyTest().model(residuals=(-300., -100., 100., 300.))
        model.update(joint_trip_ids=[0, 1, 2], joint_directions=[False, True, False],
            joint_fit_residual_paths_s=[[i*10., i*20., i*10.] for i in range(5)])
        trips = list(zip(model["joint_trip_ids"], model["joint_directions"]))
        np.random.seed(73)
        state = np.random.get_state()
        joint = joint_residual_draws(model, trips)
        self.assertTrue(marginal_reproduction(model, trips))
        np.testing.assert_array_equal(state[1], np.random.get_state()[1])
        self.assertEqual(state[2:], np.random.get_state()[2:])
        shuffled = joint_residual_draws(model, trips[::-1])
        for tid in joint:
            np.testing.assert_array_equal(joint[tid], shuffled[tid])
        self.assertGreater(np.corrcoef(joint[0], joint[2])[0, 1], .9)
        independent = residual_draws(model, trips)
        self.assertLess(abs(np.corrcoef(independent[0], independent[2])[0, 1]), .4)
        with self.assertRaises(ValueError):
            joint_residual_draws(model, [(0, True), *trips[1:]])

    def test_fit_never_reads_check_outcomes_and_audit_uses_native_peak_truth(self):
        cells = duration_tests.DurationCalibrationTest().cells()
        with patch.object(spec.source, "fit_model", side_effect=distributional_tests.DistributionalSupplyTest().model):
            models = {r: spec.fit_joint_model(r, cells) for r in spec.ROOTS}
            changed = copy.deepcopy(cells)
            for cell in changed.values():
                for t in cell["calibration"][1]["directions"].values():
                    t["completion_s"] = (np.asarray(t["launch_s"])+9000.).tolist()
            self.assertEqual(spec.fit_joint_model(397, changed), models[397])
        for cell in cells.values():
            row = cell["calibration"][1]
            _, _, starts, durations = spec.trip_arrays(row)
            row["peak_fleet"] = spec.occupancy_peak(starts, starts+durations)
        result = spec.check_models(models, cells)
        self.assertEqual(result["marginal_samples_reproduced"], 524)
        self.assertEqual(result["additional_native_steps"], 0)
        self.assertEqual(len(result["panels"]), 10)
        json.dumps(result)
        cells[397, "low_noise"]["calibration"][1]["peak_fleet"] += 1
        with self.assertRaises(ValueError): spec.check_models(models, cells)


if __name__ == "__main__":
    unittest.main()
