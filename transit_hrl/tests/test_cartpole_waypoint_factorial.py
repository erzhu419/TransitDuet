import unittest

from scripts.run_cartpole_waypoint_factorial import readout


class CartpoleFactorialTest(unittest.TestCase):
    def setUp(self):
        self.plan = {"cells": [{"configuration": i, "position_weight": w} for i, w in enumerate((10., 30.))],
                     "development_seeds": [10, 11, 12, 13], "offsets": [5, 20], "steps_per_gain": 19200,
                     "scope": "diagnostic_only"}
        self.rows = []
        for i, gains in enumerate(((-2., -1.), (3., 8.))):
            self.rows.append({"configuration": i, "status": "complete", "lqr_spectral_radius": .9,
                "measured_cost": {"counts": {"environment_steps": 19200}},
                "paths": [{"seed": s, "offsets": {str(o): {
                    "current": {"episode_ise": 10., "late_ise": 5.},
                    "lag_one_bin": {"episode_ise": 10. + d, "late_ise": 5. + d / 2}}
                    for o, d in zip(self.plan["offsets"], gains)}} for s in self.plan["development_seeds"]]})

    def test_factors_use_pathwise_difference_in_differences(self):
        result = readout(self.plan, self.rows)
        r = result["metrics"]["episode_ise"]
        self.assertEqual(r["gain_30_minus_10"]["mean"], 7.)
        self.assertEqual(r["offset_20_minus_5"]["mean"], 3.)
        self.assertEqual(r["interaction"]["mean"], 4.)
        self.assertEqual(r["interaction"]["n"], 4)
        self.assertEqual(result["metrics"]["late_ise"]["interaction"]["mean"], 2.)
        self.assertFalse(result["parent_qualification_passed"])

    def test_missing_or_permuted_paired_path_rejects_readout(self):
        with self.assertRaises(ValueError):
            readout(self.plan, self.rows[:1])
        self.rows[1]["paths"].reverse()
        with self.assertRaises(ValueError):
            readout(self.plan, self.rows)
