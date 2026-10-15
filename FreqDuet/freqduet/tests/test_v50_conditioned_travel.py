from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from freqduet_conditioned_travel import fit_mean, fit_noise, ConditionedTravelPath


def observed():
    return dict(time_s=4500, dt_s=1, routes={"True": dict(station_ids=[0, 1, 2], travel_s=[50, 80])})


def legs():
    return pd.DataFrame([dict(direction="True", station_id=s, start_s=t, end_s=t+d, duration_s=d)
        for s in (0, 1) for t, d in [(100, 50), (500, 60), (1000, 50), (1500, 60), (3650, 90), (4050, 100)]])


class ConditionedTravelTest(unittest.TestCase):
    def test_incomplete_and_simultaneous_legs_cannot_change_means(self):
        prefix = legs()
        expected = fit_mean(prefix, observed())
        future = pd.DataFrame([dict(direction="True", station_id=0, start_s=4200, end_s=end, duration_s=10000)
                              for end in (4500, 4600)])
        actual = fit_mean(pd.concat([prefix, future], ignore_index=True), observed())
        self.assertEqual(expected, actual)

    def test_period_update_is_shrunk_toward_prior_and_tracks_change(self):
        model = fit_mean(legs(), observed())
        self.assertGreater(model["means"]["True:0"], 55)
        self.assertLess(model["means"]["True:0"], 100)

    def test_new_noise_uses_only_mature_prequential_labels(self):
        model = fit_mean(legs(), observed())
        innovations = pd.DataFrame([dict(direction="True", station_id=0, time_s=4300, end_s=end,
            actual=actual, expected=50) for end, actual in [(4400, 60), (4500, 999), (4600, 999)]])
        fitted = fit_noise(model, innovations, legs())
        self.assertEqual(fitted["noise_latest_completion_s"], 4400)
        self.assertEqual(fitted["ratios"]["True:0"], [1.0])

    def test_marginal_ratio_pools_have_unit_arithmetic_mean(self):
        model = fit_mean(legs(), observed())
        innovations = pd.DataFrame(columns=["direction", "station_id", "time_s", "end_s", "actual", "expected"])
        fitted = fit_noise(model, innovations, legs())
        for pool in fitted["ratios"].values():
            self.assertAlmostEqual(float(np.mean(pool)), 1)
        factor = np.asarray(fitted["shared_factor"])
        np.testing.assert_allclose(np.diag(factor @ factor.T), 1, atol=1e-12)

    def test_shared_component_is_stable_within_bin_but_refreshes(self):
        model = dict(now_s=0, dt_s=1, bin_s=300, horizon_s=1800, keys=["True:0"],
            means={"True:0": 50}, ratios={"True:0": list(np.linspace(.1, 1.9, 1000))},
            shared_fraction=1., shared_factor=[[1.]])
        path = ConditionedTravelPath(model, "conditioned_shared", 0)
        vehicle = dict(direction=True, travel_station_id=0)
        self.assertEqual(path(vehicle, 1, 1), path(vehicle, 200, 1))
        self.assertNotEqual(path(vehicle, 1, 1), path(vehicle, 301, 1))

    def test_mean_mode_and_private_rng_preserve_geometry(self):
        model = fit_noise(fit_mean(legs(), observed()),
            pd.DataFrame(columns=["direction", "station_id", "time_s", "end_s", "actual", "expected"]), legs())
        np.random.seed(12)
        before = np.random.get_state()
        path = ConditionedTravelPath(model, "conditioned_mean", 0)
        expected = model["means"]["True:0"]
        self.assertAlmostEqual(path(dict(direction=True, travel_station_id=0), 4500, .5), expected * .5)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])


if __name__ == "__main__":
    unittest.main()
