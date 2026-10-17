from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v94_compensation_confirmation as confirmation


class CompensationConfirmationTest(unittest.TestCase):
    def test_archived_actor_split_fresh_days_and_budget_are_fixed(self):
        self.assertEqual(len(confirmation.TRAIN_SEEDS), 12)
        self.assertEqual(len(confirmation.VALIDATION_SEEDS), 10)
        self.assertFalse(set(confirmation.DEVELOPMENT_SEEDS) & set(confirmation.VALIDATION_SEEDS))
        self.assertFalse(set(confirmation.DAYS) & set(confirmation.frozen.DAYS))
        self.assertEqual(len(confirmation.ITEMS) * len(confirmation.MODES), 1728)

    def test_original_joint_gate_is_not_replaced_by_a_better_passenger_arm(self):
        metrics = dict.fromkeys(("passenger_person_s", "cv_squared", "readiness_delay_s",
                                "trips_unlaunched", "trips_incomplete"), None)
        stats = {m: dict(ci95=[-1., -0.1 if m == "passenger_person_s" else 0.]) for m in metrics}
        self.assertTrue(confirmation.joint_gate(dict(preserve_vs_late=stats)))
        stats["cv_squared"]["ci95"][1] = .00001
        self.assertFalse(confirmation.joint_gate(dict(preserve_vs_late=stats)))

    def test_crossed_statistics_average_repeats_and_keep_common_day_uncertainty(self):
        rows = [dict(seed=seed, day=day, effects=dict(preserve_vs_late=dict(metric=value)))
            for seed in (1, 2) for day, value in ((10, -2.), (11, 2.)) for rep in range(3)]
        stats = confirmation.describe(rows, "effects")["preserve_vs_late"]["metric"]
        self.assertEqual(stats["mean"], 0.)
        self.assertEqual(stats["ci95"], [-2., 2.])
        self.assertEqual(stats["n_training_seeds"], 2)
        self.assertEqual(stats["n_demand_days"], 2)

    def test_timing_accounting_reconstructs_measured_cv_change(self):
        def rule(times, held):
            records = [dict(trip_id=i, station_id=2, direction=True, time_s=t,
                release=100.*i, realized_holding=h, other_elapsed=t-100.*i-h)
                for i, (t, h) in enumerate(zip(times, held))]
            return SimpleNamespace(coordinates=confirmation.frozen.coordinate_moments(records),
                summary=dict(arrival=confirmation.frozen.timing.timing_series(records)))
        b, c = rule([100., 220., 410.], [0., 10., 20.]), rule([100., 215., 400.], [0., 5., 10.])
        effects = confirmation.timing_effect(b, c)
        total = sum(v for k, v in effects.items() if k.endswith(("_diagonal", "_cross")))
        np.testing.assert_allclose(total + effects["mean_normalization"], effects["cv_squared"], atol=1e-10, rtol=0)


if __name__ == "__main__":
    unittest.main()
