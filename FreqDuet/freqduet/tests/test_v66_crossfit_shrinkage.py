from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v66_crossfit_shrinkage as shrink


class CrossfitShrinkageTest(unittest.TestCase):
    def test_folds_exclude_whole_days_and_cover_training_set_once(self):
        for case in range(4):
            covered = []
            for fold in range(4):
                train, held = shrink.split_indices(case, fold)
                self.assertEqual((len(train), len(held)), (24, 8))
                self.assertTrue(set(train).isdisjoint(held))
                self.assertEqual(sorted(train + held), shrink.expanded.training_indices(case, "large"))
                self.assertTrue(all(shrink.expanded.specification(i)["split"] == "train" for i in train + held))
                covered.extend(held)
            self.assertEqual(sorted(covered), shrink.expanded.training_indices(case, "large"))

    def test_fitting_prior_and_scaler_only_see_excluded_day_training_subset(self):
        encoder = dict(local_dim=2, buses=1)
        all_days = [dict(states=np.full((2, 19), i, dtype=np.float32), prefix=np.zeros(2),
                         actions=np.zeros((2, 1), dtype=np.float32), returns=np.zeros(2), final_cv_squared=float(i))
                    for i in range(32)]
        chosen = [day for i, day in enumerate(all_days) if i % 4 != 0]
        observed = []

        def record(models, days, prior, lr, seed):
            observed.append((days, prior, lr, seed))
            return {}

        with patch.object(shrink.residual, "fit_models", side_effect=record):
            models, prior, _ = shrink.fit_fold_models(chosen, encoder, 8, .001, 66)
        self.assertEqual(prior, np.mean([d["final_cv_squared"] for d in chosen]))
        self.assertEqual(len(observed[0][0]), 24)
        center, scale = shrink.residual.training_statistics([shrink.residual.features(d, encoder) for d in chosen])
        for model in models.values():
            np.testing.assert_array_equal(model.center.numpy(), center)
            np.testing.assert_array_equal(model.scale.numpy(), scale)
        changed = [dict(d, final_cv_squared=1e6, states=d["states"] * 1e6) if i % 4 == 0 else d
                   for i, d in enumerate(all_days)]
        with patch.object(shrink.residual, "fit_models", return_value={}):
            again, repeated_prior, _ = shrink.fit_fold_models([d for i, d in enumerate(changed) if i % 4 != 0], encoder, 8, .001, 66)
        self.assertEqual(repeated_prior, prior)
        for name in models:
            for key in models[name].state_dict():
                torch.testing.assert_close(models[name].state_dict()[key], again[name].state_dict()[key], atol=0, rtol=0)

    def test_shrinkage_is_exact_day_balanced_constrained_mse_minimum(self):
        rows = [shrink.quadratic_stats(np.ones(2), .2), shrink.quadratic_stats(np.ones(100), .8)]
        fitted = shrink.estimate_shrinkage(rows)
        self.assertAlmostEqual(fitted["alpha"], .5)
        self.assertLess(fitted["calibrated_mse"], fitted["prior_mse"])
        self.assertLess(fitted["calibrated_mse"], fitted["raw_mse"])
        self.assertEqual(shrink.estimate_shrinkage([shrink.quadratic_stats(np.ones(3), -1.)])["alpha"], 0.)
        self.assertEqual(shrink.estimate_shrinkage([shrink.quadratic_stats(np.ones(3), 2.)])["alpha"], 1.)
        self.assertEqual(shrink.estimate_shrinkage([shrink.quadratic_stats(np.zeros(3), .5)])["alpha"], 0.)

    def test_zero_and_one_recover_prior_and_raw_without_changing_prefix(self):
        prior, prediction, prefix = .081234567891, np.array([-.01, .02], dtype=np.float32), np.array([.09, .10])
        np.testing.assert_array_equal(shrink.remaining_prediction(prior, prediction, prefix, 0.), prior - prefix)
        np.testing.assert_array_equal(shrink.remaining_prediction(prior, prediction, prefix, 1.), prior + prediction.astype(float) - prefix)
        self.assertTrue((shrink.remaining_prediction(prior, prediction, prefix, 0.) < 0).all())

    def test_fresh_demands_and_private_sampling_streams_are_disjoint(self):
        old = set(shrink.expanded.TRAIN_SEEDS) | set(shrink.expanded.VALIDATION_SEEDS) | set(shrink.residual.VALIDATION_SEEDS)
        old |= set(shrink.expanded.terminal.VALIDATION_SEEDS) | {s for s, _ in shrink.warmup.DAYS}
        self.assertTrue(set(shrink.VALIDATION_SEEDS).isdisjoint(old))
        specs = [shrink.specification(i) for i in range(128)]
        self.assertEqual(len({s["label"] for s in specs}), 128)
        samplers = {s["sampler_seed"] for s in specs}
        self.assertEqual(len(samplers), 128)
        old_samplers = {shrink.expanded.specification(i)["sampler_seed"] for i in range(160)}
        old_samplers |= {shrink.residual.specification(i)["sampler_seed"] for i in range(64)}
        self.assertTrue(samplers.isdisjoint(old_samplers))

    def test_gate_keeps_unshrunk_models_all_old_controls_and_both_metrics(self):
        day = {name: dict(mae=.01, rmse=.02) for name in shrink.residual.CHANNELS}
        day["shrunk_mlp"] = dict(mae=.001, rmse=.002)
        self.assertTrue(shrink.strict_gate([day, day], "shrunk_mlp"))
        for baseline in shrink.residual.CHANNELS:
            changed = dict(day, **{baseline: dict(mae=.0005, rmse=.001)})
            self.assertFalse(shrink.strict_gate([day, changed], "shrunk_mlp"))
        self.assertFalse(shrink.strict_gate([dict(day, shrunk_mlp=dict(mae=.001, rmse=.03))], "shrunk_mlp"))

    def test_intervals_use_thirty_two_demand_day_clusters_not_actors(self):
        cases = [dict(days=[dict(scenario_seed=s, left=dict(mae=float(case)), right=dict(mae=0.))
                            for s in shrink.VALIDATION_SEEDS]) for case in range(4)]
        values = shrink.day_deltas(cases, "left", "right", "mae")
        fitted = shrink.expanded.paired_interval(values, 66)
        self.assertEqual(fitted["n_demand_days"], 32)
        self.assertEqual(fitted["mean"], 1.5)
        self.assertEqual(fitted["ci95"], [1.5, 1.5])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
