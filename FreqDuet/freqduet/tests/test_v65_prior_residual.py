from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v65_prior_residual as residual


class PriorResidualTest(unittest.TestCase):
    def test_new_validation_days_and_private_samplers(self):
        old = set(residual.expanded.TRAIN_SEEDS) | set(residual.expanded.VALIDATION_SEEDS)
        old |= set(residual.expanded.terminal.VALIDATION_SEEDS) | {s for s, _ in residual.warmup.DAYS}
        self.assertTrue(set(residual.VALIDATION_SEEDS).isdisjoint(old))
        specs = [residual.specification(i) for i in range(64)]
        self.assertEqual(len({s["label"] for s in specs}), 64)
        previous = {residual.expanded.specification(i)["sampler_seed"] for i in range(160)}
        self.assertEqual(len({s["sampler_seed"] for s in specs}), 64)
        self.assertTrue(previous.isdisjoint({s["sampler_seed"] for s in specs}))

    def test_features_preserve_full_state_and_do_not_read_future_labels(self):
        encoder = dict(local_dim=2, buses=1)
        states = np.arange(2 * 19, dtype=np.float32).reshape(2, 19) / 100
        day = dict(states=states, prefix=np.array([.03, .07]), final_cv_squared=.08,
                   returns=np.array([.05, .01]), actions=np.array([[5], [15]], dtype=np.float32))
        first = residual.features(day, encoder)
        second = residual.features(dict(day, final_cv_squared=100., returns=np.ones(2) * 100), encoder)
        np.testing.assert_array_equal(first["states"], second["states"])
        np.testing.assert_array_equal(first["states"][:, :19], states)
        np.testing.assert_allclose(first["states"][:, -1], day["prefix"] * states[:, 10])

    def test_training_statistics_weight_days_equally_and_ignore_labels(self):
        days = [dict(states=np.zeros((2, 2), dtype=np.float32), actions=np.zeros((2, 1), dtype=np.float32)),
                dict(states=np.ones((6, 2), dtype=np.float32) * 2, actions=np.ones((6, 1), dtype=np.float32) * 2)]
        center, scale = residual.training_statistics(days)
        np.testing.assert_array_equal(center, np.ones(3))
        np.testing.assert_array_equal(scale, np.ones(3))
        changed = [dict(d, returns=np.full(len(d["states"]), 100.)) for d in days]
        np.testing.assert_array_equal(residual.training_statistics(changed)[0], center)

    def test_both_models_start_at_prior_with_distinct_capacities(self):
        states, actions = torch.randn(3, 4), torch.randn(3, 1)
        models = {name: residual.ResidualValue(name, np.zeros(5, dtype=np.float32),
                                              np.ones(5, dtype=np.float32), 8) for name in residual.CANDIDATES}
        for model in models.values():
            torch.testing.assert_close(model(states, actions), torch.zeros(3, 1), atol=0, rtol=0)
        self.assertLess(sum(p.numel() for p in models["linear_residual"].parameters()),
                        sum(p.numel() for p in models["mlp_residual"].parameters()))

    def test_paired_fitting_has_identical_inputs_targets_and_budget(self):
        class Recording(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Linear(3, 1)
                torch.nn.init.zeros_(self.net.weight)
                torch.nn.init.zeros_(self.net.bias)
                self.inputs = []

            def forward(self, states, actions):
                joined = torch.cat((states, actions), dim=-1)
                self.inputs.append(joined.detach().clone())
                return self.net(joined)

        days = [dict(states=np.arange(n * 2, dtype=np.float32).reshape(n, 2) / 10,
                     actions=np.zeros((n, 1), dtype=np.float32), returns=np.ones(n), final_cv_squared=.08) for n in (4, 7)]
        models = {name: Recording() for name in residual.CANDIDATES}
        stats = residual.fit_models(models, days, .08, .001, 650, steps=8, batch_size=4)
        for left, right in zip(*(model.inputs for model in models.values())):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        for value in stats.values():
            self.assertEqual((value["steps"], value["batch_size"], value["learning_rate"]), (8, 4, .001))
            self.assertEqual(value["last_block_mse"], 0.)

    def test_strict_gate_retains_all_old_controls_and_both_metrics(self):
        day = {name: dict(mae=.01, rmse=.02) for name in residual.expanded.CHANNELS}
        day["linear_residual"] = dict(mae=.001, rmse=.002)
        self.assertTrue(residual.strict_gate([day, day], "linear_residual"))
        for baseline in residual.expanded.CHANNELS:
            changed = dict(day, **{baseline: dict(mae=.0005, rmse=.001)})
            self.assertFalse(residual.strict_gate([day, changed], "linear_residual"))
        changed = dict(day, linear_residual=dict(mae=.001, rmse=.03))
        self.assertFalse(residual.strict_gate([day, changed], "linear_residual"))

    def test_intervals_cluster_four_actors_within_sixteen_demand_days(self):
        cases = [dict(days=[dict(scenario_seed=s, left=dict(mae=float(case)), right=dict(mae=0.))
                            for s in residual.VALIDATION_SEEDS]) for case in range(4)]
        deltas = residual.day_deltas(cases, "left", "right", "mae")
        result = residual.expanded.paired_interval(deltas, 65)
        self.assertEqual(result["n_demand_days"], 16)
        self.assertEqual(result["mean"], 1.5)
        self.assertEqual(result["ci95"], [1.5, 1.5])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
