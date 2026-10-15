from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v64_independent_days as expanded


class IndependentDayTest(unittest.TestCase):
    def test_new_demands_and_private_policy_streams_are_disjoint(self):
        self.assertEqual(len(expanded.ITEMS), 160)
        old = {s for s, _ in expanded.warmup.DAYS} | set(expanded.terminal.VALIDATION_SEEDS)
        self.assertTrue(set(expanded.TRAIN_SEEDS).isdisjoint(old | set(expanded.VALIDATION_SEEDS)))
        self.assertTrue(set(expanded.VALIDATION_SEEDS).isdisjoint(old))
        specs = [expanded.specification(i) for i in range(160)]
        self.assertEqual(len({s["sampler_seed"] for s in specs}), 160)
        self.assertEqual(len({s["label"] for s in specs}), 160)

    def test_small_set_is_fixed_nested_independent_days_not_decisions(self):
        for case in range(4):
            small = expanded.training_indices(case, "small")
            large = expanded.training_indices(case, "large")
            self.assertEqual(small, large[:4])
            self.assertEqual(len(large), 32)
            for indices in (small, large):
                specs = [expanded.specification(i) for i in indices]
                self.assertEqual(len({s["scenario_seed"] for s in specs}), len(indices))
                self.assertTrue(all(s["split"] == "train" for s in specs))

    def test_representations_receive_identical_minibatches_and_budgets(self):
        class RecordingCost(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(3, 1)
                torch.nn.init.zeros_(self.layer.weight)
                torch.nn.init.zeros_(self.layer.bias)
                self.inputs = []

            def forward(self, states, actions):
                joined = torch.cat([states, actions], dim=-1)
                self.inputs.append(joined.detach().clone())
                return self.layer(joined)

        days = [dict(states=np.arange(n * 2, dtype=np.float32).reshape(n, 2) / 10,
                     actions=np.zeros((n, 1), dtype=np.float32), returns=np.linspace(-.02, .02, n),
                     final_cv_squared=.08) for n in (4, 7)]
        models = {rep: RecordingCost() for rep in expanded.REPRESENTATIONS}
        stats = expanded.fit_pair(models, days, .001, 640, steps=8, batch_size=4)
        self.assertEqual(len(models["direct"].inputs), 8)
        for left, right in zip(models["direct"].inputs, models["factored"].inputs):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        for rep in expanded.REPRESENTATIONS:
            self.assertEqual((stats[rep]["steps"], stats[rep]["batch_size"], stats[rep]["learning_rate"]), (8, 4, .001))

    def test_strict_gate_retains_legacy_and_both_small_model_controls(self):
        day = {name: dict(mae=.01, rmse=.02) for name in expanded.GATE_BASELINES}
        day["large_direct"] = dict(mae=.001, rmse=.002)
        self.assertTrue(expanded.strict_gate([day, day], "direct"))
        for baseline in ("legacy_direct", "small_direct", "small_factored", "large_prior"):
            bad = dict(day, **{baseline: dict(mae=.0005, rmse=.001)})
            self.assertFalse(expanded.strict_gate([day, bad], "direct"))
        bad = dict(day, large_direct=dict(mae=.001, rmse=.03))
        self.assertFalse(expanded.strict_gate([day, bad], "direct"))

    def test_bootstrap_clusters_shared_actor_contexts_within_each_demand_day(self):
        cases = [dict(days=[dict(scenario_seed=s, left=dict(mae=float(case)), right=dict(mae=0.))
                            for s in expanded.VALIDATION_SEEDS]) for case in range(4)]
        deltas = expanded.day_deltas(cases, "left", "right", "mae")
        result = expanded.paired_interval(deltas, 64)
        self.assertEqual(result["n_demand_days"], 8)
        self.assertEqual(result["mean"], 1.5)
        self.assertEqual(result["ci95"], [1.5, 1.5])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
