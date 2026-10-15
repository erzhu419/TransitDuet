from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_freqduet_v62_cost_warmup import (ITEMS, calibration_gate, data_indices, fit_cost,
                                         predict, specification)
from run_freqduet_v61_cost_calibration import remaining_cost


class CostWarmupTest(unittest.TestCase):
    def test_fixed_matrix_holds_out_entire_demand_days(self):
        self.assertEqual(len(ITEMS), 24)
        for case in range(4):
            train = [specification(i) for i in data_indices(case, "train")]
            validation = [specification(i) for i in data_indices(case, "validation")]
            self.assertEqual(len(train), 4)
            self.assertEqual(len(validation), 2)
            self.assertTrue({d["scenario_seed"] for d in train}.isdisjoint(
                d["scenario_seed"] for d in validation))
            self.assertEqual(len({d["label"] for d in train + validation}), 6)

    def test_final_mean_baseline_uses_current_prefix_and_retains_sign(self):
        _, truth = remaining_cost([.10, .20, .05], .08)
        np.testing.assert_allclose(.08 - np.array([.10, .20, .05]), truth)
        self.assertLess(truth[1], 0)

    def test_gate_requires_both_metrics_on_each_complete_day(self):
        day = dict(fitted=dict(mae=.01, rmse=.02), frozen=dict(mae=.2, rmse=.3),
                   zero=dict(mae=.02, rmse=.03), causal_final_mean=dict(mae=.015, rmse=.025))
        self.assertTrue(calibration_gate([day, day]))
        bad = dict(day, fitted=dict(mae=.01, rmse=.026))
        self.assertFalse(calibration_gate([day, bad]))
        self.assertFalse(calibration_gate([dict(day, fitted=day["causal_final_mean"])]))

    def test_signed_supervision_changes_only_the_supplied_cost_model(self):
        states = np.linspace(-1, 1, 64, dtype=np.float32)[:, None]
        day = dict(states=states, actions=np.zeros_like(states), returns=states[:, 0].astype(float) * .04)
        model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.fill_(.2)
        actor = torch.nn.Linear(2, 2)
        actor_before = {k: v.clone() for k, v in actor.state_dict().items()}

        class CostLinear(torch.nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.layer = layer

            def forward(self, states, actions):
                return self.layer(torch.cat([states, actions], dim=-1))

        cost = CostLinear(model)
        before = float(np.square(predict(cost, day) - day["returns"]).mean())
        fit_cost(cost, [day], .01, 62, steps=256, batch_size=32)
        self.assertLess(float(np.square(predict(cost, day) - day["returns"]).mean()), before / 100)
        self.assertLess(float(predict(cost, day)[0]), 0)
        for name, value in actor.state_dict().items():
            torch.testing.assert_close(value, actor_before[name], rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
