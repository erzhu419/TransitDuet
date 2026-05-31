import unittest

import numpy as np

from freq_hrl.policies import (
    FrequencyTradingController,
    FrequencyTradingPlanner,
    LinearFrequencyTradingController,
    LinearFrequencyTradingPlanner,
    LinearTradingParams,
)


class PolicyInterfacesTest(unittest.TestCase):
    def test_trading_planner_and_controller_emit_decisions(self):
        planner = FrequencyTradingPlanner()
        controller = FrequencyTradingController()
        freq = {
            "x_low": np.asarray([0.001, -0.001]),
            "x_mid": np.asarray([0.0, 0.0]),
            "x_high": np.asarray([0.0003, -0.0002]),
            "promotion": {"promote": False, "promotion_strength": 0.0},
        }
        obs = {
            "raw_signal": np.asarray([0.001, -0.001]),
            "position": np.asarray([0.0, 0.0]),
        }
        upper = planner.plan(obs, np.zeros(1), context={"frequency": freq, "n_assets": 2})
        lower = controller.act(obs, np.zeros(1), upper, context={"frequency": freq})
        self.assertEqual(upper.action.shape, (2,))
        self.assertIn("execution_speed", lower.action)
        self.assertIn("residual_order", lower.action)

    def test_linear_trading_policy_uses_shared_frequency_params(self):
        params = LinearTradingParams.from_vector(np.asarray([1.2, 0.3, 0.2, 0.0, 0.6, 0.1, 0.0]))
        planner = LinearFrequencyTradingPlanner(params)
        controller = LinearFrequencyTradingController(params)
        freq = {
            "x_low": np.asarray([0.001, -0.001]),
            "x_mid": np.asarray([0.0002, 0.0001]),
            "x_high": np.asarray([0.0003, -0.0002]),
            "promotion": {"promote": True, "promotion_strength": 0.5},
        }
        obs = {
            "raw_signal": np.asarray([0.001, -0.001]),
            "position": np.asarray([0.0, 0.0]),
        }
        upper = planner.plan(obs, np.zeros(1), context={"frequency": freq, "n_assets": 2})
        lower = controller.act(obs, np.zeros(1), upper, context={"frequency": freq})
        self.assertEqual(upper.action.shape, (2,))
        self.assertEqual(lower.action["execution_speed"].shape, (2,))
        self.assertEqual(LinearTradingParams.from_mapping(params.to_mapping()).to_vector().shape, (7,))


if __name__ == "__main__":
    unittest.main()
