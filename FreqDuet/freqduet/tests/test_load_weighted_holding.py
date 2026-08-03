import unittest

import numpy as np

from lower.holding_externality import LoadWeightedHoldingPenalty


class LoadWeightedHoldingPenaltyTest(unittest.TestCase):
    def test_uses_frozen_observation_load_not_mutable_bus_state(self):
        penalty = LoadWeightedHoldingPenalty.from_config({
            "enable": True,
            "reward_weight": 0.06,
            "action_norm_s": 45.0,
            "load_clip": 1.0,
            "source": "observation_load",
        })
        observation = np.zeros(15, dtype=np.float32)
        observation[9] = 0.75

        value, load, normalized_delay = penalty.evaluate(
            observation,
            30.0,
            base_state_dim=8,
            context_features=("queue", "load", "schedule_slack"),
        )

        self.assertAlmostEqual(load, 0.75)
        self.assertAlmostEqual(normalized_delay, 0.5)
        self.assertAlmostEqual(value, 0.03)

    def test_disabled_contract_is_exact_zero_without_load_feature(self):
        penalty = LoadWeightedHoldingPenalty.from_config({"enable": False})

        self.assertEqual(
            penalty.evaluate(
                [1.0], 45.0, base_state_dim=8, context_features=()),
            (0.0, 0.0, 0.0),
        )

    def test_enabled_contract_requires_deployable_apc_load(self):
        penalty = LoadWeightedHoldingPenalty.from_config({
            "enable": True,
            "reward_weight": 0.01,
        })

        with self.assertRaisesRegex(ValueError, "deployable"):
            penalty.validate_observation_contract(
                observation_mode="latent_oracle_legacy",
                context_features=("load",),
            )
        with self.assertRaisesRegex(ValueError, "APC load"):
            penalty.validate_observation_contract(
                observation_mode="deployable_apc_avl_v4",
                context_features=("queue",),
            )


if __name__ == "__main__":
    unittest.main()
