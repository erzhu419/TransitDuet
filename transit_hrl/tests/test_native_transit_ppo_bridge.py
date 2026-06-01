import unittest

import numpy as np

from freq_hrl.experiments.transit.native_shared_ppo import (
    NativeTransitPPOBridge,
    _NativeLowerReplayCollector,
    _SharedPPOPolicyProxy,
)


class _FakeNativeRunner:
    upper_state_dim = 5
    lower_state_dim = 3
    upper_action_dim = 4
    upper_action_low = np.asarray([-60.0, -60.0, -30.0, -30.0], dtype=np.float32)
    upper_action_high = np.asarray([20.0, 20.0, 30.0, 30.0], dtype=np.float32)
    lower_action_bins = np.asarray([0.0, 10.0, 20.0, 30.0], dtype=np.float32)
    timetable_planner = object()
    timetable_terminal_dispatch = True
    timetable_promotion_replan = True
    cfg = {
        "lower": {"action_range": 30.0},
        "frequency": {"method": "dynamic_harmonic_nb"},
        "upper": {"timetable_planner": {"promotion_replan": True}},
    }


class NativeTransitPPOBridgeTest(unittest.TestCase):
    def test_bridge_maps_shared_latents_to_native_bounds(self):
        bridge = NativeTransitPPOBridge.from_runner(_FakeNativeRunner(), hidden_dim=0)
        upper = bridge.upper_latent_to_native(np.asarray([-100.0, 0.0, 100.0, 1.0]))
        self.assertEqual(upper.shape, (4,))
        self.assertTrue(np.all(upper >= _FakeNativeRunner.upper_action_low - 1e-5))
        self.assertTrue(np.all(upper <= _FakeNativeRunner.upper_action_high + 1e-5))
        lower = bridge.lower_latent_to_native(np.asarray([0.0]))
        self.assertEqual(lower.shape, (1,))
        self.assertIn(float(lower[0]), set(_FakeNativeRunner.lower_action_bins.tolist()))
        self.assertGreaterEqual(float(lower[0]), 0.0)
        self.assertLessEqual(float(lower[0]), 30.0)

    def test_bridge_act_methods_return_native_actions(self):
        bridge = NativeTransitPPOBridge.from_runner(_FakeNativeRunner(), hidden_dim=0)
        upper = bridge.act_upper_native(np.zeros(5, dtype=np.float32), sample=False)
        lower = bridge.act_lower_native(np.zeros(3, dtype=np.float32), sample=False)
        self.assertEqual(upper["native_action"].shape, (4,))
        self.assertEqual(lower["native_action"].shape, (1,))
        contract = bridge.contract_dict()
        self.assertEqual(contract["shared_core"], "freq_hrl.rl.DualActorCriticPPO")
        self.assertTrue(contract["terminal_dispatch"])
        self.assertTrue(contract["promotion_replan"])

    def test_native_episode_collector_builds_shared_ppo_batch(self):
        bridge = NativeTransitPPOBridge.from_runner(_FakeNativeRunner(), hidden_dim=0)
        upper_proxy = _SharedPPOPolicyProxy(bridge, "upper")
        lower_proxy = _SharedPPOPolicyProxy(bridge, "lower")
        collector = _NativeLowerReplayCollector(lower_proxy, upper_proxy, bridge.contract)
        upper_state = np.arange(5, dtype=np.float32)
        lower_state = np.arange(3, dtype=np.float32)
        upper_proxy.get_action(upper_state, deterministic=True)
        lower_proxy.get_action(lower_state, deterministic=True)
        collector.push(
            lower_state,
            np.asarray([10.0], dtype=np.float32),
            reward=-1.0,
            cost=0.25,
            next_state=lower_state + 1.0,
            done=False,
            trip_id=3,
        )
        batch = collector.to_batch()
        self.assertIsNotNone(batch)
        self.assertEqual(batch.upper_state.shape, (1, 5))
        self.assertEqual(batch.lower_state.shape, (1, 3))
        self.assertEqual(batch.upper_action.shape, (1, 4))
        self.assertEqual(batch.lower_action.shape, (1, 1))
        self.assertAlmostEqual(float(batch.constraint[0]), 0.25)


if __name__ == "__main__":
    unittest.main()
