import unittest
from types import SimpleNamespace

import numpy as np

from freq_hrl.experiments.transit.native_shared_ppo import (
    NativeTransitPPOBridge,
    _NativeLowerReplayCollector,
    _SharedPPOPolicyProxy,
    install_shared_ppo_episode_loop,
    wait_aware_replan_action,
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

    def __init__(self):
        self.upper_trainer = SimpleNamespace(policy_net=None, replay_buffer=None)
        self.lower_trainer = SimpleNamespace(policy_net=None)
        self.replay_buffer = None
        self.timetable_replan_interval_s = 1200.0
        self.timetable_planner = SimpleNamespace(horizon_s=2400.0)


class _FakeHoldFeedbackRunner(_FakeNativeRunner):
    upper_state_dim = 9

    def __init__(self):
        super().__init__()
        self.upper_state_dim = 9
        self.freq_holdfb_dim = 4


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

    def test_bridge_supports_learned_promotion_gate_action(self):
        bridge = NativeTransitPPOBridge.from_runner(
            _FakeNativeRunner(),
            hidden_dim=0,
            learned_promotion_gate=True,
        )
        contract = bridge.contract_dict()
        self.assertEqual(contract["upper_action_dim"], 4)
        self.assertEqual(contract["upper_model_action_dim"], 5)
        self.assertTrue(contract["learned_promotion_gate"])
        native = bridge.upper_latent_to_native(np.asarray([0.0, 0.0, 0.0, 0.0, 2.0]))
        self.assertEqual(native.shape, (4,))
        self.assertGreater(bridge.promotion_gate_value(np.asarray([0.0, 0.0, 0.0, 0.0, 2.0])), 0.5)
        recovered = bridge.upper_latent_to_native(bridge.upper_native_to_latent(native, gate_latent=2.0))
        self.assertTrue(np.allclose(native, recovered, atol=1e-4))

    def test_learned_gate_seed_alignment_preserves_native_policy(self):
        bridge4 = NativeTransitPPOBridge.from_runner(
            _FakeNativeRunner(),
            hidden_dim=0,
            learned_promotion_gate=False,
            native_policy_init_seed=19,
        )
        bridge5 = NativeTransitPPOBridge.from_runner(
            _FakeNativeRunner(),
            hidden_dim=0,
            learned_promotion_gate=True,
            native_policy_init_seed=19,
        )
        state = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
        action4 = bridge4.act_upper_native(state, sample=False)["native_action"]
        action5 = bridge5.act_upper_native(state, sample=False)["native_action"]
        lower4 = bridge4.act_lower_native(np.ones(3, dtype=np.float32), sample=False)["native_action"]
        lower5 = bridge5.act_lower_native(np.ones(3, dtype=np.float32), sample=False)["native_action"]
        self.assertTrue(np.allclose(action4, action5, atol=1e-6))
        self.assertTrue(np.allclose(lower4, lower5, atol=1e-6))

    def test_hold_feedback_seed_alignment_preserves_native_action_policy(self):
        bridge_base = NativeTransitPPOBridge.from_runner(
            _FakeNativeRunner(),
            hidden_dim=0,
            learned_promotion_gate=False,
            native_policy_init_seed=31,
        )
        bridge_hold = NativeTransitPPOBridge.from_runner(
            _FakeHoldFeedbackRunner(),
            hidden_dim=0,
            learned_promotion_gate=True,
            native_policy_init_seed=31,
        )
        state_base = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
        state_hold = np.concatenate([
            state_base,
            np.asarray([0.2, 1.4, 0.1, 0.9], dtype=np.float32),
        ])
        action_base = bridge_base.act_upper_native(
            state_base, sample=False)["native_action"]
        action_hold = bridge_hold.act_upper_native(
            state_hold, sample=False)["native_action"]
        self.assertTrue(np.allclose(action_base, action_hold, atol=1e-6))

    def test_learned_gate_sampled_action_preserves_native_policy(self):
        import torch

        bridge4 = NativeTransitPPOBridge.from_runner(
            _FakeNativeRunner(),
            hidden_dim=0,
            learned_promotion_gate=False,
            native_policy_init_seed=23,
        )
        bridge5 = NativeTransitPPOBridge.from_runner(
            _FakeNativeRunner(),
            hidden_dim=0,
            learned_promotion_gate=True,
            native_policy_init_seed=23,
        )
        state = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
        torch.manual_seed(99)
        action4 = bridge4.act_upper_native(state, sample=True)["native_action"]
        torch.manual_seed(99)
        action5 = bridge5.act_upper_native(state, sample=True)["native_action"]
        self.assertTrue(np.allclose(action4, action5, atol=1e-6))

    def test_learned_gate_prior_skips_hold_feedback_tail(self):
        bridge = NativeTransitPPOBridge.from_runner(
            _FakeHoldFeedbackRunner(),
            hidden_dim=0,
            learned_promotion_gate=True,
        )
        state = np.zeros(9, dtype=np.float32)
        state[2:5] = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
        self.assertGreater(bridge.act_upper_native(state, sample=False)["promotion_gate_value"], 0.5)

        tail_only = np.zeros(9, dtype=np.float32)
        tail_only[-3:] = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
        self.assertLess(bridge.act_upper_native(tail_only, sample=False)["promotion_gate_value"], 0.5)

    def test_policy_proxy_preselects_learned_gate_action(self):
        bridge = NativeTransitPPOBridge.from_runner(
            _FakeNativeRunner(),
            hidden_dim=0,
            learned_promotion_gate=True,
        )
        proxy = _SharedPPOPolicyProxy(bridge, "upper")
        state = np.asarray([0.1, 0.2, 0.3, 1.0, 1.0], dtype=np.float32)
        self.assertTrue(proxy.evaluate_promotion_gate(
            state,
            threshold=0.30,
            sample=False,
            preselect_action=True,
        ))
        native = proxy.get_action(state, deterministic=True)
        self.assertEqual(native.shape, (4,))
        self.assertEqual(proxy.gate_replans, 1)
        self.assertEqual(proxy.decisions, 1)

    def test_policy_proxy_can_preselect_active_plan_override(self):
        bridge = NativeTransitPPOBridge.from_runner(
            _FakeNativeRunner(),
            hidden_dim=0,
            learned_promotion_gate=True,
        )
        proxy = _SharedPPOPolicyProxy(bridge, "upper")
        state = np.asarray([0.1, 0.2, 0.3, 1.0, 1.0], dtype=np.float32)
        active_action = np.asarray([-10.0, -20.0, 5.0, 10.0], dtype=np.float32)
        self.assertTrue(proxy.evaluate_promotion_gate(
            state,
            threshold=0.30,
            sample=False,
            preselect_action=True,
            native_action_override=active_action,
            native_action_blend=0.0,
        ))
        native = proxy.get_action(state, deterministic=True)
        self.assertTrue(np.allclose(native, active_action, atol=1e-4))

    def test_wait_aware_replan_action_shortens_active_direction(self):
        bridge = NativeTransitPPOBridge.from_runner(
            _FakeNativeRunner(),
            hidden_dim=0,
            learned_promotion_gate=True,
        )
        active_action = np.asarray([0.0, 0.0, 5.0, 5.0], dtype=np.float32)
        adjusted, meta = wait_aware_replan_action(
            active_action,
            bridge=bridge,
            planner_key=True,
            freq_summary={
                "freq_low_demand": 0.4,
                "freq_low_forecast": 0.7,
                "freq_low_slope": 0.2,
                "freq_high_energy": 0.3,
                "freq_promotion_strength": 1.0,
            },
            state=np.zeros(5, dtype=np.float32),
            wait_gain_s=20.0,
            max_shift_s=12.0,
            holdfb_dim=0,
            state_wait_weight=0.0,
            frequency_weight=1.0,
            min_pressure=0.0,
        )
        self.assertLess(float(adjusted[0]), float(active_action[0]))
        self.assertLess(float(adjusted[1]), float(active_action[1]))
        self.assertAlmostEqual(float(adjusted[2]), float(active_action[2]), places=5)
        self.assertAlmostEqual(float(adjusted[3]), float(active_action[3]), places=5)
        self.assertGreater(meta["pressure"], 0.0)
        self.assertLess(meta["signed_shift_s"], 0.0)

    def test_wait_aware_replan_uses_same_direction_wait_feedback(self):
        bridge = NativeTransitPPOBridge.from_runner(
            _FakeHoldFeedbackRunner(),
            hidden_dim=0,
            learned_promotion_gate=True,
        )
        active_action = np.asarray([0.0, 0.0, 5.0, 5.0], dtype=np.float32)
        freq_summary = {
            "freq_low_demand": 0.0,
            "freq_low_forecast": 0.0,
            "freq_low_slope": 0.0,
            "freq_high_energy": 0.0,
            "freq_promotion_strength": 0.0,
        }
        other_wait_state = np.zeros(9, dtype=np.float32)
        other_wait_state[-1] = 1.0
        unchanged, meta0 = wait_aware_replan_action(
            active_action,
            bridge=bridge,
            planner_key=True,
            freq_summary=freq_summary,
            state=other_wait_state,
            wait_gain_s=16.0,
            max_shift_s=10.0,
            holdfb_dim=4,
            state_wait_weight=1.0,
            frequency_weight=0.0,
            min_pressure=0.25,
        )
        self.assertTrue(np.allclose(unchanged, active_action, atol=1e-5))
        self.assertEqual(meta0["abs_shift_s"], 0.0)

        same_wait_state = np.zeros(9, dtype=np.float32)
        same_wait_state[-3] = 1.0
        adjusted, meta1 = wait_aware_replan_action(
            active_action,
            bridge=bridge,
            planner_key=True,
            freq_summary=freq_summary,
            state=same_wait_state,
            wait_gain_s=16.0,
            max_shift_s=10.0,
            holdfb_dim=4,
            state_wait_weight=1.0,
            frequency_weight=0.0,
            min_pressure=0.25,
        )
        self.assertLess(float(adjusted[0]), float(active_action[0]))
        self.assertGreater(meta1["abs_shift_s"], 0.0)

        held_state = np.zeros(9, dtype=np.float32)
        held_state[-4] = 1.0
        held_state[-3] = 1.0
        guarded, meta2 = wait_aware_replan_action(
            active_action,
            bridge=bridge,
            planner_key=True,
            freq_summary=freq_summary,
            state=held_state,
            wait_gain_s=16.0,
            max_shift_s=10.0,
            holdfb_dim=4,
            state_wait_weight=1.0,
            frequency_weight=0.0,
            min_pressure=0.25,
            hold_guard_weight=1.0,
        )
        self.assertTrue(np.allclose(guarded, active_action, atol=1e-5))
        self.assertEqual(meta2["abs_shift_s"], 0.0)
        self.assertEqual(meta2["state_same_hold"], 1.0)
        self.assertEqual(meta2["state_same_wait"], 1.0)

        low_wait_state = np.zeros(9, dtype=np.float32)
        low_wait_state[-3] = 0.4
        low_wait_guarded, meta3 = wait_aware_replan_action(
            active_action,
            bridge=bridge,
            planner_key=True,
            freq_summary=freq_summary,
            state=low_wait_state,
            wait_gain_s=16.0,
            max_shift_s=10.0,
            holdfb_dim=4,
            state_wait_weight=1.0,
            frequency_weight=0.0,
            min_pressure=0.25,
            same_wait_min=0.55,
        )
        self.assertTrue(np.allclose(low_wait_guarded, active_action, atol=1e-5))
        self.assertEqual(meta3["abs_shift_s"], 0.0)
        self.assertEqual(meta3["wait_guard_active"], 1.0)

        high_wait_state = np.zeros(9, dtype=np.float32)
        high_wait_state[-3] = 1.2
        high_wait_guarded, meta4 = wait_aware_replan_action(
            active_action,
            bridge=bridge,
            planner_key=True,
            freq_summary=freq_summary,
            state=high_wait_state,
            wait_gain_s=16.0,
            max_shift_s=10.0,
            holdfb_dim=4,
            state_wait_weight=1.0,
            frequency_weight=0.0,
            min_pressure=0.25,
            same_wait_max=0.82,
        )
        self.assertTrue(np.allclose(high_wait_guarded, active_action, atol=1e-5))
        self.assertEqual(meta4["abs_shift_s"], 0.0)
        self.assertEqual(meta4["wait_guard_active"], 1.0)

    def test_wait_aware_replan_respects_dispatch_gap_guard(self):
        bridge = NativeTransitPPOBridge.from_runner(
            _FakeHoldFeedbackRunner(),
            hidden_dim=0,
            learned_promotion_gate=True,
        )
        active_action = np.asarray([0.0, 0.0, 5.0, 5.0], dtype=np.float32)
        freq_summary = {
            "freq_low_demand": 0.0,
            "freq_low_forecast": 0.0,
            "freq_low_slope": 0.0,
            "freq_high_energy": 0.0,
            "freq_promotion_strength": 0.0,
        }
        bunched_state = np.zeros(13, dtype=np.float32)
        bunched_state[3] = 0.50
        bunched_state[8] = 1.00
        bunched_state[-3] = 1.0
        guarded, meta0 = wait_aware_replan_action(
            active_action,
            bridge=bridge,
            planner_key=True,
            freq_summary=freq_summary,
            state=bunched_state,
            wait_gain_s=16.0,
            max_shift_s=10.0,
            holdfb_dim=4,
            state_wait_weight=1.0,
            frequency_weight=0.0,
            min_pressure=0.25,
            gap_guard_min_ratio=0.95,
        )
        self.assertTrue(np.allclose(guarded, active_action, atol=1e-5))
        self.assertEqual(meta0["abs_shift_s"], 0.0)
        self.assertEqual(meta0["gap_guard_active"], 1.0)

        delayed_state = bunched_state.copy()
        delayed_state[3] = 1.20
        adjusted, meta1 = wait_aware_replan_action(
            active_action,
            bridge=bridge,
            planner_key=True,
            freq_summary=freq_summary,
            state=delayed_state,
            wait_gain_s=16.0,
            max_shift_s=10.0,
            holdfb_dim=4,
            state_wait_weight=1.0,
            frequency_weight=0.0,
            min_pressure=0.25,
            gap_guard_min_ratio=0.95,
        )
        self.assertLess(float(adjusted[0]), float(active_action[0]))
        self.assertEqual(meta1["gap_guard_active"], 0.0)

        extreme_gap_state = bunched_state.copy()
        extreme_gap_state[3] = 1.60
        guarded_high, meta2 = wait_aware_replan_action(
            active_action,
            bridge=bridge,
            planner_key=True,
            freq_summary=freq_summary,
            state=extreme_gap_state,
            wait_gain_s=16.0,
            max_shift_s=10.0,
            holdfb_dim=4,
            state_wait_weight=1.0,
            frequency_weight=0.0,
            min_pressure=0.25,
            gap_guard_min_ratio=0.95,
            gap_guard_max_ratio=1.30,
        )
        self.assertTrue(np.allclose(guarded_high, active_action, atol=1e-5))
        self.assertEqual(meta2["abs_shift_s"], 0.0)
        self.assertEqual(meta2["gap_guard_active"], 1.0)

    def test_learned_gate_hook_can_preselect_wait_aware_replan_action(self):
        runner = _FakeNativeRunner()
        bridge = NativeTransitPPOBridge.from_runner(
            runner,
            hidden_dim=0,
            learned_promotion_gate=True,
        )
        installed = install_shared_ppo_episode_loop(
            runner,
            bridge,
            learned_promotion_gate=True,
            promotion_gate_threshold=0.30,
            promotion_gate_strength_min=0.80,
            promotion_gate_age_min=0.50,
            promotion_gate_preselect_action=True,
            promotion_gate_plan_blend=0.0,
            promotion_replan_policy="wait_aware",
            promotion_replan_wait_gain_s=20.0,
            promotion_replan_max_shift_s=12.0,
            promotion_replan_state_wait_weight=0.0,
            promotion_replan_frequency_weight=1.0,
        )
        hook = runner.freq_hrl_learned_promotion_gate
        state = np.asarray([0.1, 0.2, 0.3, 1.0, 1.0], dtype=np.float32)
        active_action = np.asarray([0.0, 0.0, 5.0, 5.0], dtype=np.float32)
        freq_summary = {
            "freq_promotion_flag": 1.0,
            "freq_promotion_strength": 1.0,
            "freq_promotion_age": 1.0,
            "freq_low_demand": 0.4,
            "freq_low_forecast": 0.7,
            "freq_low_slope": 0.2,
            "freq_high_energy": 0.3,
        }
        self.assertTrue(hook(
            s_upper=state,
            elapsed=100.0,
            active_plan={"origin": 0.0, "action": active_action},
            planner_key=True,
            freq_summary=freq_summary,
        ))
        self.assertTrue(hasattr(runner, "freq_hrl_promotion_action_override"))
        self.assertLess(
            float(runner.freq_hrl_promotion_action_override[0]),
            float(active_action[0]),
        )
        native = installed["upper_proxy"].get_action(state, deterministic=True)
        self.assertLess(float(native[0]), float(active_action[0]))
        self.assertLess(float(native[1]), float(active_action[1]))
        self.assertAlmostEqual(float(native[2]), float(active_action[2]), places=5)
        self.assertAlmostEqual(float(native[3]), float(active_action[3]), places=5)
        self.assertEqual(installed["upper_proxy"].gate_replans, 1)
        self.assertEqual(installed["upper_proxy"].wait_replan_abs_shifts[-1], 12.0)

    def test_learned_wait_aware_replan_uses_current_actor_plan_base(self):
        runner = _FakeNativeRunner()
        bridge = NativeTransitPPOBridge.from_runner(
            runner,
            hidden_dim=0,
            learned_promotion_gate=True,
            native_policy_init_seed=37,
        )
        state = np.asarray([0.1, 0.2, 0.3, 1.0, 1.0], dtype=np.float32)
        actor_action = bridge.act_upper_native(state, sample=False)["native_action"]
        active_action = np.asarray([20.0, 20.0, 30.0, 30.0], dtype=np.float32)
        installed = install_shared_ppo_episode_loop(
            runner,
            bridge,
            learned_promotion_gate=True,
            promotion_gate_threshold=0.30,
            promotion_gate_strength_min=0.80,
            promotion_gate_age_min=0.50,
            promotion_gate_preselect_action=True,
            promotion_gate_plan_blend=0.0,
            promotion_replan_policy="learned_wait_aware",
            promotion_replan_wait_gain_s=10.0,
            promotion_replan_max_shift_s=5.0,
            promotion_replan_state_wait_weight=0.0,
            promotion_replan_frequency_weight=1.0,
            promotion_replan_base_action="actor",
        )
        hook = runner.freq_hrl_learned_promotion_gate
        freq_summary = {
            "freq_promotion_flag": 1.0,
            "freq_promotion_strength": 1.0,
            "freq_promotion_age": 1.0,
            "freq_low_demand": 0.4,
            "freq_low_forecast": 0.7,
            "freq_low_slope": 0.2,
            "freq_high_energy": 0.3,
        }
        self.assertTrue(hook(
            s_upper=state,
            elapsed=100.0,
            active_plan={"origin": 0.0, "action": active_action},
            planner_key=True,
            freq_summary=freq_summary,
        ))
        native = installed["upper_proxy"].get_action(state, deterministic=True)
        self.assertLess(float(native[0]), float(actor_action[0]))
        self.assertLess(float(native[1]), float(actor_action[1]))
        self.assertAlmostEqual(float(native[2]), float(actor_action[2]), places=5)
        self.assertAlmostEqual(float(native[3]), float(actor_action[3]), places=5)
        self.assertGreater(installed["upper_proxy"].wait_replan_base_delta_abs[-1], 0.0)
        self.assertGreater(installed["upper_proxy"].wait_replan_final_delta_abs[-1], 0.0)
        self.assertEqual(installed["upper_proxy"].wait_replan_actor_base_used[-1], 1.0)

    def test_learned_actor_replan_trust_region_caps_base_delta(self):
        runner = _FakeNativeRunner()
        bridge = NativeTransitPPOBridge.from_runner(
            runner,
            hidden_dim=0,
            learned_promotion_gate=True,
            native_policy_init_seed=37,
        )
        installed = install_shared_ppo_episode_loop(
            runner,
            bridge,
            learned_promotion_gate=True,
            promotion_gate_threshold=0.30,
            promotion_gate_strength_min=0.80,
            promotion_gate_age_min=0.50,
            promotion_gate_preselect_action=True,
            promotion_replan_policy="learned_wait_aware",
            promotion_replan_wait_gain_s=0.0,
            promotion_replan_max_shift_s=0.0,
            promotion_replan_state_wait_weight=0.0,
            promotion_replan_frequency_weight=0.0,
            promotion_replan_base_action="actor",
            promotion_replan_actor_base_trust_s=2.0,
        )
        state = np.asarray([0.1, 0.2, 0.3, 1.0, 1.0], dtype=np.float32)
        active_action = np.asarray([20.0, 20.0, 30.0, 30.0], dtype=np.float32)
        freq_summary = {
            "freq_promotion_flag": 1.0,
            "freq_promotion_strength": 1.0,
            "freq_promotion_age": 1.0,
        }
        self.assertTrue(runner.freq_hrl_learned_promotion_gate(
            s_upper=state,
            elapsed=100.0,
            active_plan={"origin": 0.0, "action": active_action},
            planner_key=True,
            freq_summary=freq_summary,
        ))
        native = installed["upper_proxy"].get_action(state, deterministic=True)
        self.assertLessEqual(
            float(np.mean(np.abs(native - active_action))),
            2.0 + 1e-4,
        )
        self.assertLessEqual(
            installed["upper_proxy"].wait_replan_base_delta_abs[-1],
            2.0 + 1e-4,
        )

    def test_learned_gate_can_trigger_without_preselecting_plan_action(self):
        bridge = NativeTransitPPOBridge.from_runner(
            _FakeNativeRunner(),
            hidden_dim=0,
            learned_promotion_gate=True,
        )
        proxy = _SharedPPOPolicyProxy(bridge, "upper")
        state = np.asarray([0.1, 0.2, 0.3, 1.0, 1.0], dtype=np.float32)
        self.assertTrue(proxy.evaluate_promotion_gate(
            state,
            threshold=0.30,
            sample=False,
            preselect_action=False,
        ))
        self.assertFalse(proxy.preselected)
        native = proxy.get_action(state, deterministic=True)
        self.assertEqual(native.shape, (4,))
        self.assertEqual(proxy.gate_replans, 1)
        self.assertEqual(proxy.decisions, 1)

    def test_learned_gate_hook_respects_plan_elapsed_guard(self):
        runner = _FakeNativeRunner()
        bridge = NativeTransitPPOBridge.from_runner(
            runner,
            hidden_dim=0,
            learned_promotion_gate=True,
        )
        installed = install_shared_ppo_episode_loop(
            runner,
            bridge,
            learned_promotion_gate=True,
            promotion_gate_threshold=0.30,
            promotion_gate_strength_min=0.80,
            promotion_gate_age_min=0.50,
            promotion_gate_min_elapsed_s=900.0,
            promotion_gate_cooldown_s=900.0,
        )
        hook = runner.freq_hrl_learned_promotion_gate
        state = np.asarray([0.1, 0.2, 0.3, 1.0, 1.0], dtype=np.float32)
        freq_summary = {
            "freq_promotion_flag": 1.0,
            "freq_promotion_strength": 1.0,
            "freq_promotion_age": 1.0,
        }
        self.assertFalse(hook(
            s_upper=state,
            elapsed=300.0,
            active_plan={"origin": 0.0},
            planner_key=True,
            freq_summary=freq_summary,
        ))
        self.assertEqual(installed["upper_proxy"].gate_evaluations, 0)
        self.assertTrue(hook(
            s_upper=state,
            elapsed=950.0,
            active_plan={"origin": 0.0},
            planner_key=True,
            freq_summary=freq_summary,
        ))
        self.assertFalse(hook(
            s_upper=state,
            elapsed=960.0,
            active_plan={"origin": 0.0},
            planner_key=True,
            freq_summary=freq_summary,
        ))
        self.assertEqual(installed["upper_proxy"].gate_replans, 1)

    def test_learned_gate_hook_respects_lf_hf_guards(self):
        runner = _FakeNativeRunner()
        bridge = NativeTransitPPOBridge.from_runner(
            runner,
            hidden_dim=0,
            learned_promotion_gate=True,
        )
        installed = install_shared_ppo_episode_loop(
            runner,
            bridge,
            learned_promotion_gate=True,
            promotion_gate_threshold=0.30,
            promotion_gate_strength_min=0.80,
            promotion_gate_age_min=0.50,
            promotion_gate_min_elapsed_s=0.0,
            promotion_gate_low_signal_min=0.10,
            promotion_gate_max_hf_to_lf_ratio=2.0,
            promotion_gate_max_replans=1,
            promotion_gate_max_total_replans=1,
        )
        hook = runner.freq_hrl_learned_promotion_gate
        state = np.asarray([0.1, 0.2, 0.3, 1.0, 1.0], dtype=np.float32)
        base_summary = {
            "freq_promotion_flag": 1.0,
            "freq_promotion_strength": 1.0,
            "freq_promotion_age": 1.0,
            "freq_low_demand": 0.5,
            "freq_low_forecast": 0.5,
            "freq_low_slope": 0.01,
            "freq_middle": 0.0,
            "freq_middle_energy": 0.0,
            "freq_high_energy": 0.0,
        }
        kwargs = {
            "s_upper": state,
            "elapsed": 100.0,
            "active_plan": {"origin": 0.0},
            "planner_key": True,
        }
        self.assertFalse(hook(freq_summary=base_summary, **kwargs))
        high_hf = dict(base_summary, freq_low_slope=0.20, freq_high_energy=1.0)
        self.assertFalse(hook(freq_summary=high_hf, **kwargs))
        valid = dict(base_summary, freq_low_slope=0.20, freq_high_energy=0.20)
        self.assertTrue(hook(freq_summary=valid, **kwargs))
        self.assertFalse(hook(freq_summary=valid, **kwargs))
        other_key = dict(kwargs, planner_key=False)
        self.assertFalse(hook(freq_summary=valid, **other_key))
        self.assertEqual(installed["upper_proxy"].gate_replans, 1)

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

    def test_lower_hf_wait_action_prior_reduces_holding(self):
        bridge = NativeTransitPPOBridge.from_runner(_FakeNativeRunner(), hidden_dim=0)
        proxy = _SharedPPOPolicyProxy(
            bridge,
            "lower",
            lower_hf_wait_action_gain_s=10.0,
            lower_hf_wait_feature_offset=2,
        )
        state = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
        no_prior = _SharedPPOPolicyProxy(bridge, "lower").get_action(
            state,
            deterministic=True,
        )
        with_prior = proxy.get_action(state, deterministic=True)
        self.assertLessEqual(float(with_prior[0]), float(no_prior[0]))


if __name__ == "__main__":
    unittest.main()
