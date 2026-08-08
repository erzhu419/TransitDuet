import unittest

import numpy as np

from freq_hrl.core import LeakageRegularizer
from freq_hrl.domains.trading import (
    PortfolioExecutionConfig,
    PortfolioExecutionEnv,
    TradingCreditAssigner,
)
from freq_hrl.experiments.trading.ppo_actor_critic import (
    causal_lf_plan_reference,
    decode_hf_tactical_action,
    decode_hierarchical_lower_action,
    evaluate_hf_lower_intervention,
    make_plan_mapper,
    promotion_gate_feature_vector,
    resolve_method_contract,
    train_ppo_actor_critic,
)
from freq_hrl.experiments.trading.offpolicy_baseline_validation import (
    train_flat_offpolicy_baseline,
)
from freq_hrl.experiments.trading.performance_validation import (
    SUPPORT_MIXTURE_COMPONENTS,
)
from freq_hrl.experiments.trading.strong_learned_baseline_validation import (
    count_parameters,
)
from freq_hrl.policies import BernsteinPlanCurve
from freq_hrl.rl import LearnedPlanActionMapper, LearnedPlanCurveState


class FullMethodContractTest(unittest.TestCase):
    def test_causal_lf_reference_uses_only_declared_lf_inputs(self):
        common = {
            "x_low": np.asarray([0.0014, -0.0007]),
            "x_low_forecast": np.asarray([[0.0007, 0.0014]]),
        }
        first = causal_lf_plan_reference(
            {**common, "x_high": np.asarray([100.0, -100.0])},
            2,
            gain=1.0,
            forecast_blend=0.25,
        )
        second = causal_lf_plan_reference(
            {**common, "x_high": np.asarray([-100.0, 100.0])},
            2,
            gain=1.0,
            forecast_blend=0.25,
        )
        np.testing.assert_allclose(first, second)
        self.assertLessEqual(float(np.sum(np.abs(first))), 1.0)

    def test_v4_ablation_contracts_change_only_the_registered_mechanism(self):
        full = resolve_method_contract("full_freq_hrl_v4")
        expected_changes = {
            "ablate_promotion_v4": {"learned_promotion_gate"},
            "ablate_hf_lower_v4": {"lower_hf_overlay"},
            "ablate_leakage_v4": {"constrain_raw_lower_effect"},
        }
        for contract, expected in expected_changes.items():
            with self.subTest(contract=contract):
                ablated = resolve_method_contract(contract)
                changed = {
                    key for key in full if full[key] != ablated[key]
                }
                self.assertEqual(changed, expected)

    def test_v5_ablation_contracts_change_only_the_registered_mechanism(self):
        full = resolve_method_contract("full_freq_hrl_v5")
        expected_changes = {
            "ablate_promotion_v5": {"learned_promotion_gate"},
            "ablate_hf_lower_v5": {"lower_hf_overlay"},
            "ablate_leakage_v5": {"constrain_raw_lower_effect"},
        }
        for contract, expected in expected_changes.items():
            with self.subTest(contract=contract):
                ablated = resolve_method_contract(contract)
                changed = {
                    key for key in full if full[key] != ablated[key]
                }
                self.assertEqual(changed, expected)
                self.assertTrue(ablated["separate_hf_tactical"])

    def test_v6_ablation_contracts_change_only_the_registered_mechanism(self):
        full = resolve_method_contract("full_freq_hrl_v6")
        expected_changes = {
            "ablate_promotion_v6": {"learned_promotion_gate"},
            "ablate_hf_lower_v6": {"lower_hf_overlay"},
            "ablate_leakage_v6": {"constrain_raw_lower_effect"},
        }
        for contract, expected in expected_changes.items():
            with self.subTest(contract=contract):
                ablated = resolve_method_contract(contract)
                changed = {
                    key for key in full if full[key] != ablated[key]
                }
                self.assertEqual(changed, expected)
                self.assertTrue(ablated["promotion_plan_advantage_credit"])
                self.assertTrue(ablated["fixed_rms_leakage_budget"])
                self.assertTrue(ablated["hf_predictability_summary"])

    def test_hf_lower_action_is_bounded_and_separate_from_tracking_speed(self):
        speed, overlay = decode_hierarchical_lower_action(
            np.asarray([-10.0, 10.0, 10.0, -10.0]),
            assets=2,
            enable_hf_overlay=True,
            hf_order_scale=0.025,
        )
        self.assertGreaterEqual(float(speed.min()), 0.05)
        self.assertLessEqual(float(speed.max()), 1.0)
        np.testing.assert_allclose(overlay, [0.025, -0.025], atol=1e-9)

    def test_independent_hf_tactical_action_is_bounded(self):
        overlay = decode_hf_tactical_action(
            np.asarray([10.0, -10.0]),
            assets=2,
            hf_order_scale=0.025,
        )
        np.testing.assert_allclose(overlay, [0.025, -0.025], atol=1e-9)

    def test_learned_gate_features_do_not_include_heuristic_gate_decision(self):
        base = {
            key: np.asarray([0.1, -0.2], dtype=np.float64)
            for key in (
                "x_low",
                "x_low_forecast",
                "x_low_uncertainty",
                "x_mid",
                "x_high",
                "x_high_delta",
                "x_high_energy",
                "x_high_persistence",
                "shock_age",
            )
        }
        inactive = {
            **base,
            "promotion": {"promote": False, "promotion_strength": 0.0},
        }
        active = {
            **base,
            "promotion": {"promote": True, "promotion_strength": 1.0},
        }
        kwargs = dict(
            position=np.asarray([0.0, 0.0]),
            target=np.asarray([0.2, -0.2]),
            leakage_feedback=0.1,
            progress=0.5,
            elapsed_steps=6,
            upper_period=12,
        )
        np.testing.assert_allclose(
            promotion_gate_feature_vector(inactive, **kwargs),
            promotion_gate_feature_vector(active, **kwargs),
        )

    def test_full_training_entrypoint_executes_all_v4_contracts(self):
        payload, rows, model = train_ppo_actor_critic(
            train_seeds=[42],
            validation_seeds=[84],
            eval_seeds=[123],
            steps=36,
            assets=2,
            scenario="persistent_shift",
            iterations=1,
            seed=7,
            leakage_scale=0.1,
            plan_basis_dim=3,
            plan_horizon_s=600.0,
            lower_lf_constraint_coef=0.01,
            lower_lf_constraint_target=0.0,
            lower_lf_dual_lr=0.01,
            upper_period=12,
            min_upper_duration=3,
            execution_timeline_contract="causal_post_trade_v3",
            method_contract="full_freq_hrl_v4",
            volume_impact_bps=10.0,
            plan_smoothness_weight=0.01,
            promotion_replan_cost=0.001,
        )
        self.assertEqual(payload["method_contract"], "full_freq_hrl_v4")
        self.assertEqual(
            payload["execution_timeline_contract"], "causal_post_trade_v3"
        )
        self.assertEqual(payload["mark_to_market_timing"], "post_trade")
        self.assertTrue(payload["executed_plan_curve"])
        self.assertTrue(payload["additive_frequency_credit"])
        self.assertTrue(payload["raw_lower_effect_constraint"])
        self.assertTrue(payload["learned_promotion_gate"])
        self.assertTrue(payload["hf_lower_overlay_enabled"])
        self.assertEqual(payload["config"]["lower_action_dim"], 4)
        self.assertTrue(payload["heuristic_promotion_disabled"])
        self.assertGreater(payload["promotion_gate_state_dim"], 0)
        self.assertTrue(payload["plan_anchor_first_coefficient"])
        self.assertEqual(
            count_parameters(model), payload["capacity_actual_parameter_count"]
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertLessEqual(
            row["task_credit_reconstruction_max_abs_error"], 1e-10
        )
        self.assertAlmostEqual(row["LowerLFDrift"], row["RawLowerLFDrift"])
        self.assertAlmostEqual(
            row["LowerLFDriftAbs"], row["RawLowerLFDriftAbs"]
        )
        self.assertGreater(row["plan_target_step_change_mean"], 0.0)
        self.assertLess(row["upper_decision_count"], row["lower_decision_count"])
        self.assertGreater(row["promotion_gate_transition_count"], 0)
        self.assertEqual(row["promotion_gate_owned_primitive_steps"], 27)
        self.assertEqual(row["promotion_scheduled_boundary_close_count"], 2)
        self.assertGreaterEqual(row["promotion_gate_probability_mean"], 0.0)
        self.assertLessEqual(row["promotion_gate_probability_mean"], 1.0)
        self.assertEqual(row["heuristic_promotion_disabled"], 1.0)
        self.assertEqual(row["hf_lower_overlay_enabled"], 1.0)
        self.assertAlmostEqual(
            row["hf_overlay_task_effect_total"],
            row["hf_overlay_return_total"]
            - row["hf_overlay_incremental_cost_total"],
        )
        self.assertGreater(
            payload["history"][-1]["promotion_actor_optimizer_steps"], 0.0
        )
        intervention_rows = evaluate_hf_lower_intervention(
            model,
            eval_seeds=[123],
            rollout_kwargs={
                "steps": 36,
                "assets": 2,
                "scenario": "persistent_shift",
                "leakage_scale": 0.0,
                "plan_mapper": make_plan_mapper(
                    assets=2,
                    plan_basis_dim=3,
                    plan_horizon_s=600.0,
                    plan_eval_offset_s=300.0,
                    plan_coefficient_scale=0.75,
                    anchor_first_coefficient=True,
                ),
                "upper_period": 12,
                "min_upper_duration": 3,
                "policy_mode": "freq_hrl",
                "mark_to_market_timing": "post_trade",
                "volume_impact_bps": 10.0,
                "execute_plan_curve": True,
                "use_additive_frequency_credit": True,
                "constrain_raw_lower_effect": True,
                "plan_smoothness_weight": 0.01,
                "learned_promotion_gate": True,
                "heuristic_promotion_gate": False,
                "promotion_replan_cost": 0.001,
                "enable_hf_lower": True,
                "lower_hf_order_scale": 0.025,
                "execution_timeline_contract": "causal_post_trade_v3",
                "method_contract": "full_freq_hrl_v4",
            },
        )
        self.assertEqual(len(intervention_rows), 1)
        intervention = intervention_rows[0]
        self.assertTrue(intervention["paired_exogenous_path_identity"])
        self.assertGreaterEqual(intervention["lower_hf_action_sensitivity"], 0.0)
        self.assertTrue(np.isfinite(intervention["total_return_delta"]))

    def test_v5_trains_independent_tracking_and_hf_tactical_streams(self):
        payload, rows, model = train_ppo_actor_critic(
            train_seeds=[42],
            validation_seeds=[84],
            eval_seeds=[123],
            steps=36,
            assets=2,
            scenario="persistent_shift",
            iterations=1,
            seed=7,
            leakage_scale=0.1,
            plan_basis_dim=3,
            plan_horizon_s=600.0,
            lower_lf_constraint_coef=0.01,
            lower_lf_constraint_target=0.0,
            lower_lf_dual_lr=0.01,
            upper_period=12,
            min_upper_duration=3,
            execution_timeline_contract="causal_post_trade_v3",
            method_contract="full_freq_hrl_v5",
            volume_impact_bps=10.0,
            plan_smoothness_weight=0.01,
            promotion_replan_cost=0.001,
            upper_learning_rate=3e-4,
            lower_learning_rate=2e-4,
            hf_learning_rate=1e-4,
            promotion_learning_rate=5e-5,
        )
        self.assertEqual(payload["method_contract"], "full_freq_hrl_v5")
        self.assertEqual(payload["config"]["lower_action_dim"], 2)
        self.assertEqual(payload["config"]["hf_state_dim"], 17)
        self.assertEqual(payload["config"]["hf_action_dim"], 2)
        self.assertTrue(payload["hf_tactical_stream_enabled"])
        self.assertTrue(payload["exact_three_way_credit"])
        self.assertEqual(
            payload["trajectory_contract"]["hf"],
            "one independent tactical transition per primitive step with "
            "a dedicated marginal HF reward",
        )
        self.assertEqual(
            count_parameters(model), payload["capacity_actual_parameter_count"]
        )
        self.assertGreater(
            payload["history"][-1]["hf_actor_optimizer_steps"], 0.0
        )
        self.assertGreater(
            payload["history"][-1]["hf_value_optimizer_steps"], 0.0
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["hf_tactical_transition_count"], 36)
        self.assertEqual(row["hf_tactical_stream_enabled"], 1.0)
        self.assertEqual(row["exact_three_way_credit"], 1.0)
        self.assertLessEqual(
            row["task_credit_reconstruction_max_abs_error"], 1e-10
        )
        self.assertAlmostEqual(
            row["total_lower_credit_mean"],
            row["lower_credit_mean"] + row["hf_tactical_credit_mean"],
        )
        intervention_rows = evaluate_hf_lower_intervention(
            model,
            eval_seeds=[123],
            rollout_kwargs={
                "steps": 36,
                "assets": 2,
                "scenario": "persistent_shift",
                "leakage_scale": 0.0,
                "plan_mapper": make_plan_mapper(
                    assets=2,
                    plan_basis_dim=3,
                    plan_horizon_s=600.0,
                    plan_eval_offset_s=300.0,
                    plan_coefficient_scale=0.75,
                    anchor_first_coefficient=True,
                ),
                "upper_period": 12,
                "min_upper_duration": 3,
                "policy_mode": "freq_hrl",
                "mark_to_market_timing": "post_trade",
                "volume_impact_bps": 10.0,
                "execute_plan_curve": True,
                "use_additive_frequency_credit": True,
                "constrain_raw_lower_effect": True,
                "plan_smoothness_weight": 0.01,
                "learned_promotion_gate": True,
                "heuristic_promotion_gate": False,
                "promotion_replan_cost": 0.001,
                "enable_hf_lower": True,
                "separate_hf_tactical": True,
                "lower_hf_order_scale": 0.025,
                "execution_timeline_contract": "causal_post_trade_v3",
                "method_contract": "full_freq_hrl_v5",
            },
        )
        self.assertEqual(len(intervention_rows), 1)
        self.assertGreaterEqual(
            intervention_rows[0]["lower_hf_overlay_sensitivity"], 0.0
        )

    def test_v6_uses_one_fixed_architecture_for_all_mechanism_ablations(self):
        common = dict(
            train_seeds=[42],
            validation_seeds=[84],
            eval_seeds=[123],
            steps=28,
            assets=2,
            scenario="support_mixture",
            iterations=1,
            seed=7,
            plan_basis_dim=3,
            plan_horizon_s=600.0,
            upper_period=12,
            min_upper_duration=3,
            execution_timeline_contract="causal_post_trade_v3",
            volume_impact_bps=10.0,
            plan_smoothness_weight=0.01,
        )
        models = {}
        payloads = {}
        rows = {}
        for contract in (
            "full_freq_hrl_v6",
            "ablate_promotion_v6",
            "ablate_hf_lower_v6",
            "ablate_leakage_v6",
        ):
            leakage_enabled = contract != "ablate_leakage_v6"
            promotion_enabled = contract != "ablate_promotion_v6"
            payload, contract_rows, model = train_ppo_actor_critic(
                **common,
                method_contract=contract,
                leakage_scale=0.01 if leakage_enabled else 0.0,
                lower_lf_constraint_coef=0.01 if leakage_enabled else 0.0,
                lower_lf_dual_lr=0.01 if leakage_enabled else 0.0,
                promotion_replan_cost=0.001 if promotion_enabled else 0.0,
            )
            payloads[contract] = payload
            rows[contract] = contract_rows[0]
            models[contract] = model

        parameter_counts = {count_parameters(model) for model in models.values()}
        self.assertEqual(len(parameter_counts), 1)
        for contract, payload in payloads.items():
            with self.subTest(contract=contract):
                self.assertTrue(payload["fixed_ablation_architecture"])
                self.assertEqual(payload["capacity_ratio"], 1.0)
                self.assertEqual(payload["config"]["lower_state_dim"], 19)
                self.assertEqual(payload["config"]["hf_state_dim"], 19)
                self.assertGreater(payload["config"]["promotion_state_dim"], 0)
                self.assertEqual(
                    payload["promotion_credit_mode"],
                    "incremental_plan_advantage",
                )
                self.assertEqual(payload["leakage_cost_mode"], "fixed_rms_budget")
                self.assertTrue(payload["hf_predictability_summary"])
                self.assertTrue(payload["training_support_ood_excluded"])

        full_row = rows["full_freq_hrl_v6"]
        self.assertEqual(full_row["promotion_credit_mode"], "incremental_plan_advantage")
        self.assertEqual(full_row["leakage_cost_mode"], "fixed_rms_budget")
        self.assertEqual(full_row["hf_predictability_enabled"], 1.0)
        self.assertEqual(full_row["training_support_ood_excluded"], 1.0)
        self.assertEqual(
            rows["ablate_hf_lower_v6"]["hf_tactical_transition_count"], 0
        )
        self.assertEqual(rows["ablate_hf_lower_v6"]["exact_three_way_credit"], 0.0)
        self.assertEqual(
            rows["ablate_promotion_v6"]["promotion_gate_transition_count"], 0
        )

    def test_rollout_executes_reference_projection_and_promotion_controls(self):
        cap = 0.00025
        payload, rows, _ = train_ppo_actor_critic(
            train_seeds=[42],
            validation_seeds=[84],
            eval_seeds=[123],
            steps=36,
            assets=2,
            scenario="persistent_shift",
            iterations=1,
            seed=7,
            leakage_scale=0.01,
            lower_lf_constraint_coef=0.01,
            lower_lf_dual_lr=0.01,
            plan_basis_dim=3,
            plan_horizon_s=600.0,
            upper_period=12,
            min_upper_duration=3,
            execution_timeline_contract="causal_post_trade_v3",
            method_contract="full_freq_hrl_v6",
            upper_plan_reference_mode="causal_lf",
            upper_plan_reference_gain=1.0,
            upper_plan_reference_forecast_blend=0.25,
            hard_hf_budget_projection=True,
            hf_lf_budget_rms=cap,
            promotion_deterministic_threshold=0.01,
            promotion_adapt_gain=0.25,
            promotion_cooldown_steps=8,
        )
        self.assertEqual(payload["upper_plan_reference_mode"], "causal_lf")
        self.assertTrue(payload["hard_hf_budget_projection"])
        self.assertEqual(payload["promotion_cooldown_steps"], 8)
        row = rows[0]
        self.assertGreater(row["upper_plan_reference_target_abs"], 0.0)
        self.assertGreater(row["upper_plan_reference_coeff_abs"], 0.0)
        self.assertGreater(row["promotion_replan_count"], 0)
        self.assertGreater(row["promotion_cooldown_block_count"], 0)
        self.assertEqual(row["hard_hf_budget_projection"], 1.0)
        self.assertLessEqual(
            row["hf_overlay_rms_after_projection_max"], cap + 1e-12
        )
        self.assertLessEqual(row["hf_leakage_budget_ratio_max"], 1.0 + 1e-9)

    def test_support_training_uses_full_reset_episodes_for_ppo(self):
        steps = 16
        payload, rows, _ = train_ppo_actor_critic(
            train_seeds=[42],
            validation_seeds=[84],
            eval_seeds=[123],
            steps=steps,
            assets=2,
            scenario="support_mixture",
            training_scenarios=SUPPORT_MIXTURE_COMPONENTS,
            iterations=1,
            seed=7,
            leakage_scale=0.01,
            lower_lf_constraint_coef=0.01,
            lower_lf_dual_lr=0.01,
            plan_basis_dim=3,
            plan_horizon_s=600.0,
            upper_period=8,
            min_upper_duration=2,
            execution_timeline_contract="causal_post_trade_v3",
            method_contract="full_freq_hrl_v6",
        )
        multiplier = len(SUPPORT_MIXTURE_COMPONENTS)
        self.assertEqual(
            payload["training_path_protocol"],
            "independent_full_episode_support_batch_v1",
        )
        self.assertEqual(payload["training_episode_count_per_root"], multiplier)
        self.assertEqual(payload["environment_steps_train"], steps * multiplier)
        self.assertEqual(
            payload["environment_steps_validation"], 2 * steps * multiplier
        )
        self.assertEqual(payload["environment_steps_eval"], steps * multiplier)
        self.assertEqual(payload["unique_training_path_count"], multiplier)
        self.assertEqual(rows[0]["support_episode_count"], multiplier)
        self.assertEqual(rows[0]["lower_decision_count"], steps * multiplier)
        self.assertEqual(
            set(rows[0]["support_episode_scenarios"].split("|")),
            set(SUPPORT_MIXTURE_COMPONENTS),
        )

    def test_support_training_uses_same_episode_budget_for_offpolicy(self):
        steps = 8
        payload, rows, _ = train_flat_offpolicy_baseline(
            policy_mode="flat_td3",
            train_seeds=[42],
            validation_seeds=[84],
            eval_seeds=[123],
            steps=steps,
            assets=2,
            scenario="support_mixture",
            training_scenarios=SUPPORT_MIXTURE_COMPONENTS,
            iterations=1,
            seed=7,
            warmup_steps=10_000,
            batch_size=8,
        )
        multiplier = len(SUPPORT_MIXTURE_COMPONENTS)
        self.assertEqual(
            payload["training_path_protocol"],
            "independent_full_episode_support_batch_v1",
        )
        self.assertEqual(payload["environment_steps_train"], steps * multiplier)
        self.assertEqual(payload["environment_steps_eval"], steps * multiplier)
        self.assertEqual(payload["unique_training_path_count"], multiplier)
        self.assertEqual(rows[0]["support_episode_count"], multiplier)

    def test_support_training_uses_same_episode_budget_for_flat_ppo(self):
        steps = 8
        payload, rows, _ = train_ppo_actor_critic(
            train_seeds=[42],
            validation_seeds=[84],
            eval_seeds=[123],
            steps=steps,
            assets=2,
            scenario="support_mixture",
            training_scenarios=SUPPORT_MIXTURE_COMPONENTS,
            iterations=1,
            seed=7,
            policy_mode="flat_ppo",
        )
        multiplier = len(SUPPORT_MIXTURE_COMPONENTS)
        self.assertEqual(
            payload["training_path_protocol"],
            "independent_full_episode_support_batch_v1",
        )
        self.assertEqual(payload["environment_steps_train"], steps * multiplier)
        self.assertEqual(payload["environment_steps_eval"], steps * multiplier)
        self.assertEqual(payload["unique_training_path_count"], multiplier)
        self.assertEqual(rows[0]["support_episode_count"], multiplier)

    def test_full_contract_rejects_noncausal_or_flat_configuration(self):
        common = dict(
            train_seeds=[42],
            validation_seeds=[84],
            eval_seeds=[123],
            steps=12,
            assets=2,
            scenario="persistent_shift",
            iterations=1,
            seed=7,
            leakage_scale=0.1,
            plan_basis_dim=3,
            method_contract="full_freq_hrl_v4",
        )
        with self.assertRaisesRegex(ValueError, "causal_post_trade_v3"):
            train_ppo_actor_critic(**common)
        with self.assertRaisesRegex(ValueError, "hierarchical"):
            train_ppo_actor_critic(
                **common,
                policy_mode="flat_ppo",
                execution_timeline_contract="causal_post_trade_v3",
            )

    def test_post_trade_timeline_makes_current_action_causally_effective(self):
        returns = np.asarray([[0.10]], dtype=np.float64)
        pre = PortfolioExecutionEnv(
            returns,
            config=PortfolioExecutionConfig(
                transaction_cost_bps=0.0,
                slippage_bps=0.0,
                inventory_drift_penalty=0.0,
                mark_to_market_timing="pre_trade",
            ),
        )
        post = PortfolioExecutionEnv(
            returns,
            config=PortfolioExecutionConfig(
                transaction_cost_bps=0.0,
                slippage_bps=0.0,
                inventory_drift_penalty=0.0,
                mark_to_market_timing="post_trade",
            ),
        )
        for env in (pre, post):
            env.set_target([1.0])
        _, pre_reward, _, pre_info = pre.lower_step(1.0)
        _, post_reward, _, post_info = post.lower_step(1.0)
        self.assertAlmostEqual(pre_reward, 0.0)
        self.assertAlmostEqual(post_reward, 0.10)
        np.testing.assert_allclose(pre_info["market_position"], [0.0])
        np.testing.assert_allclose(post_info["market_position"], [1.0])

    def test_volume_impact_is_observed_and_additive(self):
        env = PortfolioExecutionEnv(
            np.zeros((1, 1), dtype=np.float64),
            volumes=np.asarray([[0.5]], dtype=np.float64),
            config=PortfolioExecutionConfig(
                transaction_cost_bps=10.0,
                slippage_bps=0.0,
                volume_impact_bps=100.0,
                inventory_drift_penalty=0.0,
                mark_to_market_timing="post_trade",
            ),
        )
        env.set_target([1.0])
        _, reward, _, info = env.lower_step(1.0)
        self.assertAlmostEqual(info["linear_transaction_cost"], 0.001)
        self.assertAlmostEqual(info["volume_impact_cost"], 0.02)
        self.assertAlmostEqual(info["transaction_cost"], 0.021)
        self.assertAlmostEqual(reward, -0.021)

    def test_hf_overlay_has_a_tracking_only_counterfactual(self):
        env = PortfolioExecutionEnv(
            np.asarray([[0.10]], dtype=np.float64),
            config=PortfolioExecutionConfig(
                transaction_cost_bps=0.0,
                slippage_bps=0.0,
                volume_impact_bps=0.0,
                inventory_drift_penalty=0.0,
                mark_to_market_timing="post_trade",
            ),
        )
        env.set_target([0.0])
        _, reward, _, info = env.lower_step({
            "execution_speed": [1.0],
            "residual_order": [0.2],
        })
        np.testing.assert_allclose(info["tracking_only_position"], [0.0])
        np.testing.assert_allclose(info["hf_overlay_position_effect"], [0.2])
        self.assertAlmostEqual(info["tracking_portfolio_return"], 0.0)
        self.assertAlmostEqual(info["hf_overlay_return"], 0.02)
        self.assertAlmostEqual(info["hf_overlay_task_effect"], 0.02)
        self.assertAlmostEqual(info["tracking_task_reward"], 0.0)
        self.assertAlmostEqual(info["tracking_hf_task_reconstruction_error"], 0.0)
        self.assertAlmostEqual(reward, 0.02)

    def test_hf_overlay_projection_enforces_realized_rms_budget(self):
        env = PortfolioExecutionEnv(
            np.asarray([[0.10, -0.05]], dtype=np.float64),
            config=PortfolioExecutionConfig(
                transaction_cost_bps=0.0,
                slippage_bps=0.0,
                volume_impact_bps=0.0,
                inventory_drift_penalty=0.0,
                mark_to_market_timing="post_trade",
            ),
        )
        env.set_target([0.3, -0.2])
        _, _, _, info = env.lower_step({
            "execution_speed": [0.5, 0.5],
            "residual_order": [0.4, -0.2],
            "hf_overlay_rms_cap": 0.05,
        })
        self.assertTrue(info["hf_overlay_projected"])
        self.assertLess(info["hf_overlay_projection_scale"], 1.0)
        self.assertGreater(info["hf_overlay_rms_before_projection"], 0.05)
        self.assertAlmostEqual(
            info["hf_overlay_rms_after_projection"], 0.05
        )
        self.assertLessEqual(np.sum(np.abs(info["position"])), 1.0)

    def test_hf_overlay_projection_rejects_invalid_budget(self):
        env = PortfolioExecutionEnv(np.zeros((1, 1), dtype=np.float64))
        env.set_target([0.0])
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            env.lower_step({
                "execution_speed": [1.0],
                "residual_order": [0.1],
                "hf_overlay_rms_cap": 0.0,
            })

    def test_stepwise_hf_projection_implies_episode_lf_budget(self):
        cap = 0.03
        rng = np.random.default_rng(9)
        env = PortfolioExecutionEnv(
            np.zeros((48, 3), dtype=np.float64),
            config=PortfolioExecutionConfig(
                transaction_cost_bps=0.0,
                slippage_bps=0.0,
                inventory_drift_penalty=0.0,
            ),
        )
        effects = []
        for _ in range(48):
            env.set_target(rng.uniform(-0.4, 0.4, size=3))
            _, _, done, info = env.lower_step({
                "execution_speed": rng.uniform(0.1, 1.0, size=3),
                "residual_order": rng.normal(0.0, 0.2, size=3),
                "hf_overlay_rms_cap": cap,
            })
            effects.append(info["hf_overlay_position_effect"])
            if done:
                break
        leakage = LeakageRegularizer(lower_lf_window=24).compute(
            np.zeros_like(effects), np.asarray(effects)
        )
        self.assertLessEqual(leakage["LowerLFDriftAbs"], cap * cap + 1e-12)

    def test_hf_counterfactual_includes_incremental_drawdown(self):
        env = PortfolioExecutionEnv(
            np.asarray([[-0.20]], dtype=np.float64),
            config=PortfolioExecutionConfig(
                transaction_cost_bps=0.0,
                slippage_bps=0.0,
                inventory_drift_penalty=0.0,
                drawdown_penalty=0.5,
                mark_to_market_timing="post_trade",
            ),
        )
        env.set_target([0.0])
        _, reward, _, info = env.lower_step(
            {"execution_speed": [1.0], "residual_order": [0.5]}
        )
        self.assertAlmostEqual(info["tracking_drawdown_cost"], 0.0)
        self.assertAlmostEqual(info["hf_overlay_incremental_drawdown_cost"], 0.05)
        self.assertAlmostEqual(info["hf_overlay_return"], -0.1)
        self.assertAlmostEqual(info["hf_overlay_task_effect"], -0.15)
        self.assertAlmostEqual(reward, -0.15)
        self.assertAlmostEqual(info["tracking_hf_task_reconstruction_error"], 0.0)

    def test_tactical_credit_exactly_reconstructs_three_policy_reward(self):
        env = PortfolioExecutionEnv(
            np.asarray([[-0.08, 0.03]], dtype=np.float64),
            volumes=np.asarray([[0.8, 1.2]], dtype=np.float64),
            config=PortfolioExecutionConfig(
                transaction_cost_bps=8.0,
                slippage_bps=2.0,
                volume_impact_bps=15.0,
                inventory_drift_penalty=0.1,
                drawdown_penalty=0.2,
                mark_to_market_timing="post_trade",
            ),
        )
        plan = np.asarray([0.4, -0.2], dtype=np.float64)
        env.set_target(plan)
        _, reward, _, info = env.lower_step(
            {
                "execution_speed": [0.5, 0.75],
                "residual_order": [-0.1, 0.05],
            }
        )
        credit = TradingCreditAssigner().assign_tactical(
            info,
            active_plan=plan,
            upper_leakage_cost=0.03,
            tracking_leakage_cost=0.02,
            hf_leakage_cost=0.01,
            plan_smoothness_cost=0.005,
        )
        self.assertAlmostEqual(
            credit.upper_task_credit
            + credit.tracking_task_credit
            + credit.hf_task_credit,
            reward,
        )
        self.assertAlmostEqual(credit.hf_task_credit, info["hf_overlay_task_effect"])
        self.assertAlmostEqual(credit.task_reconstruction_error, 0.0)
        self.assertAlmostEqual(
            credit.upper_training_credit
            + credit.tracking_training_credit
            + credit.hf_training_credit,
            reward - 0.03 - 0.02 - 0.01 - 0.005,
        )

    def test_frequency_credit_exactly_reconstructs_observed_reward(self):
        env = PortfolioExecutionEnv(
            np.asarray([[0.04, -0.02]], dtype=np.float64),
            volumes=np.asarray([[1.0, 2.0]], dtype=np.float64),
            config=PortfolioExecutionConfig(
                transaction_cost_bps=10.0,
                slippage_bps=5.0,
                volume_impact_bps=20.0,
                inventory_drift_penalty=0.2,
                drawdown_penalty=0.1,
                mark_to_market_timing="post_trade",
            ),
        )
        plan = np.asarray([0.6, -0.2], dtype=np.float64)
        env.set_target(plan)
        _, reward, _, info = env.lower_step([0.5, 1.0])
        credit = TradingCreditAssigner().assign(
            info,
            active_plan=plan,
            upper_leakage_cost=0.03,
            lower_leakage_cost=0.02,
            plan_smoothness_cost=0.01,
        )
        self.assertAlmostEqual(credit.task_reward, reward)
        self.assertAlmostEqual(
            credit.upper_task_credit + credit.lower_task_credit,
            reward,
        )
        self.assertAlmostEqual(credit.task_reconstruction_error, 0.0)
        self.assertAlmostEqual(
            credit.upper_training_credit + credit.lower_training_credit,
            reward - 0.03 - 0.02 - 0.01,
        )

    def test_lower_credit_changes_with_hf_outcome_at_fixed_plan(self):
        assigner = TradingCreditAssigner()
        base = {
            "market_position": np.asarray([0.25]),
            "transaction_cost": 0.0,
            "inventory_drift_cost": 0.0,
            "drawdown_cost": 0.0,
        }
        plan = np.asarray([0.5])
        positive = assigner.assign(
            {
                **base,
                "asset_returns": np.asarray([0.10]),
                "task_reward": 0.025,
            },
            plan,
        )
        negative = assigner.assign(
            {
                **base,
                "asset_returns": np.asarray([-0.10]),
                "task_reward": -0.025,
            },
            plan,
        )
        self.assertLess(positive.lower_task_credit, 0.0)
        self.assertGreater(negative.lower_task_credit, 0.0)

    def test_learned_plan_curve_executes_and_replans_continuously(self):
        mapper = LearnedPlanActionMapper(
            curve=BernsteinPlanCurve(
                horizon_s=120.0,
                basis_dim=3,
                n_entities=1,
                delta_min=-1.0,
                delta_max=1.0,
            ),
            coefficient_scale=1.0,
            anchor_first_coefficient=True,
        )
        self.assertEqual(mapper.action_dim, 2)
        state = LearnedPlanCurveState(mapper=mapper, gross_cap=1.0)
        activation = state.activate(
            now_s=0.0,
            current_value=[0.2],
            latent_action=[0.4, 0.8],
        )
        self.assertAlmostEqual(float(activation.coefficients[0]), 0.0)
        at_zero = state.value_at(0.0)
        at_middle = state.value_at(60.0)
        at_end = state.value_at(120.0)
        np.testing.assert_allclose(at_zero, [0.2], atol=1e-12)
        self.assertNotAlmostEqual(float(at_middle[0]), float(at_zero[0]))
        self.assertNotAlmostEqual(float(at_end[0]), float(at_middle[0]))

        before_replan = state.value_at(45.0)
        state.activate(
            now_s=45.0,
            current_value=[-0.9],
            latent_action=[-0.5, -0.8],
        )
        after_replan = state.value_at(45.0)
        np.testing.assert_allclose(after_replan, before_replan, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
