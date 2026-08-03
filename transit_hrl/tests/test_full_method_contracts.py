import unittest

import numpy as np

from freq_hrl.domains.trading import (
    PortfolioExecutionConfig,
    PortfolioExecutionEnv,
    TradingCreditAssigner,
)
from freq_hrl.experiments.trading.ppo_actor_critic import train_ppo_actor_critic
from freq_hrl.policies import BernsteinPlanCurve
from freq_hrl.rl import LearnedPlanActionMapper, LearnedPlanCurveState


class FullMethodContractTest(unittest.TestCase):
    def test_full_training_entrypoint_executes_all_v3_contracts(self):
        payload, rows, _ = train_ppo_actor_critic(
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
            method_contract="full_freq_hrl_v3",
            volume_impact_bps=10.0,
            plan_smoothness_weight=0.01,
        )
        self.assertEqual(payload["method_contract"], "full_freq_hrl_v3")
        self.assertEqual(
            payload["execution_timeline_contract"], "causal_post_trade_v3"
        )
        self.assertEqual(payload["mark_to_market_timing"], "post_trade")
        self.assertTrue(payload["executed_plan_curve"])
        self.assertTrue(payload["additive_frequency_credit"])
        self.assertTrue(payload["raw_lower_effect_constraint"])
        self.assertTrue(payload["plan_anchor_first_coefficient"])
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
            method_contract="full_freq_hrl_v3",
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
