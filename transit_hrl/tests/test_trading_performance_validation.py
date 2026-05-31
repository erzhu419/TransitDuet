import unittest

from freq_hrl.experiments.trading.performance_validation import (
    BASELINES,
    SCENARIOS,
    make_synthetic_market,
    run_baseline,
)
from freq_hrl.experiments.trading.encoder_ablation import run_encoder_ablation
from freq_hrl.experiments.trading.policy_entry import (
    run_eval,
    run_actor_critic_episode,
    run_pg_episode,
    train_actor_critic,
    train_linear_policy,
    train_policy_gradient,
)
from freq_hrl.experiments.trading.pressure_test_matrix import summarize as summarize_pressure
from freq_hrl.experiments.trading.promotion_recovery_validation import (
    aggregate_variants,
    paired_deltas as promotion_paired_deltas,
)
from freq_hrl.policies import ActorCriticTradingParams, PolicyGradientTradingParams


class TradingPerformanceValidationTest(unittest.TestCase):
    def test_validation_run_returns_task_and_frequency_metrics(self):
        row = run_baseline(seed=11, baseline="freq_hrl", steps=80, n_assets=2)
        for key in [
            "total_return",
            "sharpe",
            "max_drawdown",
            "turnover",
            "base_mean_reward",
            "leakage_reward_penalty",
            "freq_attr_low_frequency_cost",
            "freq_attr_high_frequency_cost",
            "freq_attr_leakage_cost",
            "freq_attr_promotion_adaptation_cost",
            "promotion_delay",
            "PromotionDelay",
            "ShockResponseTime",
            "regime_promotion_accuracy",
            "recovery_cost_120",
            "recovery_regret_120",
            "post_shift_cum_pnl_120",
            "UpperHFPower",
            "LowerLFDrift",
            "FocusScore",
        ]:
            self.assertIn(key, row)

    def test_no_leakage_baseline_disables_reward_penalty(self):
        row = run_baseline(seed=11, baseline="no_leakage", steps=80, n_assets=2)
        self.assertEqual(row["leakage_reward_penalty_total"], 0.0)

    def test_baseline_matrix_runs(self):
        for baseline in BASELINES:
            with self.subTest(baseline=baseline):
                row = run_baseline(seed=7, baseline=baseline, steps=100, n_assets=2)
                self.assertEqual(row["baseline"], baseline)

    def test_scenario_matrix_generates_all_pressure_cases(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                data = make_synthetic_market(seed=3, steps=100, n_assets=2, scenario=scenario)
                self.assertEqual(data["returns"].shape, (100, 2))
                row = run_baseline(
                    seed=3,
                    baseline="freq_hrl",
                    steps=100,
                    n_assets=2,
                    scenario=scenario,
                )
                self.assertEqual(row["scenario"], scenario)

    def test_state_space_and_wavelet_encoders_run_in_trading_validation(self):
        for method in ("state_space", "haar_wavelet"):
            with self.subTest(method=method):
                row = run_baseline(
                    seed=5,
                    baseline="freq_hrl",
                    steps=60,
                    n_assets=2,
                    freq_method=method,
                )
                self.assertEqual(row["freq_method"], method)
                self.assertIn("sharpe", row)

    def test_encoder_ablation_entry_aggregates_methods(self):
        rows, summary = run_encoder_ablation(
            seeds=[1],
            steps=40,
            assets=2,
            scenario="persistent_shift",
            methods=["ema", "state_space"],
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["freq_method"] for row in summary}, {"ema", "state_space"})

    def test_pressure_summary_ignores_encoder_name_field(self):
        rows = [
            run_baseline(seed=1, baseline="freq_hrl", steps=40, n_assets=2),
            run_baseline(seed=1, baseline="no_promotion", steps=40, n_assets=2),
        ]
        summary = summarize_pressure(rows)
        self.assertEqual({row["baseline"] for row in summary}, {"freq_hrl", "no_promotion"})

    def test_promotion_recovery_validation_pairs_regret_direction(self):
        rows = []
        for seed in [1, 2]:
            rows.append({
                "variant": "no_promotion",
                "baseline": "no_promotion",
                "seed": seed,
                "scenario": "promotion_recovery",
                "freq_method": "ema",
                "sharpe": 1.0,
                "total_return": 0.10,
                "post_shift_cum_pnl_120": 0.20,
                "recovery_regret_120": 0.30,
            })
            rows.append({
                "variant": "recovery_tuned",
                "baseline": "freq_hrl",
                "seed": seed,
                "scenario": "promotion_recovery",
                "freq_method": "ema",
                "sharpe": 1.5,
                "total_return": 0.12,
                "post_shift_cum_pnl_120": 0.25,
                "recovery_regret_120": 0.20,
            })
        paired = promotion_paired_deltas(
            rows,
            reference="no_promotion",
            metrics=["post_shift_cum_pnl_120", "recovery_regret_120"],
            n_boot=100,
            seed=7,
        )
        tuned = next(row for row in paired if row["variant"] == "recovery_tuned")
        summary = aggregate_variants(rows)
        self.assertEqual(next(row for row in summary if row["variant"] == "recovery_tuned")["n"], 2)
        self.assertAlmostEqual(tuned["post_shift_cum_pnl_120_delta_mean"], 0.05)
        self.assertAlmostEqual(tuned["recovery_regret_120_delta_mean"], -0.10)
        self.assertEqual(tuned["post_shift_cum_pnl_120_win_rate"], 1.0)
        self.assertEqual(tuned["recovery_regret_120_win_rate"], 1.0)

    def test_learned_linear_policy_entry_trains_and_evaluates(self):
        model = train_linear_policy(
            train_seeds=[42],
            steps=50,
            assets=2,
            scenario="persistent_shift",
            generations=1,
            population=2,
            elite_frac=0.5,
            seed=1,
        )
        self.assertEqual(model["policy"], "linear")
        self.assertIn("params", model)
        row = run_eval(seed=123, steps=50, assets=2, policy="linear")
        self.assertIn("sharpe", row)

    def test_policy_gradient_entry_trains_and_evaluates(self):
        model = train_policy_gradient(
            train_seeds=[42],
            steps=40,
            assets=2,
            scenario="persistent_shift",
            iterations=1,
            learning_rate=0.01,
            seed=1,
        )
        self.assertEqual(model["policy"], "pg_linear")
        self.assertIn("params", model)
        row = run_eval(seed=123, steps=40, assets=2, policy="pg_linear")
        self.assertIn("sharpe", row)

    def test_policy_gradient_leakage_regularizer_enters_policy_loss(self):
        _, row = run_pg_episode(
            PolicyGradientTradingParams(),
            seed=42,
            steps=40,
            assets=2,
            scenario="persistent_shift",
            rng_seed=7,
            leakage_policy_loss_scale=0.01,
            leakage_constraint_threshold=0.0,
            leakage_lagrange_multiplier=0.1,
        )
        self.assertIn("policy_loss_leakage_penalty", row)
        self.assertIn("leakage_constraint_violation", row)
        self.assertGreaterEqual(row["policy_loss_leakage_penalty"], 0.0)

        model = train_policy_gradient(
            train_seeds=[42],
            steps=30,
            assets=2,
            scenario="persistent_shift",
            iterations=1,
            learning_rate=0.01,
            seed=1,
            leakage_policy_loss_scale=0.01,
            leakage_constraint_threshold=0.0,
            leakage_lagrange_lr=0.1,
        )
        self.assertEqual(model["trainer"], "on_policy_reinforce_leakage_constrained")
        self.assertIn("leakage_lagrange_multiplier", model["history"][0])

    def test_actor_critic_entry_trains_and_evaluates(self):
        actor_grad, upper_grad, lower_grad, row = run_actor_critic_episode(
            ActorCriticTradingParams(),
            seed=42,
            steps=35,
            assets=2,
            scenario="persistent_shift",
            rng_seed=7,
        )
        self.assertEqual(actor_grad.shape[0], len(PolicyGradientTradingParams.trainable_names()))
        self.assertEqual(upper_grad.shape[0], ActorCriticTradingParams.upper_value_dim())
        self.assertEqual(lower_grad.shape[0], ActorCriticTradingParams.lower_value_dim())
        self.assertIn("td_error_abs_mean", row)

        model = train_actor_critic(
            train_seeds=[42],
            steps=35,
            assets=2,
            scenario="persistent_shift",
            iterations=1,
            actor_learning_rate=0.01,
            critic_learning_rate=0.01,
            seed=1,
        )
        self.assertEqual(model["policy"], "ac_linear")
        self.assertEqual(model["trainer"], "td0_actor_critic")
        self.assertIn("critic_value_loss", model["history"][0])
        row = run_eval(seed=123, steps=35, assets=2, policy="ac_linear")
        self.assertIn("sharpe", row)


if __name__ == "__main__":
    unittest.main()
