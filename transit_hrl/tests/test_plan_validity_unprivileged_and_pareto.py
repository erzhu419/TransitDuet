"""Unit tests for unprivileged query acquisition and multi-budget Pareto execution."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from freq_hrl.experiments.plan_validity_transfer import (
    PlanValidityTaskSpec,
    jitter_schedule,
    run_episode,
    time_contract,
)
from freq_hrl.experiments.plan_validity_unprivileged_and_pareto import (
    BudgetScheduleSpec,
    collect_unprivileged_renewal_labels,
    default_pareto_specs,
    run_multi_budget_episode,
    unprivileged_observable_opportunities,
    unprivileged_uniform_opportunities,
)
from freq_hrl.experiments.pointmaze_budgeted_trigger import fit_budgeted_trigger_predictors
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from freq_hrl.rl import GoalConditionedActorCriticPPO, GoalConditionedPPOConfig


def _dummy_args(horizon: int = 300) -> SimpleNamespace:
    return SimpleNamespace(
        env_id="PointMaze_UMaze-v3",
        horizon=horizon,
        upper_period_seconds=0.50,
        history_seconds=0.64,
        fast_period_seconds=0.04,
        maximum_subgoal_delta=0.75,
        branch_window_seconds=0.50,
        max_offset_steps=25,
        check_stride_steps=5,
        branch_fit_seeds=[91001, 91002],
        regime_dwell_seconds=(0.80, 1.60),
        target_speed_modes=(-1.25, -0.55, 0.55, 1.25),
        force_pulse_amplitude=0.18,
        force_pulse_duration_seconds=(0.04, 0.10),
        force_pulse_gap_seconds=(0.45, 1.10),
        distractor_amplitude=0.50,
        distractor_dwell_seconds=(0.35, 0.85),
    )


def _dummy_controller(scale) -> GoalConditionedActorCriticPPO:
    import torch

    torch.manual_seed(12345)
    dim = 6 + 6 * scale.history_steps
    return GoalConditionedActorCriticPPO(
        GoalConditionedPPOConfig(
            upper_state_dim=dim,
            lower_state_dim=dim,
            goal_dim=2,
            action_dim=2,
            hidden_dim=32,
            learning_rate=3e-4,
            epochs=2,
            minibatch_size=128,
            init_log_std=-0.7,
        ),
    )


class UnprivilegedAndParetoTests(unittest.TestCase):
    def test_pareto_specs_validate_cleanly(self):
        specs = default_pareto_specs(1200)
        self.assertEqual([s.calls_per_episode for s in specs], [12, 15, 20, 24, 30, 40])

    def test_uniform_and_observable_queries_have_zero_privileged_access(self):
        args = _dummy_args(horizon=300)
        scale = time_contract(args)
        spec = PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id)
        controller = _dummy_controller(scale)
        sched = jitter_schedule(args, 91001)

        uni = unprivileged_uniform_opportunities(
            seed=91001, horizon=300, schedule=sched, count=6
        )
        self.assertEqual(len(uni), 6)
        self.assertTrue(all(0 < q.step < 250 and q.step not in sched for q in uni))
        self.assertTrue(all(q.source_step is None and q.lag_steps is None for q in uni))

        obs = unprivileged_observable_opportunities(
            controller,
            task_spec=spec,
            seed=91001,
            args=args,
            time_scale=scale,
            schedule=sched,
            count=6,
        )
        self.assertEqual(len(obs), 6)
        self.assertTrue(all(0 < q.step < 250 and q.step not in sched for q in obs))

    def test_b24_multi_budget_matches_canonical_run_episode_bit_for_bit(self):
        args = _dummy_args(horizon=300)
        scale = time_contract(args)
        spec = PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id)
        controller = _dummy_controller(scale)

        labels = collect_unprivileged_renewal_labels(
            controller,
            acquisition="uniform",
            task_spec=spec,
            args=args,
            time_scale=scale,
            queries_per_path=6,
        )
        self.assertEqual(len(labels), 12)
        fitted = fit_budgeted_trigger_predictors(
            labels, predictors=("causal_validity_interactions",)
        )["causal_validity_interactions"]
        legacy_rule = TimingRule(
            "causal_validity_interactions",
            "legacy",
            view="causal_validity_interactions",
            threshold=fitted["threshold"],
            model=fitted,
        )
        dist_rule = TimingRule("dist", "threshold", threshold=0.25, feature="tracking_distance")
        clock_rule = TimingRule("clock_10", "clock", threshold=10 / 25)

        b24 = BudgetScheduleSpec(calls_per_episode=6, bin_steps=50, max_offset_steps=25, check_stride_steps=5)
        for rule in (clock_rule, dist_rule, legacy_rule):
            _, ref = run_episode(
                controller, task_spec=spec, seed=92001, args=args, time_scale=scale, rule=rule
            )
            got = run_multi_budget_episode(
                controller,
                task_spec=spec,
                seed=92001,
                args=args,
                feature_time_scale=scale,
                budget_spec=b24,
                rule=rule,
            )
            self.assertEqual(got["decision_steps"], ref["decision_steps"])
            self.assertAlmostEqual(
                got["tracking_squared_error_integral"],
                ref["tracking_squared_error_integral"],
                places=12,
            )
            self.assertAlmostEqual(got["episode_return"], ref["episode_return"], places=12)

        for spec_other in (
            BudgetScheduleSpec(3, 100, 50, 10),
            BudgetScheduleSpec(10, 30, 15, 3),
        ):
            out = run_multi_budget_episode(
                controller,
                task_spec=spec,
                seed=92001,
                args=args,
                feature_time_scale=scale,
                budget_spec=spec_other,
                rule=legacy_rule,
            )
            self.assertEqual(out["upper_decision_count"], spec_other.calls_per_episode)
            self.assertTrue(out["protocol_valid"])


if __name__ == "__main__":
    unittest.main()
