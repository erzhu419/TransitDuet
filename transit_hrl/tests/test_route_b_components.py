"""Unit tests for Route B components: ReacherPolygonal, Generic-MLP, TEMPLE, OC, Native-Periodic, and CEM."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from freq_hrl.domains.mujoco.reacher_polygonal import ReacherPolygonalSpec
from freq_hrl.experiments.plan_validity_transfer import run_episode, time_contract
from freq_hrl.experiments.plan_validity_unprivileged_and_pareto import (
    BudgetScheduleSpec,
    collect_unprivileged_renewal_labels,
    run_multi_budget_episode,
)
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from freq_hrl.experiments.route_b_baselines import (
    CEMSubgoalControllerWrapper,
    RouteBTimingRule,
    fit_deliberation_option_critic_rule,
    fit_generic_mlp_validity_predictor,
    fit_temple_gate_rule,
    train_native_periodic_controller,
)


def _smoke_args() -> SimpleNamespace:
    return SimpleNamespace(
        env_id="PointMaze_UMaze-v3",
        optimizer_seed=77001,
        iterations=2,
        horizon=300,
        checkpoint_evaluation_interval=1,
        reference_hidden_dim=32,
        learning_rate=3e-4,
        upper_period_seconds=0.50,
        history_seconds=0.64,
        fast_period_seconds=0.04,
        maximum_subgoal_delta=0.50,
        branch_window_seconds=0.50,
        max_offset_steps=25,
        check_stride_steps=5,
        train_seeds=[81001],
        selection_seeds=[82001],
        branch_fit_seeds=[83001, 83002],
        timing_selection_seeds=[84001],
        regime_dwell_seconds=(0.80, 1.60),
        target_speed_modes=(-1.35, -0.65, 0.65, 1.35),
        force_pulse_amplitude=0.18,
        force_pulse_duration_seconds=(0.04, 0.10),
        force_pulse_gap_seconds=(0.45, 1.10),
        distractor_amplitude=0.50,
        distractor_dwell_seconds=(0.35, 0.85),
    )


class RouteBComponentTests(unittest.TestCase):
    def test_reacher_polygonal_and_all_route_b_methods_execute_cleanly(self):
        args = _smoke_args()
        scale = time_contract(args)
        spec = ReacherPolygonalSpec(_task_options(args))

        payload, periodic_ctrl = train_native_periodic_controller(spec, args, scale)
        self.assertEqual(payload["policy"], "native_periodic_goal_conditioned_ppo")

        labels = collect_unprivileged_renewal_labels(
            periodic_ctrl,
            acquisition="uniform",
            task_spec=spec,
            args=args,
            time_scale=scale,
            queries_per_path=6,
        )
        mlp_rule, oof = fit_generic_mlp_validity_predictor(
            labels, seed=77001, hidden_dim=16, epochs=30
        )
        self.assertEqual(len(oof), len(labels))
        reloaded = RouteBTimingRule.from_dict(mlp_rule.to_dict())
        _, ep_mlp = run_episode(
            periodic_ctrl, task_spec=spec, seed=85001, args=args, time_scale=scale, rule=reloaded
        )
        self.assertEqual(ep_mlp["upper_decision_count"], 6)

        temple_rule, _ = fit_temple_gate_rule(
            periodic_ctrl, task_spec=spec, args=args, time_scale=scale, seed=77001, rounds=2, hidden_dim=16
        )
        _, ep_temple = run_episode(
            periodic_ctrl, task_spec=spec, seed=85001, args=args, time_scale=scale, rule=temple_rule
        )
        self.assertEqual(ep_temple["upper_decision_count"], 6)

        oc_rule, _ = fit_deliberation_option_critic_rule(
            periodic_ctrl, task_spec=spec, args=args, time_scale=scale, seed=77001, rollout_episodes=6, hidden_dim=16
        )
        _, ep_oc = run_episode(
            periodic_ctrl, task_spec=spec, seed=85001, args=args, time_scale=scale, rule=oc_rule
        )
        self.assertEqual(ep_oc["upper_decision_count"], 6)

        cem_ctrl = CEMSubgoalControllerWrapper(
            periodic_ctrl,
            population_size=32,
            elites=8,
            cem_iters=3,
            maximum_subgoal_delta=args.maximum_subgoal_delta,
        )
        ep_cem = run_multi_budget_episode(
            cem_ctrl,
            task_spec=spec,
            seed=85001,
            args=args,
            feature_time_scale=scale,
            budget_spec=BudgetScheduleSpec(6, 50, 25, 5),
            rule=reloaded,
        )
        self.assertEqual(ep_cem["upper_decision_count"], 6)
        self.assertGreater(ep_cem["planner_wall_seconds"], ep_cem["trigger_score_wall_seconds"])


if __name__ == "__main__":
    unittest.main()
