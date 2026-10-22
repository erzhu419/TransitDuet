from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.experiments import pointmaze_adaptive_pair_diagnostic as native
from freq_hrl.experiments import pointmaze_timing_baselines as timing
from scripts import run_pointmaze_timing_baselines as runner
from test_pointmaze_adaptive_pair_diagnostic import ToyController, ToyHistory, ToyTask


def fitting_rows():
    rng = np.random.default_rng(305)
    names = tuple(dict.fromkeys((*timing.VIEWS["current"], *timing.VIEWS["history"])))
    x = rng.normal(size=(72, len(names)))
    return [{"seed": 1 + index // 24, "feature_names": names,
             "causal_features": values.tolist(), "offset_fraction": 0.4, "remaining_fraction": 0.75,
             "renew_ise_advantage": float(values[0] - 0.3 * values[1]),
             "timing_advantage": float(values[names.index("target_error_0")]),
             "candidate_feature_has_future_access": False,
             "candidate_feature_has_regime_label": False}
            for index, values in enumerate(x)]


def args(**overrides):
    result = Namespace(
        optimizer_seed=305001, horizon=300, env_id="unused-by-toy",
        upper_period_seconds=0.5, history_seconds=0.64, fast_period_seconds=0.04,
        maximum_subgoal_delta=0.75, max_offset_steps=25, check_stride_steps=5,
        pairs_per_path=2, regime_dwell_seconds=[0.5, 1.5], target_speed_modes=[0.1, 0.2],
        force_pulse_amplitude=0.1, force_pulse_duration_seconds=[0.1, 0.2],
        force_pulse_gap_seconds=[0.4, 0.6], distractor_amplitude=0.1,
        distractor_dwell_seconds=[0.1, 0.3],
    )
    vars(result).update(overrides)
    return result


class TimingBaselinesTest(unittest.TestCase):
    def test_capacity_is_real_features_and_matching_effective_df_not_zero_padding(self):
        rows = fitting_rows()
        rules = timing.fit_capacity_controls(rows, effective_df=4)
        for rule in rules:
            self.assertEqual(rule.model["coefficient_count"], 171)
            self.assertAlmostEqual(rule.model["effective_df_slopes"], 4, places=7)
            self.assertEqual(len(set(rule.model["feature_names"])), 170)
            values = np.asarray(rows[0]["causal_features"])
            self.assertTrue(np.isfinite(rule.score(rows[0]["feature_names"], values, 0.4, 0.75)))
        self.assertNotEqual(rules[0].model["feature_names"], rules[1].model["feature_names"])

    def test_future_or_regime_features_are_rejected(self):
        rows = fitting_rows()
        for flag in ("candidate_feature_has_future_access", "candidate_feature_has_regime_label"):
            rows[0][flag] = True
            with self.assertRaises(ValueError):
                timing.feature_view(rows, "history", quadratic=False)
            rows[0][flag] = False

    def test_thresholds_use_fit_inputs_not_label_signs(self):
        rows = fitting_rows()
        before = [r.to_dict() for r in timing.threshold_candidates(rows)]
        for row in rows:
            row["renew_ise_advantage"] = 1e9
            row["timing_advantage"] = -1e9
        self.assertEqual(before, [r.to_dict() for r in timing.threshold_candidates(rows)])
        self.assertEqual([r.threshold for r in timing.threshold_candidates(rows) if r.kind == "clock"],
                         [0, 0.2, 0.4, 0.6, 0.8, 1])

    def test_learned_policy_can_stop_and_wait_at_same_offset(self):
        rows = fitting_rows()
        rng_before = torch.get_rng_state().clone()
        for view in ("current", "history"):
            rule = timing.fit_termination(rows, view=view, seed=305, hidden_dim=16, epochs=180)
            scores = np.array([rule.score(r["feature_names"], np.asarray(r["causal_features"]), 0.4, 0.75)
                               for r in rows])
            truth = np.array([r["timing_advantage"] for r in rows])
            self.assertGreater(np.mean((scores >= 0) == (truth > 0)), 0.95)
            self.assertTrue(np.any(scores > 0) and np.any(scores < 0))
            self.assertEqual(rule.model["parameter_count"], 609)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))

    def test_credit_magnitude_not_positive_label_count_determines_stop(self):
        rows = fitting_rows()[:3]
        for row, advantage in zip(rows, [1.0, 1.0, -10.0]):
            row["causal_features"] = rows[0]["causal_features"]
            row["timing_advantage"] = advantage
        rule = timing.fit_termination(rows, view="current", seed=17, epochs=120, hidden_dim=4)
        self.assertLess(rule.score(rows[0]["feature_names"], np.asarray(rows[0]["causal_features"]), 0.4, 0.75), 0)

    def test_collapsed_best_checkpoint_is_retained_not_replaced_by_active_one(self):
        collapsed = timing.TimingRule("collapsed", "clock", threshold=1)
        active = timing.TimingRule("active", "clock", threshold=0.4)
        selected, _ = timing.select_rule([collapsed, active], lambda rule: [{
            "tracking_squared_error_integral": 1 if rule.name == "collapsed" else 2}])
        self.assertEqual(selected.name, "collapsed")
        rows = [{"seed": 1, "decision_steps": [0, 75, 125], "queries": [
            {"step": 50, "offset": 0, "stop": False}], "tracking_squared_error_integral": 1}]
        status = timing.termination_readiness(rows, rows)
        self.assertFalse(status["behaviorally_noncollapsed"])
        self.assertFalse(status["eligible_for_fresh_confirmation"])

    def test_clock_variation_across_bins_is_not_state_conditioning(self):
        rows = [{"seed": seed, "decision_steps": [0, 50, 125],
                 "queries": [{"step": 50, "offset": 0, "stop": True},
                             {"step": 100, "offset": 0, "stop": False}],
                 "tracking_squared_error_integral": 1} for seed in [1, 2]]
        clocks = [{**row, "tracking_squared_error_integral": 2} for row in rows]
        status = timing.termination_readiness(rows, clocks)
        self.assertTrue(status["behaviorally_noncollapsed"])
        self.assertFalse(status["state_conditional_at_same_bin_and_offset"])
        self.assertFalse(status["eligible_for_fresh_confirmation"])
        rows[1]["queries"][0]["stop"] = False
        status = timing.termination_readiness(rows, clocks)
        self.assertTrue(status["state_conditional_at_same_bin_and_offset"])

    def test_shared_native_runner_budget_prefix_and_adaptive_deferral(self):
        options = args()
        time_scale = PhysicalTimeScaleContract(dt_seconds=0.01, upper_period_seconds=0.5,
                                               history_seconds=0.64, fast_period_seconds=0.04)
        rule = timing.TimingRule("clock_5", "clock", threshold=0.2)
        with (
            patch.object(native, "_make_task", side_effect=lambda **kw: ToyTask()),
            patch.object(native, "pointmaze_goal_bounds", return_value=(np.full(2, -10), np.full(2, 10))),
            patch.object(native, "PointMazeRegimeFeatureBuilder", ToyHistory),
            patch.object(native, "_causal_plan_features", side_effect=lambda **kw: (
                ("tracking_distance",), kw["observation"].task_measurement.copy(), None)),
        ):
            factual = timing.rollout_rule(ToyController(), rule, seed=1, args=options, time_scale=time_scale)
            now, wait = [timing.rollout_rule(ToyController(), rule, seed=1, args=options,
                                             time_scale=time_scale, intervention_step=55, arm=arm)
                         for arm in ("now", "wait_one_check")]
            self.assertEqual(factual["decision_steps"], [0, 55, 105, 155, 205, 255])
            self.assertEqual(now["decision_steps"][1], 55)
            self.assertEqual(wait["decision_steps"][1], 60)
            self.assertEqual(wait["decision_steps"][2:], factual["decision_steps"][2:])
            np.testing.assert_array_equal(now["prefix"], wait["prefix"])
            pairs, steps = timing.collect_timing_pairs(ToyController(), [rule, rule], seeds=[1],
                                                       args=options, time_scale=time_scale, round_index=0)
            self.assertEqual(len(pairs), 2)
            self.assertEqual(steps, (1 + 2 * 2) * 300)
            for pair in pairs:
                self.assertEqual(pair["timing_advantage"], float(pair["timing_advantage"]))

    def test_plan_default_never_trains_or_samples_and_role_overlap_fails(self):
        command = ["--optimizer-seed", "305001", "--train-seeds", "1", "2", "3",
                   "--selection-seeds", "4", "--branch-fit-seeds", "5", "6", "7",
                   "--timing-selection-seeds", "8", "--trigger-eval-seeds", "9"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prepared.json"
            with patch.object(runner, "run_cell", side_effect=AssertionError("must not execute")):
                self.assertEqual(runner.main([*command, "--output", str(output)]), 0)
                result = json.loads(output.read_text())
                self.assertEqual(result["status"], "prepared")
                self.assertFalse(result["protocol"]["scheduler_submission"])
                self.assertEqual(result["cells"], [])
                self.assertEqual(list(Path(directory).iterdir()), [output])
            with self.assertRaisesRegex(ValueError, "distinct"):
                runner.main([*command, "--trigger-eval-seeds", "5", "--output", str(output)])

    def test_development_suite_never_executes_reserved_test_paths(self):
        options = args(
            train_seeds=[11, 12], selection_seeds=[21], branch_fit_seeds=[31, 32, 33],
            timing_selection_seeds=[41, 42], trigger_eval_seeds=[51, 52], iterations=2,
            reference_hidden_dim=4, learning_rate=3e-4, checkpoint_evaluation_interval=1,
            max_events_per_class=1, ridge_alpha_grid=[1.0], threshold_quantile=0.75,
            effective_df=2, timing_rounds=2, termination_epochs=2, termination_hidden_dim=4)
        rows = fitting_rows()
        for row in rows:
            row.update(keep_primitive_steps_replayed=10, renew_primitive_steps_replayed=10)
        events = []

        def controller_training(**kwargs):
            self.assertEqual(kwargs["eval_seeds"], [41, 42])
            return {"selected_checkpoint_iteration": 1, "capacity": {}, "runtime_versions": {}}, object()

        def fit(*args, **kwargs):
            events.append("fit")
            self.assertNotIn("test", events)
            return timing.fit_termination(*args, **kwargs)

        def rollout(controller, rule, *, seed, **kwargs):
            self.assertIn(seed, [41, 42])
            return {"seed": seed, "mode": rule.name, "decision_steps": [0, 75, 125, 175, 225, 275],
                    "tracking_squared_error_integral": 1.0, "queries": [],
                    "episode_length": 300, "upper_decision_count": 6}

        with (
            patch.object(runner, "train_pointmaze_plan_value_cell", side_effect=controller_training),
            patch.object(runner, "evaluate_plan_renewal_dataset", return_value=rows),
            patch.object(runner, "collect_timing_pairs", return_value=(rows, 100)),
            patch.object(runner, "fit_termination", side_effect=fit),
            patch.object(runner, "rollout_rule", side_effect=rollout),
        ):
            result = runner.run_cell(options)
        self.assertNotIn("test_rows", result)
        self.assertFalse(result["test_accessed"])
        self.assertEqual(events.count("fit"), 4)
        self.assertEqual(result["ledger"]["test_steps"], 0)
        self.assertEqual(result["ledger"]["timing_pair_replay_steps"], 200)
        self.assertFalse(result["performance_claim_supported"])
        self.assertTrue(all(not state["eligible_for_fresh_confirmation"]
                            for state in result["termination_readiness"].values()))


if __name__ == "__main__":
    unittest.main()
