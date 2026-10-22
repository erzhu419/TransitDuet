import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.domains.mujoco import PointMazeRegimeDriver
from freq_hrl.domains.mujoco.reacher_tracking import ReacherTrackingDriver
from freq_hrl.experiments import plan_validity_reliability as reliability
from freq_hrl.experiments import plan_validity_transfer as shared
from freq_hrl.experiments import pointmaze_budgeted_trigger as trigger
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from freq_hrl.rl import GoalConditionedActorCriticPPO, GoalConditionedPPOConfig
from scripts import run_plan_validity_reliability as runner
from scripts import analyze_plan_validity_reliability as analyzer
from test_plan_validity_second_task import Controller, DenseTask, options
from test_pointmaze_timing_baselines import fitting_rows


def cell_report(root, risk_margin=0.01):
    paths = [1000 * root + 101, 1000 * root + 102]
    effects = {"local_advantage": 0.5, "opportunity_gap": -0.2, "tail_advantage": -0.5,
               "full_advantage": -0.2, "first_200_step_advantage": 0.1}
    return {"status": "complete", "protocol": {"task": "pointmaze", "optimizer_seed": root,
        "evaluation_seeds": paths, "risk_spec": {"distance": 1, "duration_steps": 25},
        "risk_rate_margin": risk_margin, "tail_ise_margin": 0,
        "family_alpha": reliability.FAMILY_ALPHA,
        "registered_comparator_method": "fixed", "comparator_eligible": True,
        "comparison": {"candidate_method": "history", "comparator_method": "fixed"}},
        "profiles": {"in_distribution": {"status": "complete", "episodes": [
            {"seed": seed, "candidate": {"tracking_squared_error_integral": 3,
                "reliability": {"sustained_failure": False}},
             "comparator": {"tracking_squared_error_integral": 4,
                "reliability": {"sustained_failure": False}}} for seed in paths],
            "local_errors": {"deployed": {"noise_corrected_conditional_mean_mse": 2,
                                            "mse_minus_zero_predictor": 0.2},
                             "reference": {"noise_corrected_conditional_mean_mse": 1}},
            "probes": [{"conditional_replicas": [effects.copy(), effects.copy()]}]}}}


def aggregate(reports, roots=(1, 2, 3), **kwargs):
    return reliability.aggregate_reliability(reports, expected_roots=roots, tasks=("pointmaze",),
        profiles=("in_distribution",), risk_rate_margin=kwargs.get("risk_rate_margin", 0.01), tail_ise_margin=0)


class ReliabilityTest(unittest.TestCase):
    def test_sparse_risk_records_long_runs_not_one_event_per_step(self):
        distances = np.array([0, 2, 2, 2, 2, 0, 2, 2], dtype=float)
        summary = reliability.episode_risk(distances ** 2 * 0.01, distances, [False] * 7 + [True],
            dt=0.01, period=4, distance=1, duration_steps=3)
        self.assertEqual(summary["sustained_failure_events"], 1)
        self.assertEqual(summary["longest_excursion_steps"], 4)
        self.assertAlmostEqual(summary["exceedance_seconds"], 0.06)
        self.assertEqual(summary["clipped_control_steps"], 1)
        self.assertAlmostEqual(sum(summary["bin_ise"]), 0.24)
        self.assertAlmostEqual(summary["first_half_ise"] + summary["second_half_ise"], 0.24)

    def test_zero_observed_events_do_not_prove_zero_risk(self):
        upper = reliability.event_upper_bound(0, 16)
        self.assertAlmostEqual(upper, 1 - 0.05 ** (1 / 16))
        self.assertGreater(upper, 0.1)
        result = aggregate([cell_report(root) for root in [1, 2, 3]])
        self.assertLess(result["panels"][0]["candidate_minus_comparator_ise"]["interval"][1], 0)
        self.assertFalse(result["joint_reliability_supported"])
        self.assertEqual(result["panels"][0]["root_risk_counts"][0]["harmful_discordances"], 0)

    def test_local_good_full_bad_decomposition_retains_opportunity_and_tail(self):
        base = {"prefix": np.array([1, 2]), "features": np.array([3, 4])}
        keep, renew = {**base, "branch_ise": 4}, {**base, "branch_ise": 3}
        now = {**base, "window_ise": 1, "tail_ise": 5, "first_200_step_ise": 3,
               "tracking_squared_error_integral": 8}
        wait = {**base, "window_ise": 1.5, "tail_ise": 2.5, "first_200_step_ise": 2,
                "tracking_squared_error_integral": 6}
        effect = reliability.decompose_probe(keep, renew, now, wait)
        self.assertEqual(effect["local_advantage"], 1)
        self.assertEqual(effect["opportunity_gap"], -0.5)
        self.assertEqual(effect["tail_advantage"], -2.5)
        self.assertEqual(effect["full_advantage"], -2)
        self.assertTrue(effect["local_to_full_sign_flip"])
        wait["prefix"] = np.array([2, 3])
        with self.assertRaisesRegex(RuntimeError, "causal state"):
            reliability.decompose_probe(keep, renew, now, wait)

    def test_fixed_clock_distribution_uses_only_fitting_bank_scale(self):
        row = lambda seed, step, value: {"seed": seed, "opportunity_step": step,
            "feature_names": ["x"], "causal_features": [value]}
        fit = [row(1, 50, -1), row(1, 100, 1)]
        ref = [row(2, 50, 0), row(2, 150, 0)]
        deployed = [row(2, 50, 0), row(2, 150, 4)]
        result = reliability.state_distribution(fit, ref, deployed, horizon=300)
        self.assertEqual(result["panels"]["first_half"]["paired_standardized_state_rms"], 0)
        self.assertEqual(result["panels"]["second_half"]["paired_standardized_state_rms"], 4)
        self.assertEqual(result["panels"]["second_half"]["deployed_outside_fit_range_fraction"], 1)
        with self.assertRaisesRegex(ValueError, "same unique"):
            reliability.state_distribution(fit, ref, deployed[::-1], horizon=300)

    def test_conditional_mse_correction_and_zero_predictor_comparison(self):
        rows = [{"deployed_score": 2, "local_advantage": 2,
                 "conditional_replicas": [{"local_advantage": 1}, {"local_advantage": 3}]}]
        result = reliability.local_error_summary(rows, "deployed")
        self.assertEqual(result["mse_to_conditional_repeat_mean"], 0)
        self.assertEqual(result["noise_corrected_conditional_mean_mse"], -1)
        self.assertEqual(result["mse_minus_zero_predictor"], -4)

    def test_roots_not_probes_are_statistical_units_and_missing_roots_block(self):
        reports = [cell_report(root, 0.99) for root in [1, 2, 3]]
        for report in reports:
            report["profiles"]["in_distribution"]["probes"] *= 100
        result = aggregate(reports, risk_rate_margin=0.99)
        panel = result["panels"][0]
        self.assertEqual(panel["candidate_minus_comparator_ise"]["n"], 3)
        self.assertEqual(panel["local_transfer_diagnostics"]["full_advantage"]["n"], 3)
        self.assertTrue(result["joint_reliability_supported"])
        missing = aggregate(reports[:2], risk_rate_margin=0.99)
        self.assertEqual(missing["panels"][0]["missing_or_failed_roots"], [3])
        self.assertFalse(missing["joint_reliability_supported"])
        reports[2]["profiles"]["in_distribution"]["episodes"].pop()
        with self.assertRaisesRegex(ValueError, "path roster"):
            aggregate(reports, risk_rate_margin=0.99)

    def test_analysis_cannot_relax_frozen_margins_or_drop_a_failed_profile(self):
        reports = [cell_report(root) for root in [1, 2, 3]]
        with self.assertRaisesRegex(ValueError, "frozen case plan"):
            aggregate(reports, risk_rate_margin=0.5)
        reports[1]["profiles"]["in_distribution"] = {"status": "failed", "error": "fixture"}
        result = aggregate(reports)
        self.assertEqual(result["panels"][0]["missing_or_failed_roots"], [2])
        self.assertFalse(result["joint_reliability_supported"])

    def test_ineligible_comparator_is_retained_but_cannot_close_the_claim(self):
        reports = [cell_report(root, 0.99) for root in (1, 2, 3)]
        reports[1]["protocol"]["comparator_eligible"] = False
        result = aggregate(reports, risk_rate_margin=0.99)
        self.assertEqual(result["panels"][0]["optimizer_roots"], 3)
        self.assertFalse(result["panels"][0]["comparator_eligible"])
        self.assertFalse(result["joint_reliability_supported"])

    def test_profiles_change_only_the_declared_exogenous_stress(self):
        original = _task_options(options())
        fast = reliability.profile_options(original, "fast_regime")
        tail = reliability.profile_options(original, "force_tail")
        self.assertEqual(original["force_pulse_amplitude"], fast["force_pulse_amplitude"])
        self.assertEqual(tail["regime_dwell_seconds"], original["regime_dwell_seconds"])
        self.assertEqual(tail["force_pulse_amplitude"], 3 * original["force_pulse_amplitude"])

    def test_conditional_future_preserves_both_plants_past_without_native_steps(self):
        for driver_type in (PointMazeRegimeDriver, ReacherTrackingDriver):
            driver = driver_type(seed=71, horizon=1200, dt_seconds=0.01)
            future = driver.conditional_future(step=400, seed=902)
            for step in (0, 50, 399, 400):
                np.testing.assert_array_equal(np.concatenate(driver.sample(step)), np.concatenate(future.sample(step)))
            self.assertFalse(np.array_equal(np.concatenate(driver.sample(700)), np.concatenate(future.sample(700))))

    def test_frozen_controller_round_trip_loads_without_training(self):
        args = options()
        args.reference_hidden_dim = 8
        dim = 6 + 6 * shared.time_contract(args).history_steps
        model = GoalConditionedActorCriticPPO(GoalConditionedPPOConfig(upper_state_dim=dim,
            lower_state_dim=dim, goal_dim=2, action_dim=2, hidden_dim=8))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "controller.pt"
            torch.save(model.state_dict(), path)
            record = {"configuration": vars(args), "controller_checkpoint": str(path)}
            loaded, _ = runner.load_controller(record)
            for name in ("upper_actor", "lower_actor", "upper_value", "lower_value"):
                for old, new in zip(getattr(model, name).parameters(), getattr(loaded, name).parameters()):
                    self.assertTrue(torch.equal(old, new))

    def test_default_plan_does_not_load_sources_or_execute(self):
        command = ["--candidate-report", "/not-yet-created/history.json", "--candidate-run-id", "history-root1",
            "--comparator-report", "/not-yet-created/fixed.json", "--comparator-run-id", "fixed-root1",
            "--comparator-method", "fixed",
            "--task", "pointmaze", "--optimizer-seed", "1", "--evaluation-seeds", "101", "102",
            "--probe-seeds", "101", "--risk-distance", "2", "--risk-rate-margin", "0.01", "--tail-ise-margin", "0"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plan.json"
            with patch.object(runner, "run_case", side_effect=AssertionError("no execution")):
                runner.main([*command, "--output", str(path)])
            plan = json.loads(path.read_text())
            self.assertEqual(plan["status"], "prepared")
            self.assertEqual(plan["profiles"], {})
            self.assertFalse(plan["protocol"]["scheduler_submission"])
            self.assertEqual(plan["protocol"]["controller_training"], "none_load_frozen_artifacts_only")
            with patch.object(runner, "run_case", side_effect=RuntimeError("fixture interrupted work")):
                self.assertEqual(runner.main([*command, "--execute", "--output", str(path)]), 1)
            failed = json.loads(path.read_text())
            self.assertEqual(failed["status"], "failed")
            self.assertGreater(failed["evaluation_cost"]["cpu_seconds"], 0)
            with self.assertRaises(FileExistsError):
                runner.main([*command, "--execute", "--output", str(path)])

    def test_shared_kernel_and_conditional_future_replays_with_frozen_fake_controller(self):
        args = options()
        spec = shared.PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id)
        fit = trigger.fit_budgeted_trigger_predictors(fitting_rows(), alpha_grid=[1])
        name = "causal_validity_interactions"
        rule = TimingRule(name, "legacy", view=name, threshold=fit[name]["threshold"], model=fit[name])
        risk_spec = {"distance": 1, "duration_steps": 25}
        with (
            patch.object(shared, "_make_task", side_effect=lambda **kw: DenseTask(**{k: v for k, v in kw.items() if k != "env_id"})),
            patch.object(shared, "pointmaze_goal_bounds", return_value=(np.full(2, -2), np.full(2, 2))),
        ):
            factual = shared.run_episode(Controller(), task_spec=spec, seed=71, args=args,
                time_scale=shared.time_contract(args), rule=rule)[1]
            capture = shared.run_episode(Controller(), task_spec=spec, seed=71, args=args,
                time_scale=shared.time_contract(args), rule=rule, feature_steps=(50, 100, 150, 200, 250),
                risk_spec=risk_spec)[1]
            self.assertEqual(factual["decision_steps"], capture["decision_steps"])
            self.assertEqual(factual["tracking_squared_error_integral"], capture["tracking_squared_error_integral"])
            result = reliability.evaluate_profile(Controller(), Controller(), rule, TimingRule("fixed", "clock"),
                task_spec=spec, args=args, seeds=[71], probe_seeds=[71], fit_rows=capture["feature_rows"],
                probes_per_episode=1, risk_spec=risk_spec, conditional_replicates=2)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["episodes"][0]["candidate"]["upper_decision_count"], 6)
        probe = result["probes"][0]
        self.assertEqual(probe["prefix_max_abs_difference"], 0)
        self.assertEqual(len(probe["conditional_replicas"]), 2)
        for effect in probe["conditional_replicas"]:
            self.assertAlmostEqual(effect["full_advantage"], effect["local_advantage"] +
                                   effect["opportunity_gap"] + effect["tail_advantage"])

    def test_analyzer_consumes_compact_reports_only(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "case.json", Path(folder) / "summary.json"
            source.write_text(json.dumps(cell_report(1)))
            analyzer.main(["--inputs", str(source), "--expected-roots", "1", "--task", "pointmaze",
                "--comparator-method", "fixed",
                "--risk-rate-margin", "0.01", "--tail-ise-margin", "0", "--output", str(output)])
            result = json.loads(output.read_text())
            self.assertEqual(result["statistical_unit"], "optimizer_root")
            self.assertFalse(result["joint_reliability_supported"])


if __name__ == "__main__":
    unittest.main()
