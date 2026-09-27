from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.experiments import pointmaze_plan_hold as hold
from freq_hrl.experiments import pointmaze_temporal_plan as temporal
from scripts import pointmaze_plan_hold_spec as spec
from scripts.pointmaze_budgeted_trigger_stage9_spec import cell_options, seed_roles
from scripts.submit_pointmaze_deployed_pair_diagnostic_scheduleurm import task_specification
from test_pointmaze_history_information import NAMES, measured_sequences
from test_pointmaze_temporal_plan import FakeController, FakeTask


def arguments(root=208001):
    options = cell_options(root, preflight=root == 208001)
    return SimpleNamespace(**options, optimizer_seed=root,
                           **{role + "_seeds": list(options[role])
                              for role in ("train", "selection", "branch_fit", "trigger_eval")})


class RecordingController(FakeController):
    def __init__(self):
        self.goals = []

    def act_conditioned(self, state, sample):
        self.goals.append(state[:2].astype(float) + state[4:6])
        return super().act_conditioned(state, sample)


class PlanHoldTest(unittest.TestCase):
    def test_fresh_roles_grid_budget_and_derived_training_path_exclusion(self):
        all_new = set()
        for root in (208001, 209011, 209061):
            args = arguments(root)
            preflight = root == 208001
            roles = hold.path_roles(root, preflight=preflight)
            paths = roles["fit"] + roles["evaluation"]
            self.assertFalse(all_new.intersection(paths))
            all_new.update(paths)
            old = {"temporal_seed_roles": temporal.path_roles(root, preflight=preflight)}
            hold.validate_paths(args, roles, old)
            self.assertFalse(set(paths).intersection(s for values in seed_roles(root).values() for s in values))
            cases = hold.cases_for_paths(root, paths, horizon=args.horizon, pairs_per_path=2 if preflight else 20)
            self.assertEqual(len(cases), 8 if preflight else 480)
            self.assertEqual(sum(2 * (r["check_step"] + 150) for r in cases), 3760 if preflight else 657600)
            for seed in paths:
                checks = [r["check_step"] for r in cases if r["seed"] == seed]
                self.assertEqual(len(checks), len(set(checks)))
                if not preflight:
                    self.assertEqual([sum(c % 50 == o for c in checks) for o in (0, 5, 10, 15, 20)], [4] * 5)
                for check in checks:
                    renew, keep = hold.pair_schedules(seed, check, args.horizon)
                    self.assertEqual(renew[:-1], keep[:-1])
                    self.assertTrue(all(c < check for c in renew[:-1]))
                    self.assertEqual((renew[-1], keep[-1]), (check, check + 100))
            altered = deepcopy(roles)
            altered["fit"][0] = hold._training_seed(optimizer_seed=root, rollout_root=args.train_seeds[0], iteration=0)
            with self.assertRaisesRegex(ValueError, "paths overlap"):
                hold.validate_paths(args, altered, old)

    def collect(self, schedule, *, future_shift=0.):
        args, controller, task = arguments(), RecordingController(), FakeTask(future_shift)
        scale = PhysicalTimeScaleContract(dt_seconds=.01, upper_period_seconds=.5,
                                          history_seconds=.64, fast_period_seconds=.04)
        bounds = (np.array([-2., -2.]), np.array([2., 2.]))
        with patch.object(hold, "_make_task", return_value=task), patch.object(hold, "pointmaze_goal_bounds", return_value=bounds):
            row = hold.rollout_window(controller, seed=3279001, check=55, schedule=schedule,
                                      args=args, time_scale=scale)
        self.assertTrue(task.closed)
        self.assertEqual(len(controller.goals), 205)
        self.assertEqual(row["primitive_steps"], 205)
        return row, np.array(controller.goals)

    def test_old_waypoint_is_held_for_100_steps_and_late_call_executes_50_steps(self):
        renew_schedule, keep_schedule = hold.pair_schedules(3279001, 55, 300)
        renew, _ = self.collect(renew_schedule)
        keep, goals = self.collect(keep_schedule)
        np.testing.assert_allclose(goals[55:155], np.broadcast_to(goals[54], (100, 2)), atol=1e-7)
        np.testing.assert_allclose(goals[155:205], np.broadcast_to(goals[155], (50, 2)), atol=1e-7)
        self.assertFalse(np.allclose(goals[154], goals[155]))
        row = hold.combine_pair({"seed": 3279001, "check_step": 55}, renew, keep)
        self.assertEqual(row["sequence"].shape, (64, 23))
        self.assertEqual(row["upper_calls_at_horizons"], {"renew": [2]*5, "keep": [1]*4 + [2]})
        np.testing.assert_allclose(row["curve"], np.cumsum(keep["step_ise"] - renew["step_ise"])[np.array(hold.HORIZONS)-1])
        self.assertEqual(row["primitive_steps"], 410)
        changed, _ = self.collect(keep_schedule, future_shift=2.)
        np.testing.assert_equal(keep["sequence"], changed["sequence"])
        np.testing.assert_equal(keep["policy_prefix"], changed["policy_prefix"])
        self.assertFalse(np.array_equal(keep["step_ise"], changed["step_ise"]))
        self.assertTrue(np.all(keep["sequence"][-51:, -1] == 1))
        with self.assertRaisesRegex(RuntimeError, "call budget"):
            hold.combine_pair({"check_step": 55}, renew, {**keep, "calls": [0, 100, 155]})
        with self.assertRaisesRegex(RuntimeError, "before intervention"):
            hold.combine_pair({"check_step": 55}, renew, {**keep, "sequence": keep["sequence"] + 1})

    def test_curve_sign_units_and_gross_versus_settled_gates(self):
        rows = [{"curve": np.array(hold.HORIZONS)*.01*v,
                 "predicted_rates": {"history": [v]*5, "current_repeat": [0.]*5,
                                     "shuffled_history": [-v]*5}} for v in (1., -1.)]
        metrics = hold.summarize(rows)
        self.assertTrue(metrics["prediction_gate_passed"])
        self.assertTrue(metrics["decision_gate_passed"])
        self.assertEqual(metrics["history_settled_ise_benefit_vs_control"],
                         {"current_repeat": .75, "shuffled_history": 1.5, "always_keep": .75, "always_renew": .75})
        settled_only = deepcopy(rows)
        for row in settled_only:
            row["predicted_rates"]["history"][:4] = [0.]*4
        self.assertFalse(hold.summarize(settled_only)["prediction_gate_passed"])
        rows[0]["predicted_rates"]["current_repeat"][-1] = 1.
        self.assertFalse(hold.summarize(rows)["decision_gate_passed"])

    def test_query_labels_and_query_features_do_not_affect_fitted_weights(self):
        rows = [{"seed": i+1, "check_step": 100, "role": "fit" if i < 4 else "evaluation"}
                for i in range(6)]
        x = measured_sequences(rows)
        rng = np.random.default_rng(28)
        examples = [{**r, "sequence": seq, "feature_names": NAMES, "curve": rng.normal(size=5)}
                    for r, seq in zip(rows, x)]
        predictions, fits = hold.fit_curves(examples[:4], examples[4:], root=28)
        query = deepcopy(examples[4:])
        for row in query:
            row["curve"] *= 1000
        changed, changed_fits = hold.fit_curves(examples[:4], query, root=28)
        self.assertEqual(fits, changed_fits)
        for method in hold.METHODS:
            np.testing.assert_equal(predictions[method], changed[method])
            self.assertEqual((fits[method]["parameter_count"], fits[method]["scalar_linear_solves"]), (160, 5))
            self.assertEqual(fits[method]["alpha"], 1.)
        query[0]["sequence"][:, 0] += 1000
        _, feature_fits = hold.fit_curves(examples[:4], query, root=28)
        self.assertEqual(fits, feature_fits)
        with self.assertRaisesRegex(ValueError, "paths overlap"):
            hold.fit_curves(examples[:4], examples[:1], root=28)

    def test_cached_controller_is_loaded_without_training_and_factual_mismatch_stops_sampling(self):
        with TemporaryDirectory() as temporary:
            args = arguments()
            args.source_result, args.controller_result = Path(temporary)/"source.json", Path(temporary)/"original.json"
            cache = {"controller_selected_iteration": 1, "raw_server_directory": temporary}
            original = {"controller_selected_iteration": 1, "aligned_candidate_rows": [{"seed": 1,
                        "decision_steps": [0, 75, 125, 175, 225, 275], "episode_return": 3.,
                        "tracking_squared_error_integral": 4.}]}
            for path, protocol, cell in ((args.source_result, hold.CACHE_PROTOCOL, cache),
                                        (args.controller_result, hold.WINDOWED_PROTOCOL_VERSION, original)):
                path.write_text(json.dumps({"status": "complete", "protocol": {"protocol_version": protocol,
                                "optimizer_seed": 208001}, "cells": [cell]}))
            controller = SimpleNamespace(config=SimpleNamespace(hidden_dim=128), load_state_dict=lambda state: None)
            checkpoint = {"optimizer_seed": 208001, "selected_iteration": 1, "arguments": vars(args).copy(),
                          "state_dict": {"config": controller.config.__dict__}}
            with patch.object(hold.torch, "load", return_value=checkpoint), \
                    patch.object(hold, "pointmaze_plan_value_dimensions", return_value=None), \
                    patch.object(hold, "build_pointmaze_plan_value_model", return_value=(controller, {})), \
                    patch.object(hold, "rollout_timing_schedule", return_value=original["aligned_candidate_rows"][0]) as replay:
                loaded = hold.load_controller(args)
                self.assertIs(loaded[2], controller)
                self.assertEqual(loaded[-1]["absolute_errors"], {"episode_return": 0., "tracking_squared_error_integral": 0.})
                replay.return_value = {**original["aligned_candidate_rows"][0], "episode_return": 4.}
                with self.assertRaisesRegex(RuntimeError, "frozen factual"):
                    hold.load_controller(args)
                checkpoint["selected_iteration"] = 2
                with self.assertRaisesRegex(ValueError, "source checkpoint"):
                    hold.load_controller(args)
                self.assertEqual(replay.call_count, 2)

    def test_scheduler_stages_only_small_source_metadata_on_the_dynamic_pool(self):
        for preflight in (True, False):
            root = spec.roots(preflight=preflight)[0]
            task = task_specification("unit_hold", root, preflight=preflight, protocol_spec=spec)
            self.assertEqual((task["cpu"], task["ram_mb"]), (2, 3072) if preflight else (17, 24576))
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node00{i}" for i in range(1, 7)])
            self.assertIn("--controller-result", task["cmd"])
            self.assertNotIn("curve-epochs", task["cmd"])
            self.assertFalse(any("_raw" in p for p in task["stage_input_paths"]))


if __name__ == "__main__":
    unittest.main()
