from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments import pointmaze_root_response as experiment
from freq_hrl.experiments import pointmaze_plan_hold as hold
from freq_hrl.experiments import pointmaze_forecast_response as response
from freq_hrl.experiments import pointmaze_separate_motion as motion
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready, _training_seed
from scripts import pointmaze_root_response_stage33_spec as spec
from scripts import pointmaze_budgeted_trigger_stage9_spec as old
from scripts import submit_pointmaze_root_response_stage33_scheduleurm as submit
from test_pointmaze_forecast_response import ImmediatePool, motion_models
from test_pointmaze_history_information import NAMES, measured_sequences
from test_pointmaze_temporal_plan import FakeController, FakeTask


def root_result(root):
    fit, cases, _ = spec.response_cases(root, preflight=False)
    rows = []
    for i, case in enumerate(cases):
        rate = 1. if i % 2 else -1.
        rows.append({**case, "curve": np.asarray(hold.HORIZONS) * .01 * rate,
                     "predicted_rates": {m: np.full(5, rate if m == "history" else 0.) for m in response.METHODS}})
    return {"status": "complete", "protocol": {"protocol_version": spec.EXPERIMENT_PROTOCOL,
            "phase": "response", "optimizer_seed": root,
            "options": _json_ready(spec.options(root, preflight=False)), "qualification": spec.qualification_contract()},
            "cells": [{"seed_roles": spec.seed_roles(root, preflight=False), "training_pairs": len(fit), "rows": rows}]}


class RootResponseTest(unittest.TestCase):
    def test_frozen_roles_and_actual_training_seeds_are_independent(self):
        old_seeds = set()
        for root in (*old.PREFLIGHT_OPTIMIZER_SEEDS, *old.OPTIMIZER_SEEDS, *old.CONFIRMATION_OPTIMIZER_SEEDS):
            roles = old.seed_roles(root)
            old_seeds.update(seed for values in roles.values() for seed in values)
            old_seeds.update(_training_seed(optimizer_seed=root, rollout_root=seed, iteration=i)
                             for seed in roles["train"] for i in range(old.ITERATIONS))
        used = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                args = spec.arguments(root, preflight=preflight)
                flat = [seed for values in spec.seed_roles(root, preflight=preflight).values() for seed in values]
                flat += [_training_seed(optimizer_seed=root, rollout_root=seed, iteration=i)
                         for seed in args.train_seeds for i in range(args.iterations)]
                self.assertEqual(len(flat), len(set(flat)))
                self.assertFalse(set(flat) & old_seeds)
                self.assertFalse(set(flat) & used)
                used.update(flat)
                hold.validate_paths(args, {"fit": args.response_fit, "evaluation": args.response_eval},
                                    {"temporal_seed_roles": {"motion_fit": args.motion_fit, "motion_eval": args.motion_eval}})

    def test_controller_settings_match_original_not_just_its_architecture(self):
        for preflight in (True, False):
            old_options = old.cell_options(208001 if preflight else 209011, preflight=preflight)
            new = spec.options(spec.roots(preflight=preflight)[0], preflight=preflight)
            for key, value in old_options.items():
                if key not in ("train", "selection", "branch_fit", "trigger_eval", "methods"):
                    self.assertEqual(new[key], value)

    def test_full_roster_and_exact_budget(self):
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                args = spec.arguments(root, preflight=preflight)
                fit, query, excluded = spec.response_cases(root, preflight=preflight)
                budget = spec.budget(root, preflight=preflight)
                self.assertEqual(len(query), 2 if preflight else 120)
                self.assertEqual(len(fit) + excluded, 4 if preflight else 320)
                self.assertEqual(budget["controller_total_primitive_steps"], 5100 if preflight else 4243200)
                self.assertEqual(budget["scalar_rhs_count"], 125)
                for case in fit + query:
                    check = case["check_step"]
                    self.assertGreaterEqual(check, 64)
                    self.assertLessEqual(check + 150, args.horizon)
                    renew, keep = hold.pair_schedules(case["seed"], check, args.horizon)
                    self.assertEqual(renew[:-1], keep[:-1])
                    self.assertEqual((renew[-1], keep[-1]), (check, check + 100))
                self.assertEqual(budget["qualification_total_primitive_steps"],
                                 args.horizon + sum(2 * (c["check_step"] + 150) for c in fit + query))

    def test_split_fit_is_original_ridge_and_never_reads_query_outcomes(self):
        models = motion_models()
        rng = np.random.default_rng(33)
        rows = [{"seed": i + 1, "check_step": 100} for i in range(20)]
        examples = [{**r, "sequence": x, "feature_names": NAMES,
                     "curve": rng.normal(size=5), "candidate_plan": rng.normal(size=2)}
                    for r, x in zip(rows, measured_sequences(rows))]
        critics, designs = experiment.fit_response(examples[:16], models, root=33)
        pred, query_designs = experiment.predict_response(examples[16:], models, critics, root=33)
        expected, fits, original_designs = response.fit_response(examples[:16], examples[16:], models, root=33)
        changed = deepcopy(examples[16:])
        for row in changed:
            row.pop("curve")
        other, _ = experiment.predict_response(changed, models, critics, root=33)
        for method in response.METHODS:
            np.testing.assert_array_equal(pred[method], expected[method])
            np.testing.assert_array_equal(pred[method], other[method])
            np.testing.assert_array_equal(critics[method].fitted["weights"], fits[method]["weights"])
            np.testing.assert_array_equal(designs[method], original_designs[method][0])
            np.testing.assert_array_equal(query_designs[method], original_designs[method][1])

    def test_whole_root_bootstrap_and_frozen_joint_gate(self):
        results = [root_result(root) for root in spec.OPTIMIZER_ROOTS]
        aggregate = experiment.root_aggregate(results)
        self.assertTrue(aggregate["qualification_passed"])
        self.assertEqual(len(aggregate["endpoints"]), 15)
        self.assertEqual(aggregate["endpoints"]["ise:always_keep"]["simultaneous_ci95"], [.75, .75])
        self.assertEqual(aggregate["qualification"]["statistical_unit"], "optimizer_seed_root")
        self.assertEqual(aggregate, experiment.root_aggregate(list(reversed(results))))
        # One controller root, not its 120 opportunities, is the resampling unit.
        for row in results[0]["cells"][0]["rows"]:
            row["predicted_rates"]["history"] *= -100
        self.assertFalse(experiment.root_aggregate(results)["qualification_passed"])
        with self.assertRaisesRegex(ValueError, "eight-root roster"):
            experiment.root_aggregate(results[:-1])
        with self.assertRaisesRegex(ValueError, "eight-root roster"):
            experiment.root_aggregate(results[:-1] + [results[0]])
        changed = deepcopy(results)
        changed[0]["cells"][0]["rows"].pop()
        with self.assertRaisesRegex(ValueError, "missing or substituted"):
            experiment.root_aggregate(changed)

    def test_pipeline_freezes_heads_before_query_and_keeps_failed_motion_root(self):
        args = spec.arguments(310001, preflight=True)
        controller = FakeController()
        events = []
        original_sample, original_fit = experiment.sample_pairs, experiment.fit_response
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory / "cell" / "result.json"
            trained = {"selected_checkpoint_iteration": 1, "world_low": [-2., -2.], "world_high": [2., 2.]}

            def sample(*values, label):
                events.append(label)
                return original_sample(*values, label=label)

            def fit(*values, **kwargs):
                self.assertEqual(events, ["fit"])
                events.append("fitted")
                return original_fit(*values, **kwargs)

            with patch.object(experiment, "load_controller", return_value=(controller, trained, {})), \
                    patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment, "sample_pairs", side_effect=sample), \
                    patch.object(experiment, "fit_response", side_effect=fit), \
                    patch.object(motion, "motion_metrics", return_value={"motion_gate_passed": False}), \
                    patch.object(hold, "_make_task", side_effect=lambda **kwargs: FakeTask()), \
                    patch.object(hold, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
                cell = experiment.qualify(args, directory / "controller.json", output)
            self.assertEqual(events, ["fit", "fitted", "query"])
            self.assertFalse(cell["motion"]["metrics"]["motion_gate_passed"])
            self.assertEqual(cell["evaluation_pairs"], 2)
            self.assertEqual(cell["controller_updates"], 0)
            self.assertEqual(cell["budget"]["candidate_proposal_inference_calls"],
                             cell["training_pairs"] + cell["evaluation_pairs"])
            raw = Path(cell["raw_server_directory"])
            with np.load(raw / "response.npz", allow_pickle=False) as data:
                curves = np.cumsum(data["query_step_ise"][:, 1] - data["query_step_ise"][:, 0], axis=1)[:, np.array(hold.HORIZONS) - 1]
                np.testing.assert_allclose(curves, data["query_curve"], atol=1e-12)
            self.assertTrue((raw / "response_fits.json").is_file())
            self.assertEqual(json.loads(output.read_text())["status"], "complete")

    def test_scheduler_resources_are_split_and_dynamic(self):
        train = submit.task_specification("unit_train", 310011, preflight=False, phase="train")
        preflight = submit.task_specification("unit_preflight", 310001, preflight=True, phase="pipeline")
        self.assertEqual((train["cpu"], train["ram_mb"]), (1, 3072))
        self.assertEqual((preflight["cpu"], preflight["ram_mb"]), (2, 3072))
        for task in (train, preflight):
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node00{i}" for i in range(1, 7)])
            self.assertFalse(any("_raw" in path for path in task["stage_input_paths"]))
        with TemporaryDirectory() as temporary, patch.object(submit, "ROOT", Path(temporary)):
            source = submit.cell_dir("train", 310011) / "result.json"
            experiment.write_json(source, {"status": "complete", "protocol": {"phase": "train", "optimizer_seed": 310011,
                "protocol_version": spec.EXPERIMENT_PROTOCOL, "options": spec.options(310011, preflight=False)}})
            task = submit.task_specification("unit_response", 310011, preflight=False, phase="response", controller_run="train")
            self.assertEqual((task["cpu"], task["ram_mb"]), (17, 24576))
            self.assertIn(str(source.parent), task["stage_input_paths"])


if __name__ == "__main__":
    unittest.main()
