import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from freq_hrl.experiments import pointmaze_plan_error as experiment
from scripts import pointmaze_plan_error_stage53_spec as spec
from scripts.submit_pointmaze_plan_error_stage53_scheduleurm import task_specification


class PlanErrorTest(unittest.TestCase):
    def fixture(self, measurement, reference, achieved):
        target = measurement[:, :2]
        distance = np.linalg.norm(achieved - target, axis=1).astype(np.float64)
        reward = np.exp(-distance)
        raw = {"measurement": measurement, "target_before": target.copy(), "lower_reference": reference,
               "achieved_after": achieved, "distance": distance, "reward": reward}
        row = {"episode_return": float(reward.sum()),
               "tracking_squared_error_integral": float(np.dot(distance, distance) * spec.source.DT_SECONDS),
               "reference_target_squared_error_integral": float(np.square(reference.astype(np.float64)
                   - target.astype(np.float64)).sum() * spec.source.DT_SECONDS)}
        return raw, row

    def test_signed_identity_and_pre_action_reward_alignment(self):
        measurement = np.tile(np.array([0, 1, 0, 0, 0, 0], dtype=np.float32), (6, 1))
        reference = np.tile(np.array([-1, 0], dtype=np.float32), (6, 1))
        achieved = np.tile(np.array([1, 2], dtype=np.float32), (6, 1))
        raw, row = self.fixture(measurement, reference, achieved)
        values = experiment.decompose(raw, row, measurement)
        np.testing.assert_allclose(values[:, :4], np.tile([.02, .08, -.08, .02], (6, 1)), atol=1e-12)
        wrong = copy.deepcopy(raw)
        wrong["target_before"][:, 0] += .5
        with self.assertRaisesRegex(AssertionError, "pre-action target"):
            experiment.decompose(wrong, row, measurement)
        wrong = measurement.copy()
        wrong[2, 0] = 1
        with self.assertRaisesRegex(AssertionError, "exogenous path"):
            experiment.decompose(raw, row, wrong)

    def test_regime_effect_is_next_target_increment_not_change_clock(self):
        target = np.column_stack(([0., 1., 2., 1., 0., -1., -2., -3.], np.zeros(8)))
        labels = experiment.event_partitions(target, (2,), 3)
        np.testing.assert_array_equal(labels["timing"][:3], [0, 0, 0])
        np.testing.assert_array_equal(labels["event"][:3], [0, 0, 0])
        np.testing.assert_array_equal(labels["timing"][3:6], [1, 1, 1])
        np.testing.assert_array_equal(labels["event"][3:6], [1, 1, 1])
        geometry = experiment.event_partitions(target, (), 3)
        np.testing.assert_array_equal(geometry["event"][3:6], [2, 2, 2])

    def test_geometry_timing_disjoint_partitions_and_empty_buckets(self):
        target = np.array([[0., 0.], [1, 0], [1, 1], [1, 2], [1, 1], [1, 0]])
        labels = experiment.event_partitions(target, (3,), 3)
        np.testing.assert_array_equal(labels["event"][:3], [2, 2, 2])
        np.testing.assert_array_equal(labels["event"][3:], [3, 3, 3])
        np.testing.assert_array_equal(labels["timing"][:3], [2, 2, 2])
        np.testing.assert_array_equal(labels["timing"][3:], [3, 3, 3])
        np.testing.assert_array_equal(labels["phase"], [0, 1, 2, 0, 1, 2])
        values = np.arange(30, dtype=np.float64).reshape(6, 5)
        means, panels = experiment.reduce_paths([values], [labels])
        np.testing.assert_array_equal(list(means.values()), values.sum(axis=0))
        for buckets in panels.values():
            self.assertEqual(sum(b["steps"] for b in buckets.values()), 6)
        self.assertEqual(panels["event"]["stable"]["steps"], 0)

    def test_signed_CIs_and_dynamic_offline_budget(self):
        signs = [-1., 1., 0., -2., 2., -3., 0., 4.]
        rows = [{"endpoints": dict(zip(spec.ENDPOINTS, signs))} for _ in range(8)]
        result = experiment.bootstrap(rows)
        self.assertEqual([r["effect"] for r in result.values()],
                         ["negative", "positive", "inconclusive", "negative", "positive", "negative", "inconclusive", "positive"])
        self.assertEqual(spec.budget(), {"raw_trace_reads": 3072, "recorded_steps_processed": 3686400,
            "exogenous_driver_regenerations": 128, "new_native_steps": 0, "optimizer_steps": 0})
        task = task_specification("unit_stage53")
        self.assertEqual(task["cpu"], 2)
        self.assertEqual(task["ram_mb"], 4096)
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03d}" for i in range(1, 7)])
        self.assertNotIn("result_dir", task)
        self.assertEqual(task["stage_input_paths"], [])

    def test_recorded_dataset_pipeline_and_missing_raw_path(self):
        root, seeds = 310001, [12103001, 12103002]
        args = spec.source.arguments(root, preflight=True)
        args.horizon = 100
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(spec.source, "roots", return_value=(root,)), \
                patch.object(spec.source, "arguments", return_value=args), \
                patch.object(spec.source, "options", return_value={"evaluation_paths": 2}), \
                patch.object(spec.source, "seed_roles", return_value={"evaluation": seeds}):
            directory = Path(directory)
            (directory / "qualification_summary.json").write_text(json.dumps({"status": "complete",
                "protocol": spec.source.EXPERIMENT_PROTOCOL, "contract": spec.source.contract(), "root_rows": [{"root": root}]}))
            expected = {}
            for seed in seeds:
                driver = experiment.PointMazeRegimeDriver(seed=seed, horizon=args.horizon,
                    dt_seconds=spec.source.DT_SECONDS, **experiment._task_options(args))
                expected[seed] = np.array([np.concatenate(driver.sample(t)) for t in range(args.horizon)])
            cell = {"status": "complete", "root": root, "evaluation_rows": {}}
            for period in spec.source.PERIODS:
                cell["evaluation_rows"][str(period)] = {}
                for index, policy in enumerate(spec.source.POLICIES):
                    cell["evaluation_rows"][str(period)][policy] = {}
                    for mode in spec.source.MODES:
                        rows = []
                        for seed in seeds:
                            measurement = expected[seed]
                            raw, row = self.fixture(measurement, measurement[:, :2] + np.float32(index * .1),
                                measurement[:, :2] + np.array([.2, -.1], dtype=np.float32))
                            row["seed"] = seed
                            rows.append(row)
                            raw_path = directory / "cells" / f"replicate_{root}_raw" / str(period) / policy / mode / f"episode_{seed}.npz"
                            raw_path.parent.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(raw_path, **raw)
                        cell["evaluation_rows"][str(period)][policy][mode] = rows
            result_path = directory / "cells" / f"replicate_{root}" / "result.json"
            result_path.parent.mkdir(parents=True)
            result_path.write_text(json.dumps(cell))
            summary = experiment.diagnose(directory)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["cost"]["raw_trace_reads"], 48)
            self.assertEqual(summary["cost"]["recorded_steps_processed"], 4800)
            self.assertEqual(summary["cost"]["new_native_steps"], 0)
            self.assertEqual(len(summary["strata"]), 12)
            raw_path.unlink()
            with self.assertRaises(FileNotFoundError):
                experiment.diagnose(directory)
