from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.mujoco import PointMazeRegimeDriver
from freq_hrl.experiments import pointmaze_history_information as probe
from freq_hrl.experiments import pointmaze_temporal_plan as temporal
from scripts import pointmaze_history_information_spec as spec
from scripts.submit_pointmaze_deployed_pair_diagnostic_scheduleurm import task_specification


NAMES = [f"{name}_{i}" for name, size in (("physical", 4), ("achieved", 2), ("target_error", 2),
         ("waypoint_error", 2), ("measured", 6), ("previous_action", 2)) for i in range(size)] + [
    "plan_age_fraction", "within_bin_fraction", "remaining_fraction", "budget_spent", "valid_observation"]


def measured_sequences(rows, *, horizon=300):
    sequences = []
    for row in rows:
        driver = PointMazeRegimeDriver(seed=row["seed"], horizon=horizon, dt_seconds=.01)
        steps = np.arange(row["check_step"]-63, row["check_step"]+1)
        x = np.zeros((64, 23), dtype=np.float32)
        x[:, 10:16] = [np.concatenate(driver.sample(int(max(0, t)))) for t in steps]
        x[:, -1] = steps >= 0
        sequences.append(x)
    return np.stack(sequences)


def cache_fixture(directory):
    root = 208001
    roles = temporal.path_roles(root, preflight=True)
    rows = [dict(row, role=role) for role, paths in roles.items()
            for row in temporal.cases_for_paths(root, paths, horizon=300, pairs_per_path=2)]
    x = measured_sequences(rows)
    curves = np.arange(len(rows)*3, dtype=float).reshape(len(rows), 3) * .001
    raw = directory / "replicate_208001_raw"
    raw.mkdir()
    np.savez_compressed(raw / "temporal_pairs.npz", sequences=x, curves=curves, feature_names=NAMES,
                        seeds=[r["seed"] for r in rows], check_steps=[r["check_step"] for r in rows],
                        roles=[r["role"] for r in rows])
    source = {"status": "complete", "protocol": {"protocol_version": temporal.PROTOCOL_VERSION,
              "optimizer_seed": root}, "cells": [{"controller_selected_iteration": 2,
              "raw_server_directory": str(raw), "feature_names": NAMES, "temporal_seed_roles": roles,
              "training_pairs": 4, "evaluation_pairs": 4,
              "controller_reconstruction_primitive_steps": {"training": 1200},
              "factual_replay_primitive_steps": 300, "temporal_replay_primitive_steps": 2000,
              "coverage": {role: {str(s): {str(o): sum(r["seed"] == s and r["check_step"] % 50 == o
                           for r in rows) for o in (0, 5, 10, 15, 20)} for s in seeds}
                           for role, seeds in roles.items()},
              "rows": [{"seed": r["seed"], "check_step": r["check_step"], "curve": y.tolist()}
                       for r, y in zip(rows, curves) if r["role"] == "evaluation"]}]}
    controller = {"status": "complete", "protocol": {"protocol_version": probe.WINDOWED_PROTOCOL_VERSION,
                  "optimizer_seed": root, "horizon": 300, "task_options": {}},
                  "cells": [{"controller_selected_iteration": 2}]}
    source_path, controller_path = directory / "source.json", directory / "controller.json"
    source_path.write_text(json.dumps(source))
    controller_path.write_text(json.dumps(controller))
    return source_path, controller_path, x, curves, rows


class HistoryInformationTest(unittest.TestCase):
    def test_current_frame_is_identical_and_history_slopes_use_observed_lags(self):
        rows = [{"seed": 1, "check_step": 100, "role": "fit"}]
        x = measured_sequences(rows)
        history = probe.causal_design(x, rows, NAMES, method="history", root=1)
        current = probe.causal_design(x, rows, NAMES, method="current_repeat", root=1)
        shuffled = probe.causal_design(x, rows, NAMES, method="shuffled_history", root=1)
        np.testing.assert_equal(history[:, :23], current[:, :23])
        np.testing.assert_equal(history[:, :23], shuffled[:, :23])
        np.testing.assert_equal(current[:, 23:], np.zeros((1, 8)))
        for i, lag in enumerate(probe.LAGS):
            np.testing.assert_allclose(history[:, 23+2*i:25+2*i],
                                       (x[:, -1, 10:12].astype(float)-x[:, -1-lag, 10:12])/(lag*.01))
        changed = x.copy()
        changed[:, -51, -1] = 0
        with self.assertRaisesRegex(ValueError, "unpadded"):
            probe.causal_design(changed, rows, NAMES, method="history", root=1)

    def test_future_only_changes_labels_and_wrong_prefix_is_rejected(self):
        rows = [{"seed": 11, "check_step": 75, "role": "fit"}]
        x = measured_sequences(rows)
        protocol = {"horizon": 300, "task_options": {}}
        labels, count = probe.target_motion_labels(x, rows, NAMES, controller_protocol=protocol)
        self.assertEqual(count, 301)
        driver = PointMazeRegimeDriver(seed=11, horizon=300, dt_seconds=.01)
        before = probe.causal_design(x, rows, NAMES, method="history", root=11)
        driver._targets[76:] += 2.
        with patch.object(probe, "PointMazeRegimeDriver", return_value=driver):
            changed, _ = probe.target_motion_labels(x, rows, NAMES, controller_protocol=protocol)
        self.assertFalse(np.array_equal(labels, changed))
        np.testing.assert_equal(before, probe.causal_design(x, rows, NAMES, method="history", root=11))
        driver._targets[75] += 1.
        with patch.object(probe, "PointMazeRegimeDriver", return_value=driver):
            with self.assertRaisesRegex(ValueError, "prefix differs"):
                probe.target_motion_labels(x, rows, NAMES, controller_protocol=protocol)

    def test_cached_source_roster_and_labels_are_matched_before_fitting(self):
        with TemporaryDirectory() as temporary:
            source_path, controller_path, x, curves, rows = cache_fixture(Path(temporary))
            *_, loaded_x, loaded_y, loaded_rows, names, cache_path = probe.load_cache(
                source_path, controller_path, root=208001)
            np.testing.assert_equal(x, loaded_x)
            np.testing.assert_equal(curves, loaded_y)
            self.assertEqual(rows, loaded_rows)
            self.assertEqual(names, NAMES)
            self.assertEqual(cache_path.name, "temporal_pairs.npz")
            source = json.loads(source_path.read_text())
            source["cells"][0]["rows"][0]["curve"][0] += 1
            source_path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, "labels differ"):
                probe.load_cache(source_path, controller_path, root=208001)

    def test_query_labels_and_features_do_not_change_fitted_parameters(self):
        rows = [{"seed": i+1, "check_step": 100, "role": "fit" if i < 4 else "evaluation"}
                for i in range(6)]
        x = measured_sequences(rows)
        rng = np.random.default_rng(27)
        y = {"target_motion": rng.normal(size=(6, 6)), "timing_response": rng.normal(size=(6, 3))}
        predictions, fits = probe.fit_probes(x, y, rows, NAMES, root=27)
        other_y = {key: value.copy() for key, value in y.items()}
        for value in other_y.values():
            value[4:] += 1000
        changed, other_fits = probe.fit_probes(x, other_y, rows, NAMES, root=27)
        self.assertEqual(fits, other_fits)
        for method in temporal.METHODS:
            for key in probe.OBJECTIVES:
                np.testing.assert_equal(predictions[method][key], changed[method][key])
                self.assertEqual(fits[method][key]["alpha"], 1.)
                self.assertEqual(fits[method][key]["training_rows"], 4)
        changed_x = x.copy()
        changed_x[5, :, 0] += 1000
        _, changed_fits = probe.fit_probes(changed_x, y, rows, NAMES, root=27)
        self.assertEqual(fits, changed_fits)
        repeated = deepcopy(rows)
        repeated[-1]["seed"] = repeated[0]["seed"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            probe.fit_probes(x, y, repeated, NAMES, root=27)

    def test_timing_sign_and_rate_units_have_hand_computed_benefits(self):
        rows = []
        for sign in (1., -1.):
            rows.append({"target_motion": [sign]*6, "timing_response": [sign]*3,
                         "lag1_extrapolation": [sign]*6, "predictions": {
                             "history": {"target_motion": [sign]*6, "timing_response": [sign]*3},
                             "current_repeat": {"target_motion": [0.]*6, "timing_response": [0.]*3},
                             "shuffled_history": {"target_motion": [-sign]*6, "timing_response": [-sign]*3}}})
        metrics = probe.summarize(rows)
        self.assertEqual(metrics["target_motion"]["rate_mse"],
                         {"history": 0., "current_repeat": 1., "shuffled_history": 4., "zero": 1., "lag1_extrapolation": 0.})
        self.assertEqual(metrics["timing_response"]["history_ise_benefit_vs_control"],
                         {"current_repeat": .25, "shuffled_history": .5, "always_wait": .25, "always_now": .25})

    def test_run_cell_uses_no_controller_weights_or_environment(self):
        with TemporaryDirectory() as temporary:
            source_path, controller_path, *_ = cache_fixture(Path(temporary))
            cell = probe.run_cell(SimpleNamespace(source_result=source_path, controller_result=controller_path,
                                                 optimizer_seed=208001))
            self.assertEqual((cell["training_rows"], cell["evaluation_rows"]), (4, 4))
            self.assertEqual(cell["linear_fits"], 6)
            self.assertEqual(cell["scalar_linear_solves"], 27)
            self.assertEqual(cell["new_environment_primitive_steps"], 0)
            self.assertEqual(cell["controller_updates"], 0)
            self.assertEqual(cell["optimizer_updates"], 0)
            self.assertEqual(cell["generated_exogenous_tape_points"], 1204)
            self.assertEqual(len(cell["path_metrics"]), 2)

    def test_scheduler_uses_dynamic_pool_and_stages_no_raw_data(self):
        for preflight in (True, False):
            root = spec.roots(preflight=preflight)[0]
            task = task_specification("unit_history_information", root, preflight=preflight, protocol_spec=spec)
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 1536))
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node00{i}" for i in range(1, 7)])
            self.assertIn("--controller-result", task["cmd"])
            self.assertFalse(any("_raw" in path for path in task["stage_input_paths"]))


if __name__ == "__main__":
    unittest.main()
