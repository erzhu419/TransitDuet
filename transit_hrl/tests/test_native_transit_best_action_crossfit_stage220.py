import copy
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_best_action_learning as learning
from scripts import run_native_transit_best_action_crossfit_stage220 as spec
from scripts.submit_native_transit_best_action_crossfit_stage220_scheduleurm import task_specification
from tests import test_native_transit_best_action_stage218 as cohort
from tests import test_native_transit_coverage_value as native_fixture


def panels_for(rows):
    output = {}
    for row in rows:
        values = np.zeros((8, 6, 4))
        values[:, :, 0] = np.arange(6.)+1
        values[:, :, 2] = values[:, :, 0]
        values[:, :, 3] = .1
        values[:, row["original_choice"]] = 0.
        candidate = (row["original_choice"]+1) % 6
        values[:, candidate, 0] = values[:, candidate, 2] = -1.
        values[:, candidate, 3] = -.2
        output[learning.state_key(row)] = {"values": values}
    return output


def zero_prediction(rows):
    n = len(rows)
    originals = np.asarray([r["original_choice"] for r in rows])
    return {"costs": {k: np.zeros((n, 6)) for k in ("ridge", "best")},
        "wait_logits": {k: np.zeros((n, 6)) for k in ("glm", *learning.WAIT_VARIANTS)},
        "choices": {k: originals.copy() for k in learning.VARIANTS}}


class CrossfitTest(unittest.TestCase):
    def test_matched_splits_are_complementary_balanced_and_grouped(self):
        rows, registered = cohort.fixture()
        for name, view in spec.source.matched_views(rows, registered).items():
            a, b = spec.split_rows(view, 0)
            reverse_a, reverse_b = spec.split_rows(view, 1)
            keys = lambda rs: {learning.state_key(r) for r in rs}
            self.assertEqual(keys(a), keys(reverse_b))
            self.assertEqual(keys(b), keys(reverse_a))
            self.assertFalse(keys(a) & keys(b))
            for rs in (a, b):
                self.assertEqual(len(rs), 60)
                groups = {(r["root"], r["scene_seed"]) for r in rs}
                self.assertEqual(len(groups), 10)
                for group in groups:
                    self.assertEqual(sum((r["root"], r["scene_seed"]) == group for r in rs), 6)
                for regime in spec.source.prefixes.SCENARIOS:
                    self.assertEqual(sum(r["scenario"] == regime for r in rs), 12)
                if name.startswith("mixed"):
                    self.assertEqual(sum(r["root"] == 397 for r in rs), 30)

    def test_training_targets_ignore_other_half_and_preserve_draw_magnitudes(self):
        rows = cohort.fixture()[0][:2]
        panels = panels_for(rows)
        first, masses = learning.half_targets(rows, panels, (0, 1, 2, 3))
        changed = copy.deepcopy(panels)
        for row in rows:
            changed[learning.state_key(row)]["values"][4:, (row["original_choice"]+1) % 6, 3] = 100.
        second, other = learning.half_targets(rows, changed, (0, 1, 2, 3))
        self.assertEqual(first, second)
        self.assertEqual(masses, other)
        self.assertTrue(all(m["draw_count"] == 4 for m in masses.values()))
        self.assertEqual(masses[learning.state_key(rows[0])]["negative_wait_min"][1], .2)
        self.assertNotIn("labels", rows[0])

    def test_future_and_held_labels_do_not_change_fit_inputs_or_predictions(self):
        rows, registered = cohort.fixture()
        train, held = spec.split_rows(spec.source.matched_views(rows, registered)["mixed_a"], 0)
        panels = panels_for(rows)
        changed = copy.deepcopy(panels)
        for row in train:
            j = (row["original_choice"]+1) % 6
            changed[learning.state_key(row)]["values"][4:, j, 3] = 10.
        for row in held:
            j = (row["original_choice"]+1) % 6
            changed[learning.state_key(row)]["values"][:, j, 3] = 10.
        captured = []
        def fitted(rs, ms, batch):
            captured.append((copy.deepcopy(rs), copy.deepcopy(ms)))
            return {}, {}
        batch = lambda rs: (np.zeros((len(rs), 6, 198)), np.zeros((len(rs), 12)))
        with patch.object(learning.neural, "batches", side_effect=batch), patch.object(learning, "fit_heads", side_effect=fitted), patch.object(
                learning, "predict", side_effect=lambda rs, models, batch: zero_prediction(rs)):
            first = spec.fit_cell(train, held, panels, "first_last", 0, lambda m: None)
            second = spec.fit_cell(train, held, changed, "first_last", 0, lambda m: None)
        self.assertEqual(captured[0], captured[1])
        for scope in spec.SCOPES:
            self.assertEqual(len(first["scopes"][scope]), 60)
            self.assertEqual([[r["decisions"][v]["selected"] for v in learning.VARIANTS] for r in first["scopes"][scope]],
                [[r["decisions"][v]["selected"] for v in learning.VARIANTS] for r in second["scopes"][scope]])
        self.assertNotEqual(first["scopes"]["held_other"], second["scopes"]["held_other"])

    def test_real_feature_interfaces_fit_and_predict_without_future_menus(self):
        rows = native_fixture.bank_fixture()[0]["training_data"][:6]
        targets, masses = learning.half_targets(rows, panels_for(rows), (0, 1, 2, 3))
        with patch.object(learning, "STEPS", 3):
            models, fit = learning.fit_heads(targets, masses)
        self.assertEqual(fit["target_draw_count"], [4])
        self.assertEqual(fit["supervised_decision_updates"], 3)
        self.assertEqual(fit["wait_residual_updates"], 9)
        self.assertEqual([fit["fits"][k]["parameter_count"] for k in ("best", *learning.WAIT_VARIANTS)], [98, 199, 3201, 3393])
        self.assertAlmostEqual(fit["fits"]["best"]["initial_loss"], np.log(6))
        self.assertLess(fit["fits"]["best"]["final_loss"], fit["fits"]["best"]["initial_loss"])
        causal = [{k: v for k, v in r.items() if k != "labels"} for r in rows]
        predictions = learning.predict(causal, models)
        self.assertEqual(set(predictions["choices"]), set(learning.VARIANTS))
        for i, row in enumerate(rows):
            self.assertEqual(predictions["costs"]["best"][i, row["original_choice"]], 0.)
            self.assertTrue(all(z[i, row["original_choice"]] == 0. for z in predictions["wait_logits"].values()))

    def test_report_retains_disputed_states_and_rejects_missing_future_cells(self):
        rows, registered = cohort.fixture()
        views = spec.source.matched_views(rows, registered)
        results = {}
        panels = panels_for(rows)
        for view, rs in views.items():
            for fold in (0, 1):
                train, held = spec.split_rows(rs, fold)
                groups = lambda rs: sorted(map(list, {(r["root"], r["scene_seed"]) for r in rs}))
                cells = []
                for partition in spec.PARTITIONS:
                    for direction in (0, 1):
                        a, b = spec.PARTITIONS[partition][direction], spec.PARTITIONS[partition][1-direction]
                        fits = {k: {"parameter_count": p, "steps": learning.STEPS, "initial_loss": 1., "final_loss": .5,
                            "parameter_update_l2": 1.} for k, p in (("best", 98), ("linear", 199), ("nonlinear_pair", 3201), ("state_conditioned_pair", 3393))}
                        fits["state_conditioned_pair"]["context_weight_l2"] = 1.
                        scopes = {s: learning.records(train if s.startswith("train") else held, panels,
                            a if s.endswith("same") else b, b if s.endswith("same") else a,
                            zero_prediction(train if s.startswith("train") else held)) for s in spec.SCOPES}
                        cells.append({"partition": partition, "direction": direction, "training_draws": list(a), "evaluation_draws": list(b),
                            "training_groups": groups(train), "held_groups": groups(held), "scopes": scopes,
                            "fit": {"training_groups": groups(train), "training_states": 60, "target_draw_count": [4],
                                "supervised_decision_updates": learning.STEPS, "wait_residual_updates": 3*learning.STEPS,
                                "new_native_episodes": 0, "rl_actor_updates": 0, "fits": fits, "glm_iterations": 5}})
                results[view, fold] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
                    "view": view, "fold": fold, "cells": cells}
        report = spec.summarize(results, views)
        for value in report["views"].values():
            self.assertEqual(value["scopes"]["held_other"]["unique_states"], 120)
            self.assertEqual(value["scopes"]["held_other"]["prediction_cases"], 480)
        for compared in report["single_vs_mixed"].values():
            self.assertEqual(compared["prediction_cases"], 480)
            self.assertEqual(compared["unique_states"], 120)
            self.assertTrue(all(v["cost_delta"] == 0 and v["wait_delta_min"] == 0 for v in compared["variants"].values()))
        self.assertFalse(report["native_deployment_eligible"])
        results["mixed_a", 0]["cells"].pop()
        with self.assertRaisesRegex(ValueError, "missing future"):
            spec.summarize(results, views)

    def test_scheduler_is_small_unbound_and_model_free(self):
        for view, fold in (("mixed_a", 0), (None, None)):
            task = task_specification("unit_best_crossfit", view, fold)
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertNotIn("_raw", task["result_dir"])
        self.assertEqual(spec.contract()["budget"]["training_cells"], 32)
        self.assertEqual(spec.contract()["budget"]["supervised_decision_updates"], 19200)
        self.assertEqual(spec.contract()["budget"]["wait_residual_updates"], 57600)
        self.assertEqual(spec.contract()["budget"]["new_native_episodes"], 0)


if __name__ == "__main__":
    unittest.main()
