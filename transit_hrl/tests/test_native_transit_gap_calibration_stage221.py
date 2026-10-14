import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.domains.transit import native_gap_calibration as learning
from scripts import run_native_transit_gap_calibration_stage221 as spec
from scripts.submit_native_transit_gap_calibration_stage221_scheduleurm import task_specification
from tests import test_native_transit_best_action_crossfit_stage220 as old
from tests import test_native_transit_best_action_stage218 as cohort
from tests import test_native_transit_coverage_value as native_fixture


def zero_prediction(rows):
    return {"costs": {k: np.zeros((len(rows), 6)) for k in ("ridge", "best", "gap")},
        "wait_logits": {k: np.zeros((len(rows), 6)) for k in learning.GATES},
        "choices": {k: np.asarray([r["original_choice"] for r in rows]) for k in learning.VARIANTS}}


class GapCalibrationTest(unittest.TestCase):
    def test_gap_objective_uses_cost_magnitude_and_has_neutral_ties(self):
        costs = torch.tensor([[0., -1., -1., 1., 2., 3.]], dtype=torch.float64)
        a = torch.zeros_like(costs, requires_grad=True)
        learning.gap_loss(a, costs, 1.).backward()
        gradient = a.grad.clone()
        costs[0, 1] = -3.
        b = torch.zeros_like(costs, requires_grad=True)
        learning.gap_loss(b, costs, 1.).backward()
        self.assertEqual(gradient[0, 1], gradient[0, 2])
        self.assertNotEqual(gradient[0, 1], b.grad[0, 1])
        c = torch.zeros_like(costs, requires_grad=True)
        value = learning.gap_loss(c, torch.zeros_like(costs), 1.)
        value.backward()
        self.assertAlmostEqual(float(value.detach()), 0.)
        self.assertTrue(torch.allclose(c.grad, torch.zeros_like(c)))

    def test_convex_shrinkage_covers_zero_one_and_interior(self):
        offset, residual = np.zeros(3), np.ones(3)*4
        for positive, negative, expected in ((0., 1., 0.), (1., 0., 1.), (2., 1., np.log(2)/4)):
            result = learning.shrinkage(offset, residual, np.ones(3)*positive, np.ones(3)*negative)
            self.assertAlmostEqual(result["alpha"], expected)
            self.assertLessEqual(result["shrunk_loss_min"], min(result["reference_loss_min"], result["raw_loss_min"])+1e-12)
        self.assertEqual(learning.shrinkage(offset, np.zeros(3), residual, residual)["alpha"], 0.)

    def test_real_fit_ignores_other_futures_and_legacy_targets_and_predicts_causally(self):
        rows = native_fixture.bank_fixture()[0]["training_data"][:6]
        panels = old.panels_for(rows)
        changed = copy.deepcopy(panels)
        for row in rows:
            j = (row["original_choice"]+1) % 6
            changed[learning.base.state_key(row)]["values"][4:, j] += 500.
        other_rows = copy.deepcopy(rows)
        for row in other_rows: row["labels"] = [{"irrelevant": 999}]
        with patch.object(learning, "STEPS", 3), patch.object(learning.base, "STEPS", 3), patch.object(learning.base.neural, "STEPS", 3):
            models, fit = learning.fit(rows, panels, (0, 1, 2, 3))
            second, other_fit = learning.fit(other_rows, changed, (0, 1, 2, 3))
        self.assertEqual(models, second)
        self.assertEqual(fit, other_fit)
        self.assertEqual(fit["supervised_decision_updates"], 6)
        self.assertEqual(fit["wait_residual_updates"], 27)
        self.assertEqual(fit["glm_fits"], 3)
        self.assertEqual(fit["analytic_cost_fits"], 1)
        self.assertEqual(fit["fits"]["gap"]["parameter_count"], 98)
        self.assertTrue(np.isfinite(fit["fits"]["gap"]["final_loss"]))
        self.assertEqual([(i["training_draws"], i["validation_draws"]) for i in fit["inner_fits"]],
            [([0, 1], [2, 3]), ([2, 3], [0, 1])])
        causal = [{k: v for k, v in r.items() if k != "labels"} for r in rows]
        prediction = learning.predict(causal, models)
        self.assertEqual(set(prediction["choices"]), set(learning.VARIANTS))
        for v in learning.WAIT_VARIANTS:
            self.assertTrue(np.allclose(prediction["wait_logits"][f"shrunk_{v}"], prediction["wait_logits"]["glm"]+
                models["calibration"][v]["alpha"]*(prediction["wait_logits"][v]-prediction["wait_logits"]["glm"])))
        for i, row in enumerate(rows):
            self.assertEqual(prediction["costs"]["gap"][i, row["original_choice"]], 0.)
            self.assertTrue(all(z[i, row["original_choice"]] == 0 for z in prediction["wait_logits"].values()))

    def test_outer_held_labels_only_change_evaluation(self):
        rows, registered = cohort.fixture()
        train, held = spec.previous.split_rows(spec.previous.source.matched_views(rows, registered)["mixed_a"], 0)
        panels, captured = old.panels_for(rows), []
        changed = copy.deepcopy(panels)
        for row in held:
            j = (row["original_choice"]+1) % 6
            changed[learning.base.state_key(row)]["values"][:, j, 3] = 10.
        def fit(rs, ps, indices, batch):
            captured.append((copy.deepcopy(rs), {learning.base.state_key(r): ps[learning.base.state_key(r)]["values"][list(indices)].tolist() for r in rs}))
            return {}, {}
        batch = lambda rs: (np.zeros((len(rs), 6, 198)), np.zeros((len(rs), 12)))
        with patch.object(learning.base.neural, "batches", side_effect=batch), patch.object(learning, "fit", side_effect=fit), patch.object(
                learning, "predict", side_effect=lambda rs, ms, batch: zero_prediction(rs)):
            first = spec.fit_cell(train, held, panels, "first_last", 0, lambda m: None)
            second = spec.fit_cell(train, held, changed, "first_last", 0, lambda m: None)
        self.assertEqual(captured[0], captured[1])
        self.assertNotEqual(first["scopes"]["held_other"], second["scopes"]["held_other"])
        self.assertTrue(all(len(rs) == 60 for rs in second["scopes"].values()))

    def test_report_coverage_budget_and_rejection_of_outer_calibration(self):
        rows, registered = cohort.fixture()
        views = spec.previous.source.matched_views(rows, registered)
        panels, results = old.panels_for(rows), {}
        groups = lambda rs: sorted(map(list, {(r["root"], r["scene_seed"]) for r in rs}))
        for view, rs in views.items():
            for fold in (0, 1):
                train, held = spec.previous.split_rows(rs, fold)
                cells = []
                for partition in spec.PARTITIONS:
                    for direction in (0, 1):
                        a, b = spec.PARTITIONS[partition][direction], spec.PARTITIONS[partition][1-direction]
                        fits = {v: {"parameter_count": count, "steps": learning.STEPS,
                            "initial_loss": 1., "final_loss": .5, "parameter_update_l2": 1.} for v, count in spec.PARAMETERS.items()}
                        inner = [{"training_draws": list(a[2*i:2*i+2]), "validation_draws": list(a[2*(1-i):2*(1-i)+2]),
                            "fits": {v: fits[v] for v in learning.WAIT_VARIANTS}, "glm_iterations": 5} for i in (0, 1)]
                        calibration = {v: {"alpha": .5, "reference_loss_min": 1., "raw_loss_min": 1.,
                            "shrunk_loss_min": .5, "candidate_cases": 600} for v in learning.WAIT_VARIANTS}
                        scopes = {s: learning.base.records(train if s.startswith("train") else held, panels,
                            a if s.endswith("same") else b, b if s.endswith("same") else a,
                            zero_prediction(train if s.startswith("train") else held)) for s in spec.SCOPES}
                        cells.append({"partition": partition, "direction": direction, "training_draws": list(a), "evaluation_draws": list(b),
                            "training_groups": groups(train), "held_groups": groups(held), "scopes": scopes,
                            "fit": {"training_groups": groups(train), "training_states": 60, "target_draw_count": [4], "fits": fits,
                                "supervised_decision_updates": 2*learning.STEPS, "wait_residual_updates": 9*learning.STEPS,
                                "glm_fits": 3, "analytic_cost_fits": 1, "calibration_scalar_fits": 3, "inner_fits": inner,
                                "calibration": calibration, "new_native_episodes": 0, "rl_actor_updates": 0, "glm_iterations": 5}})
                results[view, fold] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "view": view,
                    "fold": fold, "software_qualified": True, "cells": cells}
        report = spec.summarize(results, views)
        for view in report["views"].values():
            self.assertEqual(view["scopes"]["held_other"]["prediction_cases"], 480)
            self.assertEqual(view["scopes"]["held_other"]["unique_states"], 120)
        self.assertEqual(report["supervised_decision_updates"], 38400)
        self.assertEqual(report["wait_residual_updates"], 172800)
        self.assertTrue(all(v["cost_delta"] == 0 for x in report["single_vs_mixed"].values() for v in x["variants"].values()))
        self.assertFalse(report["native_deployment_eligible"])
        results["mixed_a", 0]["cells"][0]["fit"]["inner_fits"][0]["validation_draws"] = [4, 5]
        with self.assertRaisesRegex(ValueError, "outer evaluation"):
            spec.summarize(results, views)

    def test_scheduler_is_unbound_small_and_does_not_pull_models(self):
        for view, fold in (("mixed_a", 0), (None, None)):
            task = task_specification("unit_gap", view, fold)
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertNotIn("_raw", task["result_dir"])
            self.assertIn("stage221.py", task["cmd"])
        self.assertEqual(spec.contract()["budget"]["new_native_episodes"], 0)


if __name__ == "__main__":
    unittest.main()
