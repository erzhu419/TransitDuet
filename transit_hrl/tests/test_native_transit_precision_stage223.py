import copy
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_precision_calibration as learning
from scripts import run_native_transit_precision_stage223 as spec
from scripts.submit_native_transit_precision_stage223_scheduleurm import task_specification
from tests import test_native_transit_best_action_crossfit_stage220 as old
from tests import test_native_transit_best_action_stage218 as cohort
from tests import test_native_transit_coverage_value as native_fixture


def zero_prediction(rows):
    return {"costs": {k: np.zeros((len(rows), 6)) for k in ("f2", "f4", "best")},
        "wait_logits": {k: np.zeros((len(rows), 6)) for k in learning.GATES},
        "choices": {k: np.asarray([r["original_choice"] for r in rows]) for k in learning.VARIANTS}}


class PrecisionTest(unittest.TestCase):
    def test_common_validation_preserves_draw_masses_and_ignores_outer_future(self):
        rows, _ = cohort.fixture()
        rows = rows[:12]
        panels = old.panels_for(rows)
        for row in rows:
            j = (row["original_choice"]+1) % 6
            panels[learning.base.base.base.state_key(row)]["values"][:4, j, 3] = [2., -1., -1., -1.]
        repeated = learning.validation_rows(rows)
        data = {"originals": np.asarray([r["original_choice"] for r in repeated]), "roots": np.asarray([r["root"] for r in repeated]),
            "costs": np.ones((len(repeated), 6))}
        first = learning.common_validation(rows, panels, (0, 1, 2, 3), data)
        changed = copy.deepcopy(panels)
        for row in rows:
            changed[learning.base.base.base.state_key(row)]["values"][4:, (row["original_choice"]+1) % 6] += 100.
        second = learning.common_validation(rows, changed, (0, 1, 2, 3), data)
        self.assertTrue(all(np.array_equal(first[k], second[k]) for k in first))
        j = (repeated[0]["original_choice"]+1) % 6
        self.assertEqual(first["positive"][0, j], .5)
        self.assertEqual(first["negative"][0, j], .75)
        self.assertEqual(first["values"][0, j, 3], -.25)
        self.assertTrue(np.array_equal(first["costs"], data["costs"]))

    def test_real_precision_fits_and_outer_label_isolation(self):
        bank = native_fixture.bank_fixture()[0]["training_data"]
        groups = sorted({(r["root"], r["scene_seed"]) for r in bank})[:2]
        rows = [r for r in bank if (r["root"], r["scene_seed"]) in groups]
        panels = old.panels_for(rows)
        for row in rows:
            j = (row["original_choice"]+1) % 6
            panels[learning.base.base.base.state_key(row)]["values"][:, j, 3] = np.linspace(-.5, .5, 8)
        with patch.object(learning.base.base, "STEPS", 3), patch.object(learning.base.base.base.neural, "STEPS", 3):
            reused, _ = learning.base.collect(rows, panels, (0, 1, 2, 3))
            calibration, models, fit, data = learning.fit_cell_models(rows, panels, (0, 1, 2, 3), reused)
            changed = copy.deepcopy(panels)
            for row in rows:
                changed[learning.base.base.base.state_key(row)]["values"][4:, (row["original_choice"]+1) % 6] += 100.
            second, other_models, other_fit, other_data = learning.fit_cell_models(rows, changed, (0, 1, 2, 3), reused)
            final4, _ = learning.fit_bundle(rows, panels, (0, 1, 2, 3))
        self.assertEqual(calibration, second)
        self.assertEqual(models, other_models)
        self.assertEqual(fit, other_fit)
        self.assertTrue(all(np.array_equal(data[c][k], other_data[c][k]) for c in data for k in data[c]))
        self.assertEqual([f["draws"] for f in fit["final_fits"]], [[0, 1], [2, 3]])
        for f in fit["inner_fits"]:
            self.assertEqual(f["draws"], [0, 1, 2, 3])
            self.assertNotIn(f["held_group"], f["training_groups"])
            self.assertNotIn(f["held_group"], f["wait_training_groups"])
            self.assertTrue(all(x["steps"] == 3 for x in f["fits"].values()))
        calibration["legacy"] = calibration["c2v4"]
        final4["best"] = {"center": final4["ridge"]["center"], "scale": final4["ridge"]["scale"], "coefficients": [0.]*98}
        causal = [{k: v for k, v in r.items() if k != "labels"} for r in rows]
        prediction = learning.predict(causal, models[0], final4, calibration)
        self.assertEqual(set(prediction["choices"]), set(learning.VARIANTS))
        for i, r in enumerate(rows):
            self.assertTrue(all(z[i, r["original_choice"]] == 0. for z in prediction["wait_logits"].values()))
        self.assertTrue(np.array_equal(data["c4v4"]["offset"][:6], data["c4v4"]["offset"][6:12]))
        broken = copy.deepcopy(reused)
        broken["values"][0, (rows[0]["original_choice"]+1) % 6, 0] += 1.
        with self.assertRaisesRegex(ValueError, "prefix/future order"):
            learning.fit_cell_models(rows, panels, (0, 1, 2, 3), broken)

    def test_outer_held_prefix_labels_do_not_enter_calibration(self):
        rows, registered = cohort.fixture()
        train, held = spec.previous.previous.previous.split_rows(spec.previous.previous.previous.source.matched_views(rows, registered)["mixed_a"], 0)
        panels, changed = old.panels_for(rows), old.panels_for(rows)
        for row in held:
            changed[learning.base.base.base.state_key(row)]["values"][:, (row["original_choice"]+1) % 6, 3] = 100.
        captured = []
        def fitted(rs, ps, indices, reused, batch):
            captured.append({learning.base.base.base.state_key(r): ps[learning.base.base.base.state_key(r)]["values"][list(indices)].tolist() for r in rs})
            return {}, [{}, {}], {}, {}
        batch = lambda rs: (np.zeros((len(rs), 6, 198)), np.zeros((len(rs), 12)))
        with patch.object(learning.base.base.base.neural, "batches", side_effect=batch), patch.object(learning, "fit_cell_models", side_effect=fitted), patch.object(
                learning, "predict", side_effect=lambda rs, *args, **kwargs: zero_prediction(rs)):
            first = spec.fit_cell(train, held, panels, {"prefix_calibration": {}, "ridge": {}, "wait": {}, "best": {}}, {}, "first_last", 0)[0]
            second = spec.fit_cell(train, held, changed, {"prefix_calibration": {}, "ridge": {}, "wait": {}, "best": {}}, {}, "first_last", 0)[0]
        self.assertEqual(captured[0], captured[1])
        self.assertNotEqual(first["scopes"]["held_other"], second["scopes"]["held_other"])
        self.assertTrue(all(len(rs) == 120 for rs in second["scopes"].values()))

    def test_complete_report_and_missing_subhalf_rejection(self):
        rows, registered = cohort.fixture()
        views = spec.previous.previous.previous.source.matched_views(rows, registered)
        panels, results = old.panels_for(rows), {}
        groups = lambda rs: sorted(map(list, {(r["root"], r["scene_seed"]) for r in rs}))
        for key in spec.KEYS:
            view, fold, partition, direction = key
            train, held = spec.previous.previous.previous.split_rows(views[view], fold)
            a, b = spec.PARTITIONS[partition][direction], spec.PARTITIONS[partition][1-direction]
            def fitted(gs, draws, count):
                return {"draws": list(draws), "training_states": count, "training_groups": gs, "wait_training_groups": gs,
                    "fits": {v: {"parameter_count": spec.contract()["parameters"][v], "steps": learning.base.base.STEPS} for v in learning.WAIT_VARIANTS}}
            inner = [{"held_group": g, **fitted([x for x in groups(train) if x != g], a, 54)} for g in groups(train)]
            bounds = {str(root): {"cost": 0., "wait_min": 0.} for root, _ in groups(train)}
            cal = {c: {"risk": {v: {"alpha": .5, "candidate_cases": 600} for v in learning.WAIT_VARIANTS},
                "policy": {v: {"alpha": .5, "raw_bounds_by_root": bounds, "selected_by_root": bounds,
                    "selected_action_constraints_met": True} for v in learning.WAIT_VARIANTS}} for c in learning.CALIBRATIONS}
            fit = {"inner_fits": inner, "final_fits": [fitted(groups(train), a[:2], 60), fitted(groups(train), a[2:], 60)],
                "calibration": cal, "wait_residual_updates": 36*learning.base.base.STEPS, "analytic_cost_fits": 12,
                "glm_fits": 12, "calibration_scalar_fits": 12, "new_native_episodes": 0, "supervised_decision_updates": 0, "rl_actor_updates": 0}
            scopes = {s: [{**r, "deployment_subhalf": half} for half in (0, 1) for r in learning.base.base.base.records(
                train if s.startswith("train") else held, panels, a if s.endswith("same") else b, b if s.endswith("same") else a,
                zero_prediction(train if s.startswith("train") else held))] for s in spec.SCOPES}
            cell = {"partition": partition, "direction": direction, "training_draws": list(a), "evaluation_draws": list(b),
                "training_groups": groups(train), "held_groups": groups(held), "fit": fit, "scopes": scopes}
            results[key] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "key": list(key), "software_qualified": True, "cell": cell}
        report = spec.summarize(results, views)
        self.assertTrue(all(v["scopes"]["held_other"]["prediction_cases"] == 960 and v["scopes"]["held_other"]["unique_states"] == 120 for v in report["views"].values()))
        self.assertTrue(all(m["prediction_cases"] == 480 and m["unique_states"] == 120 for v in report["views"].values() for m in v["by_deployment_subhalf"].values()))
        self.assertEqual(report["wait_residual_updates"], 691200)
        cells = results[spec.KEYS[0]]["cell"]["scopes"]["held_other"]
        cells[0]["deployment_subhalf"] = 1
        with self.assertRaisesRegex(ValueError, "state/subhalf coverage"): spec.summarize(results, views)

    def test_primary_control_mapping_and_scheduler_budget(self):
        names = {"f4_glm": "ridge_glm", **{f"f4_raw_{v}": f"ridge_{v}" for v in learning.WAIT_VARIANTS},
            **{f"f4_legacy_{m}_{v}": f"ridge_prefix_{m}_{v}" for m in ("risk", "policy") for v in learning.WAIT_VARIANTS}}
        metric = {"mean_cost": 0., "mean_wait_min": 0., "changed_fraction": 0., "unsafe_fraction": 0.}
        result = {"views": {v: {"scopes": {s: {"variants": {n: metric.copy() for n in names},
            "wait_heads": {n: {"physical_mass_loss_min": 0.} for n in names}} for s in spec.SCOPES}} for v in spec.VIEWS}}
        old_result = {"views": {v: {"scopes": {s: {"variants": {n: metric.copy() for n in names.values()},
            "wait_heads": {n.removeprefix("ridge_"): {"physical_mass_loss_min": 0.} for n in names.values()}} for s in spec.SCOPES}} for v in spec.VIEWS}}
        self.assertEqual(spec.reproduce_controls(result, old_result), {"comparisons": 800, "max_abs_difference": 0.})
        old_result["views"]["mixed_a"]["scopes"]["held_other"]["variants"]["ridge_glm"]["mean_cost"] = 1.
        with self.assertRaisesRegex(ValueError, "failed reproduction"): spec.reproduce_controls(result, old_result)
        for key in (spec.KEYS[0], (None, None, None, None)):
            task = task_specification("unit_precision", *key)
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertNotIn("_raw", task["result_dir"])


if __name__ == "__main__":
    unittest.main()
