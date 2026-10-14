import copy
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_prefix_policy_calibration as learning
from scripts import run_native_transit_prefix_policy_stage222 as spec
from scripts.submit_native_transit_prefix_policy_stage222_scheduleurm import task_specification
from tests import test_native_transit_best_action_crossfit_stage220 as old
from tests import test_native_transit_best_action_stage218 as cohort
from tests import test_native_transit_coverage_value as native_fixture


def zero_prediction(rows):
    return {"costs": {k: np.zeros((len(rows), 6)) for k in ("ridge", "best", "gap")},
        "wait_logits": {k: np.zeros((len(rows), 6)) for k in learning.GATES},
        "choices": {k: np.asarray([r["original_choice"] for r in rows]) for k in learning.VARIANTS}}


def decision_fixture():
    offset, raw = np.ones((2, 6)), -np.ones((2, 6))
    offset[:, 0] = raw[:, 0] = 0.
    costs = np.ones((2, 6))
    costs[:, 0], costs[:, 1] = 0., -1.
    positive, negative = np.ones((2, 6))*100., np.ones((2, 6))
    positive[:, 0] = negative[:, 0] = 0.
    positive[:, 1], negative[:, 1] = 1., 2.
    values = np.zeros((2, 6, 4))
    values[..., 0] = costs
    values[..., 3] = positive-negative
    return {"originals": np.zeros(2, dtype=int), "roots": np.asarray([397, 401]),
        "costs": costs, "values": values, "offset": offset, "positive": positive, "negative": negative}, raw


class PrefixPolicyTest(unittest.TestCase):
    def test_prefix_folds_never_split_callbacks(self):
        rows, registered = cohort.fixture()
        train, _ = spec.previous.previous.split_rows(spec.previous.previous.source.matched_views(rows, registered)["mixed_a"], 0)
        folds = list(learning.prefix_folds(train))
        self.assertEqual(len(folds), 10)
        for group, training, validation in folds:
            self.assertEqual((len(training), len(validation)), (54, 6))
            self.assertTrue(all((train[i]["root"], train[i]["scene_seed"]) != group for i in training))
            self.assertTrue(all((train[i]["root"], train[i]["scene_seed"]) == group for i in validation))
        self.assertEqual(sorted(i for _, _, held in folds for i in held), list(range(60)))

    def test_policy_floor_preserves_control_when_global_risk_wants_abstention(self):
        data, raw = decision_fixture()
        mask = np.arange(6)[None, :] != data["originals"][:, None]
        risk = learning.base.shrinkage(data["offset"][mask], (raw-data["offset"])[mask], data["positive"][mask], data["negative"][mask])
        self.assertEqual(risk["alpha"], 0.)
        chosen = learning.policy_shrinkage(data, raw, risk["alpha"])
        self.assertAlmostEqual(chosen["alpha"], .5)
        for root in ("397", "401"):
            self.assertEqual(chosen["selected_by_root"][root], {"cost": -1., "wait_min": -1.})
            self.assertEqual(chosen["risk_only_selected_by_root"][root], {"cost": 0., "wait_min": 0.})
        self.assertLess(chosen["physical_mass_loss_min"], learning.physical_risk(data, raw, 1.))

    def test_endpoint_and_exact_native_ties(self):
        offset = np.ones((2, 6))
        raw = offset.copy()
        self.assertTrue(np.array_equal(learning.blended(offset, raw, 1.), raw))
        originals = np.asarray([3, 5])
        self.assertTrue(np.array_equal(learning.selected(np.zeros((2, 6)), np.zeros((2, 6)), originals), originals))
        data, raw = decision_fixture()
        result = learning.policy_shrinkage(data, data["offset"].copy(), .5)
        self.assertEqual(result["alpha"], 1.)
        self.assertTrue(result["selected_action_constraints_met"])

    def test_real_inner_fits_exclude_prefix_labels_and_unused_futures(self):
        bank = native_fixture.bank_fixture()[0]["training_data"]
        groups = sorted({(r["root"], r["scene_seed"]) for r in bank})[:2]
        rows = [r for r in bank if (r["root"], r["scene_seed"]) in groups]
        fold = list(learning.prefix_folds(rows))[0]
        panels = old.panels_for(rows)
        unused, held_changed = copy.deepcopy(panels), copy.deepcopy(panels)
        for row in rows:
            j = (row["original_choice"]+1) % 6
            unused[learning.base.base.state_key(row)]["values"][4:, j] += 100.
        for i in fold[2]:
            row = rows[i]
            j = (row["original_choice"]+1) % 6
            held_changed[learning.base.base.state_key(row)]["values"][:4, j] += 100.
        with patch.object(learning, "prefix_folds", return_value=[fold]), patch.object(learning.base, "STEPS", 3), patch.object(learning.base.base.neural, "STEPS", 3):
            first, ledger = learning.collect(rows, panels, (0, 1, 2, 3))
            second, other = learning.collect(rows, unused, (0, 1, 2, 3))
            changed, changed_ledger = learning.collect(rows, held_changed, (0, 1, 2, 3))
        self.assertEqual(ledger, other)
        self.assertEqual(ledger, changed_ledger)
        for key in first: self.assertTrue(np.array_equal(first[key], second[key]))
        for key in ("costs", "offset", *learning.WAIT_VARIANTS):
            self.assertTrue(np.array_equal(first[key], changed[key]))
        self.assertFalse(np.array_equal(first["values"], changed["values"]))
        for inner in ledger:
            self.assertNotIn(list(fold[0]), inner["training_groups"])
            self.assertNotIn(list(fold[0]), inner["wait_training_groups"])
            self.assertFalse(set(inner["training_draws"]) & set(inner["validation_draws"]))
            self.assertTrue(all(f["steps"] == 3 for f in inner["fits"].values()))

    def test_outer_future_labels_change_only_evaluation(self):
        rows, registered = cohort.fixture()
        train, held = spec.previous.previous.split_rows(spec.previous.previous.source.matched_views(rows, registered)["mixed_a"], 0)
        panels, changed = old.panels_for(rows), old.panels_for(rows)
        for row in held:
            j = (row["original_choice"]+1) % 6
            changed[learning.base.base.state_key(row)]["values"][:, j, 3] += 100.
        captured = []
        def calibration(rs, ps, indices, batch):
            captured.append({learning.base.base.state_key(r): ps[learning.base.base.state_key(r)]["values"][list(indices)].tolist() for r in rs})
            return {}, [], {}
        batch = lambda rs: (np.zeros((len(rs), 6, 198)), np.zeros((len(rs), 12)))
        with patch.object(learning.base.base.neural, "batches", side_effect=batch), patch.object(learning, "calibrate", side_effect=calibration), patch.object(
                learning, "predict", side_effect=lambda rs, ms, batch: zero_prediction(rs)):
            first = spec.fit_cell(train, held, panels, {}, "first_last", 0)[0]
            second = spec.fit_cell(train, held, changed, {}, "first_last", 0)[0]
        self.assertEqual(captured[0], captured[1])
        self.assertNotEqual(first["scopes"]["held_other"], second["scopes"]["held_other"])
        self.assertTrue(all(len(rs) == 60 for rs in second["scopes"].values()))

    def test_reused_model_prediction_is_causal_and_preserves_controls(self):
        rows = native_fixture.bank_fixture()[0]["training_data"][:6]
        panels = old.panels_for(rows)
        with patch.object(learning.base, "STEPS", 3), patch.object(learning.base.base, "STEPS", 3), patch.object(learning.base.base.neural, "STEPS", 3):
            models, _ = learning.base.fit(rows, panels, (0, 1, 2, 3))
        models["prefix_calibration"] = {"risk": {v: {"alpha": 0.} for v in learning.WAIT_VARIANTS},
            "policy": {v: {"alpha": 1.} for v in learning.WAIT_VARIANTS}}
        causal = [{k: v for k, v in r.items() if k != "labels"} for r in rows]
        original = learning.base.predict(causal, models)
        changed = learning.predict(causal, models)
        for v in learning.base.VARIANTS:
            self.assertTrue(np.array_equal(original["choices"][v], changed["choices"][v]))
        for v in learning.WAIT_VARIANTS:
            self.assertTrue(np.array_equal(changed["choices"][f"ridge_prefix_risk_{v}"], original["choices"]["ridge_glm"]))
            self.assertTrue(np.array_equal(changed["choices"][f"ridge_prefix_policy_{v}"], original["choices"][f"ridge_{v}"]))
        self.assertEqual(set(changed["choices"]), set(learning.VARIANTS))

    def test_full_report_and_control_reproduction(self):
        rows, registered = cohort.fixture()
        views = spec.previous.previous.source.matched_views(rows, registered)
        panels, results = old.panels_for(rows), {}
        groups = lambda rs: sorted(map(list, {(r["root"], r["scene_seed"]) for r in rs}))
        for key in spec.KEYS:
            view, fold, partition, direction = key
            train, held = spec.previous.previous.split_rows(views[view], fold)
            a, b = spec.PARTITIONS[partition][direction], spec.PARTITIONS[partition][1-direction]
            inner = []
            for group in groups(train):
                for d in (0, 1):
                    inner.append({"held_group": group, "direction": d, "training_draws": list(a[2*d:2*d+2]),
                        "validation_draws": list(a[2*(1-d):2*(1-d)+2]), "training_states": 54, "held_states": 6,
                        "training_groups": [g for g in groups(train) if g != group],
                        "wait_training_groups": [g for g in groups(train) if g != group],
                        "fits": {v: {"parameter_count": spec.previous.PARAMETERS[v], "steps": learning.base.STEPS} for v in learning.WAIT_VARIANTS}})
            bounds = {str(root): {"cost": 0., "wait_min": 0.} for root, _ in groups(train)}
            calibration = {"risk": {v: {"alpha": .5, "candidate_cases": 600} for v in learning.WAIT_VARIANTS},
                "policy": {v: {"alpha": .5, "raw_bounds_by_root": bounds, "selected_by_root": bounds,
                    "selected_action_constraints_met": True} for v in learning.WAIT_VARIANTS}}
            fit = {"training_states": 60, "training_groups": groups(train), "inner_fits": inner,
                "calibration": calibration, "inner_folds": 20, "analytic_cost_fits": 20, "glm_fits": 20,
                "wait_residual_updates": 60*learning.base.STEPS, "calibration_scalar_fits": 6,
                "supervised_decision_updates": 0, "new_native_episodes": 0, "rl_actor_updates": 0}
            scopes = {s: learning.base.base.records(train if s.startswith("train") else held, panels,
                a if s.endswith("same") else b, b if s.endswith("same") else a,
                zero_prediction(train if s.startswith("train") else held)) for s in spec.SCOPES}
            cell = {"partition": partition, "direction": direction, "training_draws": list(a), "evaluation_draws": list(b),
                "training_groups": groups(train), "held_groups": groups(held), "fit": fit, "scopes": scopes}
            results[key] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "key": list(key), "software_qualified": True, "cell": cell}
        report = spec.summarize(results, views)
        self.assertTrue(all(v["scopes"]["held_other"]["unique_states"] == 120 for v in report["views"].values()))
        self.assertEqual(report["wait_residual_updates"], 1152000)
        self.assertEqual(spec.reproduce_controls(report, copy.deepcopy(report)), {"comparisons": 1584, "max_abs_difference": 0.})
        changed = copy.deepcopy(report)
        changed["views"]["mixed_a"]["scopes"]["held_other"]["variants"]["ridge_glm"]["mean_cost"] = 1.
        with self.assertRaisesRegex(ValueError, "did not reproduce"): spec.reproduce_controls(report, changed)
        first = results[spec.KEYS[0]]["cell"]["fit"]["inner_fits"][0]
        first["validation_draws"] = [4, 5]
        with self.assertRaisesRegex(ValueError, "contaminated"): spec.summarize(results, views)

    def test_scheduler_is_unbound_and_only_syncs_receipts(self):
        for key in (spec.KEYS[0], (None, None, None, None)):
            task = task_specification("unit_prefix_policy", *key)
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertNotIn("_raw", task["result_dir"])
        self.assertEqual(len(spec.KEYS), 32)
        self.assertEqual(spec.contract()["budget"]["supervised_decision_updates"], 0)


if __name__ == "__main__":
    unittest.main()
