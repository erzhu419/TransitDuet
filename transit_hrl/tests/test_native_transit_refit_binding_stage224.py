import copy
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_refit_binding as learning
from scripts import run_native_transit_refit_binding_stage224 as spec
from scripts.submit_native_transit_refit_binding_stage224_scheduleurm import task_specification
from tests import test_native_transit_best_action_crossfit_stage220 as old
from tests import test_native_transit_best_action_stage218 as cohort
from tests import test_native_transit_coverage_value as native_fixture
from tests.test_native_transit_prefix_policy_stage222 import decision_fixture


def decision_data():
    data, raw = decision_fixture()
    return {**data, **{v: raw.copy() for v in learning.WAIT_VARIANTS}}


def zero_prediction(rows):
    return {"costs": {k: np.zeros((len(rows), 6)) for k in ("ridge", "best")},
        "wait_logits": {k: np.zeros((len(rows), 6)) for k in learning.GATES},
        "choices": {k: np.asarray([r["original_choice"] for r in rows]) for k in learning.VARIANTS}}


class RefitBindingTest(unittest.TestCase):
    def test_joint_floor_corrects_oof_to_final_boundary_drift(self):
        oof, final = decision_data(), decision_data()
        v = learning.WAIT_VARIANTS[-1]
        final[v][:, 1] = -.25
        old_fit = learning.policy.policy_shrinkage(oof, oof[v], 0.)
        self.assertAlmostEqual(old_fit["alpha"], .5)
        self.assertFalse(learning.floors_met(learning.policy.outcomes(final, final[v], old_fit["alpha"]),
            learning.policy.outcomes(final, final[v], 1.)))
        for enforced in (("final",), ("oof", "final")):
            fit = learning.bind(oof, final, v, 0., enforced=enforced)
            self.assertAlmostEqual(fit["alpha"], .8)
            self.assertTrue(all(fit["floor_met"].values()))
            self.assertEqual(fit["changed_choices_from_raw"], {"oof": 0, "final": 0})
        self.assertTrue(np.array_equal(learning.policy.blended(final["offset"], final[v], 1.), final[v]))
        self.assertTrue(np.array_equal(learning.policy.selected(np.zeros((2, 6)), np.zeros((2, 6)), np.array([3, 5])), [3, 5]))

    def test_final_only_and_joint_enforce_different_scopes(self):
        oof, final = decision_data(), decision_data()
        v = learning.WAIT_VARIANTS[-1]
        final["costs"][:, 1] = 1.
        final_fit = learning.bind(oof, final, v, 0., enforced=("final",))
        joint_fit = learning.bind(oof, final, v, 0., enforced=("oof", "final"))
        self.assertAlmostEqual(final_fit["alpha"], 0.)
        self.assertTrue(np.array_equal(learning.policy.selected(oof["costs"], learning.policy.blended(oof["offset"], oof[v], final_fit["alpha"]),
            oof["originals"]), oof["originals"]))
        self.assertFalse(final_fit["floor_met"]["oof"])
        self.assertAlmostEqual(joint_fit["alpha"], .5)
        self.assertTrue(all(joint_fit["floor_met"].values()))
        harmful = copy.deepcopy(final_fit["raw_bounds"]["final"])
        harmful["397"]["cost"], harmful["401"]["cost"] = -1., 1.
        self.assertFalse(learning.floors_met(harmful, final_fit["raw_bounds"]["final"]))

    def test_reused_predictions_are_causal_and_training_future_only(self):
        rows = native_fixture.bank_fixture()[0]["training_data"][:6]
        panels = old.panels_for(rows)
        with patch.object(learning.precision.base.base, "STEPS", 3), patch.object(learning.learned.neural, "STEPS", 3):
            bundle, _ = learning.precision.fit_bundle(rows, panels, (0, 1, 2, 3))
        bundle["best"] = {"center": bundle["ridge"]["center"], "scale": bundle["ridge"]["scale"], "coefficients": [0.]*98}
        first = learning.final_data(rows, panels, (0, 1, 2, 3), bundle)
        changed = copy.deepcopy(panels)
        for row in rows:
            changed[learning.learned.state_key(row)]["values"][4:, (row["original_choice"]+1) % 6] += 100.
        second = learning.final_data(rows, changed, (0, 1, 2, 3), bundle)
        self.assertTrue(all(np.array_equal(first[k], second[k]) for k in first))
        reused = learning.precision.calibrate_data(first)
        calibration, _ = learning.calibrate(first, first, reused)
        causal = [{k: val for k, val in r.items() if k != "labels"} for r in rows]
        before = copy.deepcopy(bundle)
        prediction = learning.predict(causal, bundle, calibration)
        self.assertEqual(bundle, before)
        costs, logits = learning.precision.bundle_prediction(causal, bundle)
        for v in learning.WAIT_VARIANTS:
            self.assertTrue(np.array_equal(prediction["wait_logits"][f"raw_{v}"], logits[v]))
            self.assertTrue(np.array_equal(prediction["choices"][f"raw_{v}"], learning.policy.selected(costs, logits[v], first["originals"])))
        self.assertEqual(set(prediction["choices"]), set(learning.VARIANTS))

    def test_outer_held_labels_change_evaluation_not_calibration(self):
        rows, registered = cohort.fixture()
        source = spec.previous.previous.previous.previous
        train, held = source.split_rows(source.source.matched_views(rows, registered)["mixed_a"], 0)
        panels, changed = old.panels_for(rows), old.panels_for(rows)
        for row in held:
            changed[learning.learned.state_key(row)]["values"][:, (row["original_choice"]+1) % 6, 3] += 100.
        captured = []
        def final_data(rs, ps, indices, bundle, batch):
            captured.append({learning.learned.state_key(r): ps[learning.learned.state_key(r)]["values"][list(indices)].tolist() for r in rs})
            return {}
        batch = lambda rs: (np.zeros((len(rs), 6, 198)), np.zeros((len(rs), 12)))
        with patch.object(learning.learned.neural, "batches", side_effect=batch), patch.object(learning, "final_data", side_effect=final_data), patch.object(
                learning, "calibrate", return_value=({}, {})), patch.object(learning, "predict", side_effect=lambda rs, *args, **kwargs: zero_prediction(rs)):
            first = spec.fit_cell(train, held, panels, {}, {"originals": np.zeros(120)}, {}, "first_last", 0)[0]
            second = spec.fit_cell(train, held, changed, {}, {"originals": np.zeros(120)}, {}, "first_last", 0)[0]
        self.assertEqual(captured[0], captured[1])
        self.assertNotEqual(first["scopes"]["held_other"], second["scopes"]["held_other"])

    def test_complete_coverage_and_actual_training_floor(self):
        rows, registered = cohort.fixture()
        source = spec.previous.previous.previous.previous
        views = source.source.matched_views(rows, registered)
        panels, results = old.panels_for(rows), {}
        groups = lambda rs: sorted(map(list, {(r["root"], r["scene_seed"]) for r in rs}))
        for key in spec.KEYS:
            view, fold, partition, direction = key
            train, held = source.split_rows(views[view], fold)
            a, b = spec.PARTITIONS[partition][direction], spec.PARTITIONS[partition][1-direction]
            bounds = {str(root): {"cost": 0., "wait_min": 0.} for root, _ in groups(train)}
            scopes = {s: learning.learned.records(train if s.startswith("train") else held, panels,
                a if s.endswith("same") else b, b if s.endswith("same") else a,
                zero_prediction(train if s.startswith("train") else held)) for s in spec.SCOPES}
            calibration = {mode: {v: {"alpha": .5, "enforced_scopes": list(enforced),
                "raw_bounds": {s: bounds for s in ("oof", "final")}, "selected": {s: bounds for s in ("oof", "final")},
                "floor_met": {"oof": True, "final": True}} for v in learning.WAIT_VARIANTS} for mode, enforced in spec.ENFORCED.items()}
            fit = {"calibration": calibration, "oof_only_final_diagnostic": {v: {"raw_final_bounds": bounds,
                "oof_only_on_final": bounds} for v in learning.WAIT_VARIANTS}, "final_constraint_states": 60,
                "oof_candidate_cases": 600, "calibration_scalar_fits": 6,
                **{k: 0 for k in ("analytic_cost_fits", "glm_fits", "wait_residual_updates", "new_native_episodes", "supervised_decision_updates", "rl_actor_updates")}}
            cell = {"partition": partition, "direction": direction, "training_draws": list(a), "evaluation_draws": list(b),
                "training_groups": groups(train), "held_groups": groups(held), "fit": fit, "scopes": scopes}
            results[key] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "key": list(key), "software_qualified": True, "cell": cell}
        report = spec.summarize(results, views)
        self.assertEqual((report["calibration_scalar_fits"], report["wait_residual_updates"]), (192, 0))
        self.assertTrue(all(v["scopes"]["held_other"]["prediction_cases"] == 480 and v["scopes"]["held_other"]["unique_states"] == 120 for v in report["views"].values()))
        first = results[spec.KEYS[0]]["cell"]["scopes"]["train_same"][0]
        first["decisions"][f"joint_{learning.WAIT_VARIANTS[-1]}"]["cost"] = 1.
        with self.assertRaisesRegex(ValueError, "Actual final training decisions"): spec.summarize(results, views)

    def test_primary_controls_and_unbound_scheduler(self):
        names = {"glm": "f4_glm", **{f"raw_{v}": f"f4_raw_{v}" for v in learning.WAIT_VARIANTS},
            **{f"{m}_{v}": f"f4_c2v4_{old}_{v}" for m, old in (("risk", "risk"), ("oof", "policy")) for v in learning.WAIT_VARIANTS}}
        metric = {"mean_cost": 0., "mean_wait_min": 0., "changed_fraction": 0., "unsafe_fraction": 0.}
        def report(ns):
            return {"views": {v: {"scopes": {s: {"variants": {n: metric.copy() for n in ns},
                "wait_heads": {n: {"physical_mass_loss_min": 0.} for n in ns}} for s in spec.SCOPES}} for v in spec.VIEWS}}
        a, b = report(names), report(names.values())
        self.assertEqual(spec.reproduce_controls(a, b), {"comparisons": 800, "max_abs_difference": 0.})
        b["views"]["mixed_a"]["scopes"]["held_other"]["variants"]["f4_glm"]["mean_cost"] = 1.
        with self.assertRaisesRegex(ValueError, "failed reproduction"): spec.reproduce_controls(a, b)
        for key in (spec.KEYS[0], (None, None, None, None)):
            task = task_specification("unit_binding", *key)
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertNotIn("_raw", task["result_dir"])


if __name__ == "__main__":
    unittest.main()
