import contextlib
import copy
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.domains.transit import native_action_value as paired
from freq_hrl.domains.transit import native_occupancy_value as value
from scripts import run_native_transit_occupancy_value_stage178 as spec
from scripts.analyze_native_transit_occupancy_value_stage178 import summarize
from scripts.submit_native_transit_occupancy_value_stage178_scheduleurm import task_specification
from tests.test_native_transit_action_value import summary_fixture
from tests.test_native_transit_value_context import contexts_fixture


def fixture():
    labels = summary_fixture()
    rows = copy.deepcopy(labels["training_data"])
    for start in range(0, 60, 6):
        contexts, _ = contexts_fixture(rows[start:start+6])
        # Context fixture populates an existing causal speed field, not a new label.
        for row, context in zip(rows[start:start+6], contexts):
            labels["training_data"][rows.index(row)]["context"]["route_speed_mean_mps"] = 7.
            row["occupancy_context"] = {k: context[k] for k in
                ("historical_peak", "active_buses", "queue_by_station", "route_speed_mps")}
    return {"protocol": spec.source.EXPERIMENT_PROTOCOL, "contract": spec.source.contract(),
        "software_qualified": True, "training_updates": 0, "new_native_steps": 613800,
        "source_control_reproductions": 10, "training_states": 60, "paired_labels_reused": 360,
        "fitting_groups": labels["fitting_groups"], "training_data": rows}, labels


class OccupancyValueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_physical_features_ignore_ids_outcomes_and_bus_order(self):
        summary, _ = fixture()
        row = summary["training_data"][0]
        x = value.features(row, spec.PRIMARY)
        self.assertEqual(x.shape, (6, 96))
        changed = copy.deepcopy(row)
        changed.update(root=999, scene_seed=0, scenario="unused", labels=[])
        changed["occupancy_context"]["active_buses"].reverse()
        changed["occupancy_context"]["queue_by_station"].reverse()
        for b in changed["occupancy_context"]["active_buses"]:
            b.update(trip_id=999, actual_remaining_s=-1000.)
        np.testing.assert_allclose(value.features(changed, spec.PRIMARY), x, atol=1e-14)
        changed["occupancy_context"]["historical_peak"] += 1
        self.assertFalse(np.array_equal(value.features(changed, spec.PRIMARY), x))
        np.testing.assert_array_equal(value.features(row, "base_mse")[:, 72:], np.zeros((6, 24)))

    def test_regret_gradient_prefers_actual_cheaper_action_in_original_cost_units(self):
        predicted = torch.zeros((1, 3, 2), requires_grad=True)
        actual = torch.tensor([[.0, .1, .5]])
        loss = value.expected_regret(predicted, actual)
        self.assertAlmostEqual(float(loss.detach()), .2, places=6)
        loss.backward()
        self.assertTrue(torch.all(predicted.grad[0, 0] > 0))
        self.assertTrue(torch.all(predicted.grad[0, 2] < 0))
        torch.testing.assert_close(predicted.grad[..., 0], predicted.grad[..., 1])

    def test_fits_preserve_rng_identity_architecture_and_serialized_inference(self):
        rows = fixture()[0]["training_data"][:6]
        rng = torch.get_rng_state().clone()
        models = [value.fit_value(rows, m, steps=12) for m in value.METHODS]
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertTrue(all(m["parameter_count"] == 1586 and m["parameter_update_l2"] > 0 for m in models))
        model = models[-1]
        predicted = value.predict_value(rows[0], model)
        np.testing.assert_array_equal(predicted[rows[0]["original_choice"]], [0., 0.])
        np.testing.assert_array_equal(value.predict_value(rows[0], json.loads(json.dumps(model))), predicted)
        raw = torch.tensor((value.features(rows[0], spec.PRIMARY)-model["center"])/model["scale"], dtype=torch.float32)
        w = {k: torch.tensor(v) for k, v in model["weights"].items()}
        heads = (torch.tanh(raw@w["0.weight"].T+w["0.bias"])@w["2.weight"].T+w["2.bias"])*model["cost_scale"]
        heads -= heads[rows[0]["original_choice"]].clone()
        np.testing.assert_allclose(heads.numpy(), predicted, atol=1e-8, rtol=1e-5)
        again = value.fit_value(rows, spec.PRIMARY, steps=12)
        self.assertEqual(model["weights"], again["weights"])

    def test_grouped_neural_fit_never_normalizes_or_learns_from_heldout_scene(self):
        rows = fixture()[0]["training_data"][:12]
        def fit(train): return value.fit_value(train, spec.PRIMARY, steps=4)
        a = paired.grouped_validation(rows, "scene", fit_model=fit, predictor=value.predict_value)
        changed = copy.deepcopy(rows)
        for r in changed[:6]:
            for l in r["labels"]: l["fleet_cost_delta"] += 100.
            r["occupancy_context"]["historical_peak"] += 10
        b = paired.grouped_validation(changed, "scene", fit_model=fit, predictor=value.predict_value)
        self.assertEqual(a["folds"][0]["training_groups"], b["folds"][0]["training_groups"])
        # A label change alone must not alter predictions for its own held-out group.
        for r in changed[:6]: r["occupancy_context"]["historical_peak"] -= 10
        c = paired.grouped_validation(changed, "scene", fit_model=fit, predictor=value.predict_value)
        self.assertEqual([p["predicted_heads"] for p in a["predictions"][:6]],
            [p["predicted_heads"] for p in c["predictions"][:6]])
        held = list(paired.scene_group(rows[0]))
        self.assertNotIn(held, a["folds"][0]["training_groups"])

    def test_input_labels_and_primary_gate_cannot_be_replaced_by_control_results(self):
        summary, labels = fixture()
        self.assertEqual(len(spec.training_rows(summary, labels)), 60)
        changed = copy.deepcopy(summary)
        changed["training_data"][0]["labels"][0]["actual_cost_delta"] += 1
        with self.assertRaises(ValueError): spec.training_rows(changed, labels)
        cells = {}
        actual_fit = value.fit_value
        def short_fit(rows, method):
            model = actual_fit(rows, method, steps=2)
            model["steps"] = value.STEPS
            return model
        with patch.object(value, "fit_value", side_effect=short_fit), contextlib.redirect_stdout(io.StringIO()):
            for method in value.METHODS:
                cells[method] = spec.fit_and_validate(summary, labels, method)
        result = summarize(cells)
        json.dumps(result)
        self.assertEqual(result["critic_updates"], 108000)
        self.assertEqual(result["native_screen_eligible"], cells[spec.PRIMARY]["native_screen_eligible"])
        changed = copy.deepcopy(cells)
        changed["base_mse"]["native_screen_eligible"] = True
        with self.assertRaises(ValueError): summarize(changed)

    def test_four_equal_unpinned_tasks_stage_only_code_and_fresh_compact_inputs(self):
        tasks = [task_specification("unit_occupancy", m) for m in value.METHODS]
        self.assertEqual(len({t["result_dir"] for t in tasks}), 4)
        self.assertEqual(len({t["signature"] for t in tasks}), 4)
        for t in tasks:
            self.assertIsNone(t["require_node"])
            self.assertEqual(t["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((t["cpu"], t["ram_mb"]), (1, 3072))
            self.assertEqual(len(t["stage_input_paths"]), 4)
            self.assertIn("OPENBLAS_NUM_THREADS=1", t["cmd"])


if __name__ == "__main__":
    unittest.main()
