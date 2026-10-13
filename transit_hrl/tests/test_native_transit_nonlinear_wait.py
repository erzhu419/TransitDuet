import ast
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import run_native_transit_nonlinear_wait_stage208 as spec
from scripts.submit_native_transit_nonlinear_wait_stage208_scheduleurm import task_specification
from tests.test_native_transit_decision_wait import DecisionWaitTest
from tests.test_native_transit_coverage_value import fake_predictions
from tests.test_native_transit_sequence_value import fixture


def reference(rows, masses, cost=1.):
    raw, _ = spec.neural.batches(rows)
    scale = raw.std(axis=(0, 1))
    scale[scale < 1e-8] = 1.
    records = [masses[spec.state_key(r)] for r in rows]
    absolute = [sum(np.asarray(m["positive_wait_min"])+m["negative_wait_min"])/5 for m in records]
    return {"representation": "conditional_wait", "center": raw.mean(axis=(0, 1)).tolist(), "scale": scale.tolist(),
        "coefficients": [0.]*199, "target_scale_min": max(float(np.average(absolute, weights=[m["draw_count"] for m in records])), 1e-6),
        "training_groups": sorted(map(list, {spec.source.source.paired.scene_group(r) for r in rows})), "training_states": len(rows),
        "training_mean_states": sum(m["draw_count"] == 8 for m in records), "sample_weight_sum": sum(m["draw_count"] for m in records),
        "parameter_count": 199, "false_safe_cost": cost, "ridge": .1, "max_iterations": 200, "solver_success": True, "optimizer_iterations": 0}


def prediction_rows(rows):
    return [{**r, "predicted_feasibility_score": [0.]*6,
        "predicted_harm_mass_share": [0. if i == r["original_choice"] else .5 for i in range(6)]} for r in fake_predictions(rows)]


class NonlinearWaitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = fixture()[0]["training_data"][:12]
        cls.masses = {spec.state_key(r): r for r in spec.source.build_masses(cls.rows, {})}
        cls.reference = reference(cls.rows, cls.masses)

    def test_zero_residual_and_shared_initialization_preserve_the_exact_GLM_offset(self):
        states = []
        rng = torch.random.get_rng_state().clone()
        for variant in spec.neural.VARIANTS:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(spec.neural.SEED)
                network = spec.neural.network(variant)
                inputs = torch.randn(6, 210 if variant == "state_conditioned_pair" else 198, dtype=torch.float64)
                np.testing.assert_array_equal(network(inputs).detach().numpy(), np.zeros((6, 1)))
                states.append(network.state_dict())
        torch.testing.assert_close(rng, torch.random.get_rng_state())
        torch.testing.assert_close(states[0]["0.weight"], states[1]["0.weight"][:, :198])
        torch.testing.assert_close(states[0]["0.bias"], states[1]["0.bias"])
        self.assertEqual(int(torch.count_nonzero(states[1]["0.weight"][:, 198:])), 0)

    def test_absolute_context_survives_where_identity_centering_removes_it(self):
        rows = self.rows[:2]
        raw = np.zeros((2, 6, 198))
        context = np.array([[0.]*12, [1.]*12])
        ref = {**self.reference, "center": [0.]*198, "scale": [1.]*198}
        a, _ = spec.neural.inputs(rows, (raw, context), ref, "nonlinear_pair", np.zeros(12), np.ones(12))
        b, _ = spec.neural.inputs(rows, (raw, context), ref, "state_conditioned_pair", np.zeros(12), np.ones(12))
        np.testing.assert_array_equal(a[0], a[1])
        self.assertFalse(np.array_equal(b[0], b[1]))

    def test_causal_features_ignore_labels_root_regime_and_future_remaining_times(self):
        changed = copy.deepcopy(self.rows)
        for row in changed:
            row.update(root=-1, scene_seed=-1, scenario="unused", labels=[])
            for bus in row["occupancy_context"]["active_buses"]: bus["actual_remaining_s"] = 99999.
        for a, b in zip(spec.neural.batches(self.rows), spec.neural.batches(changed)):
            np.testing.assert_array_equal(a, b)

    def test_training_gradients_numpy_inference_serialization_and_identity(self):
        with patch.object(spec.neural, "STEPS", 12):
            model = spec.neural.fit_wait(self.rows, self.masses, self.reference, "state_conditioned_pair")
        self.assertEqual((model["parameter_count"], model["steps"]), (3393, 12))
        self.assertGreater(model["hidden_gradient_l2_max"], 0.)
        self.assertGreater(model["parameter_update_l2"], 0.)
        batch = spec.neural.batches(self.rows)
        x, offset = spec.neural.inputs(self.rows, batch, model["reference"], model["variant"], model["context_center"], model["context_scale"])
        net = spec.neural.network(model["variant"])
        net.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in model["weights"].items()})
        expected = offset+net(torch.tensor(x, dtype=torch.float64)).detach().numpy()[..., 0]
        np.testing.assert_allclose(spec.neural.logits_many(self.rows, model), expected, atol=1e-12, rtol=0)
        predictions = spec.neural.predict_many(self.rows, model)
        self.assertEqual(predictions, spec.neural.predict_many(self.rows, json.loads(json.dumps(model))))
        for row, prediction in zip(self.rows, predictions): self.assertEqual(prediction["predicted_feasibility_score"][row["original_choice"]], 0.)

    def test_nonlinear_branch_learns_a_small_nonseparable_action_effect(self):
        rows = [{"root": 397, "scene_seed": i, "probe_index": 8, "original_choice": 0} for i in range(4)]
        raw, context, masses = np.zeros((4, 6, 198)), np.zeros((4, 12)), {}
        for row, (i, j) in zip(rows, ((-1, -1), (-1, 1), (1, -1), (1, 1))):
            raw[row["scene_seed"], 1:, :2] = [i, j]
            masses[spec.state_key(row)] = {"draw_count": 1, "positive_wait_min": [0.]+[float(i*j > 0)]*5,
                "negative_wait_min": [0.]+[float(i*j < 0)]*5}
        ref = {"center": [0.]*198, "scale": [1.]*198, "coefficients": [0.]*199, "false_safe_cost": 1.,
            "target_scale_min": 1., "training_groups": [[397, i] for i in range(4)]}
        model = spec.neural.fit_wait(rows, masses, ref, "nonlinear_pair", batch=(raw, context))
        self.assertLess(model["final_mass_loss"], .2)
        self.assertGreater(model["initial_mass_loss"], .69)

    def test_worker_heldout_targets_do_not_change_training_and_costs_controls_are_exact(self):
        DecisionWaitTest.setUpClass()
        bank, previous = DecisionWaitTest.bank, spec.source.manifest(DecisionWaitTest.registered)
        keys = {spec.state_key(r) for r in previous["mean_states"]}
        targets = [r for r in bank["training_data"] if spec.state_key(r) in keys]
        prepared = spec.source.build_masses(bank["training_data"], {})
        for r in prepared:
            if spec.state_key(r) in keys: r["draw_count"] = 8
        masses = {spec.state_key(r): r for r in prepared}
        fold = previous["folds"][0]
        train, test, _ = spec.source.source.coverage.split_rows(bank, fold["split"], fold["index"])
        test = [r for r in test if spec.state_key(r) in keys]
        ref = reference(train, masses)
        prior = {"frozen_cost_fit": {"training_groups": ref["training_groups"]},
            "fits": {v: {k: x for k, x in ref.items() if k not in ("center", "scale", "coefficients")} for v in spec.transfer.VARIANTS},
            "predictions": {v: prediction_rows(test) for v in spec.transfer.VARIANTS}}
        before = copy.deepcopy(prior)
        changed = copy.deepcopy(masses)
        for r in test: changed[spec.state_key(r)]["positive_wait_min"] = [999.]*6
        with patch.object(spec.neural, "STEPS", 4):
            registered = spec.manifest(previous)
            cell, models = spec.fit_fold(bank, targets, masses, fold, prior, ref)
            spec.qualify(cell, fold, registered)
            other, refit = spec.fit_fold(bank, targets, changed, fold, prior, ref)
        self.assertEqual(models, refit)
        self.assertEqual(cell["predictions"], other["predictions"])
        self.assertEqual(prior, before)
        for v in spec.transfer.VARIANTS: self.assertEqual(cell["predictions"][v], prior["predictions"][v])
        for v in spec.neural.VARIANTS:
            for old, current in zip(prior["predictions"]["symmetric_feasibility"], cell["predictions"][v]):
                self.assertEqual(old["predicted_heads"], current["predicted_heads"])

    def test_scheduler_server_only_unbound_and_python310(self):
        for task in (task_specification("unit_nonlinear"), task_specification("unit_nonlinear", {"split": "scene", "index": 0})):
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertNotIn("_raw", task["result_dir"])
        self.assertEqual(spec.contract()["budget"]["wait_optimizer_steps"], 56400)
        for path in (spec.ROOT/"freq_hrl/domains/transit/native_nonlinear_wait.py",
                spec.ROOT/"scripts/run_native_transit_nonlinear_wait_stage208.py",
                spec.ROOT/"scripts/submit_native_transit_nonlinear_wait_stage208_scheduleurm.py"):
            ast.parse(path.read_text(), feature_version=(3, 10))

    def test_all47_folds_identity_cannot_pass_cost_or_training_fit_gate(self):
        DecisionWaitTest.setUpClass()
        bank, previous = DecisionWaitTest.bank, spec.source.manifest(DecisionWaitTest.registered)
        registered = spec.manifest(previous)
        keys = {spec.state_key(r) for r in registered["mean_states"]}
        panels = {}
        for r in bank["training_data"]:
            if spec.state_key(r) in keys:
                values = np.zeros((8, 6, 4))
                labels = {l["candidate"]: l["wait_delta_min"] for l in r["labels"]}
                values[..., 3] = [labels[i] for i in range(6)]
                panels[spec.state_key(r)] = {"original_choice": r["original_choice"], "values": values}
        masses = {spec.state_key(r): r for r in spec.source.build_masses(bank["training_data"], panels)}
        cells, all_models = {}, {}
        with patch.object(spec.neural, "batches", side_effect=lambda rs: (np.zeros((len(rs), 6, 198)), np.zeros((len(rs), 12)))):
            for fold in registered["folds"]:
                train, test, _ = spec.source.source.coverage.split_rows(bank, fold["split"], fold["index"])
                test = [r for r in test if spec.state_key(r) in keys]
                ref = reference(train, masses)
                old = {v: {**ref, "false_safe_cost": cost} for v, cost in spec.source.feasibility.FALSE_SAFE_COSTS.items()}
                new = {}
                for v in spec.neural.VARIANTS:
                    dim = 210 if v == "state_conditioned_pair" else 198
                    new[v] = {"variant": v, "reference": ref, "weights": {"0.weight": np.zeros((16, dim)).tolist(), "0.bias": [0.]*16,
                        "2.weight": [[0.]*16], "2.bias": [0.]}, "context_center": [0.]*12, "context_scale": [1.]*12,
                        "training_groups": ref["training_groups"], "training_states": len(train), "training_mean_states": 240-len(test),
                        "sample_weight_sum": len(train)+7*(240-len(test)), "parameter_count": spec.neural.PARAMETERS[v], "steps": 600,
                        "initialization_seed": 208, "false_safe_cost": 1., "frozen_parameters": 199,
                        "hidden_gradient_l2_max": 1., "parameter_update_l2": 1., "final_mass_loss": 0.}
                key = fold["split"], fold["index"]
                all_models[key] = {**old, **new}
                cells[key] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
                    "fold": fold, "frozen_cost_fit": {"training_groups": ref["training_groups"]}, "frozen_glm_fits": {},
                    "fits": {v: {k: x for k, x in m.items() if k not in ("reference", "weights", "context_center", "context_scale")} for v, m in new.items()},
                    "predictions": {v: prediction_rows(test) for v in spec.VARIANTS}, "new_wait_fits": 2, "wait_optimizer_steps": 1200,
                    "new_cost_fits": 0, "actor_updates": 0, "new_native_episodes": 0}
            result = spec.summarize(cells, registered, bank, masses, panels, lambda f: all_models[f["split"], f["index"]])
        self.assertEqual((result["new_wait_fits"], result["wait_optimizer_steps"]), (94, 56400))
        for v in result["variants"].values():
            self.assertFalse(v["gates"]["cost"])
            for split in v["validation"].values(): self.assertEqual(split["overall"]["states"], 240)
        for v in spec.neural.VARIANTS: self.assertFalse(result["variants"][v]["training_fit_gate"])
        self.assertFalse(result["development_gate"] or result["third_scene_screen_eligible"] or result["native_deployment_eligible"])
        with self.assertRaises(ValueError): spec.summarize({}, registered, bank, masses, panels, lambda f: None)


if __name__ == "__main__":
    unittest.main()
