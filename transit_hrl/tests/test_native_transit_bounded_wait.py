import ast
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import run_native_transit_bounded_wait_stage211 as spec
from scripts.submit_native_transit_bounded_wait_stage211_scheduleurm import task_specification
from tests.test_native_transit_decision_wait import DecisionWaitTest
from tests.test_native_transit_nonlinear_wait import reference, prediction_rows
from tests.test_native_transit_sequence_value import fixture


class BoundedWaitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.rows = fixture()[0]["training_data"][:12]
        cls.masses = {spec.state_key(r): r for r in spec.source.build_masses(cls.rows, {})}
        cls.ref = reference(cls.rows, cls.masses)

    def test_bound_is_smooth_and_functional_penalty_uses_wait_mass_not_parameter_size(self):
        raw = torch.tensor([-100., -.4, 0., .4, 100.], dtype=torch.float64, requires_grad=True)
        bounded = spec.bounded.correction(raw)
        self.assertLessEqual(float(bounded.abs().max()), 1.)
        bounded.sum().backward()
        self.assertEqual(float(raw.grad[2]), 1.)
        self.assertGreater(float(raw.grad[1]), .5)
        r, p, n, w = [torch.tensor(a, dtype=torch.float64) for a in ([.5, 1., 100.], [1., 0., 0.], [0., 3., 0.], [8., 1., 1.])]
        self.assertAlmostEqual(float(spec.bounded.penalty(r, p, n, w, .1)), .05*5/11)
        self.assertEqual(float(spec.bounded.penalty(r, p, n, w, .1)), float(spec.bounded.penalty(r, 2*p, 2*n, w, .1)))

    def test_step0_reference_training_gradients_and_numpy_inference_match_torch(self):
        rng = torch.random.get_rng_state().clone()
        batch = spec.neural.batches(self.rows)
        for variant in spec.bounded.FUNCTIONAL_RIDGES:
            zero, _ = spec.bounded.fit_path(self.rows, self.masses, self.ref, variant, (0,))
            base, _, _ = spec.transfer.model_scores(batch[0], [r["original_choice"] for r in self.rows], self.ref)
            np.testing.assert_array_equal(spec.bounded.logits_many(self.rows, zero), base)
            self.assertEqual((zero["steps"], zero["parameter_update_l2"], zero["final_functional_penalty"]), (0, 0., 0.))
            model, curve = spec.bounded.fit_path(self.rows, self.masses, self.ref, variant, (0, 4), validation=self.rows)
            with torch.random.fork_rng(devices=[]):
                net = spec.neural.network("state_conditioned_pair")
            net.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in model["weights"].items()})
            x, offset = spec.neural.inputs(self.rows, batch, self.ref, "state_conditioned_pair", model["context_center"], model["context_scale"])
            expected = offset+spec.bounded.correction(net(torch.tensor(x, dtype=torch.float64)).squeeze(-1)).detach().numpy()
            np.testing.assert_allclose(spec.bounded.logits_many(self.rows, model), expected, atol=1e-12, rtol=0)
            self.assertEqual(curve[-1], {"step": 4, **spec.bounded.validation_loss(self.rows, self.masses, model, batch)})
            self.assertGreater(model["hidden_gradient_l2_max"], 0.)
            self.assertLessEqual(model["residual_logit_max_abs"], 1.)
            self.assertAlmostEqual(model["final_objective"], model["final_mass_loss"]+model["final_functional_penalty"])
            predictions = spec.bounded.predict_many(self.rows, model)
            self.assertEqual(predictions, spec.bounded.predict_many(self.rows, json.loads(json.dumps(model))))
            for r, p in zip(self.rows, predictions):
                self.assertEqual(p["predicted_harm_mass_share"][r["original_choice"]], 0.)
        torch.testing.assert_close(rng, torch.random.get_rng_state(), rtol=0, atol=0)

    def test_bounded_correction_can_change_safety_sign_unlike_uniform_temperature(self):
        rows = [{"root": 397, "scene_seed": i, "probe_index": 8, "original_choice": 0} for i in range(2)]
        raw, context = np.zeros((2, 6, 198)), np.zeros((2, 12))
        raw[:, 1, 0] = [-1., 1.]
        first, last = np.zeros((16, 210)), np.zeros((1, 16))
        first[0, 0], last[0, 0] = 1., 2.
        ref = {"center": [0.]*198, "scale": [1.]*198, "coefficients": [0.]*198+[.2]}
        model = {"reference": ref, "context_center": [0.]*12, "context_scale": [1.]*12, "residual_bound": 1.,
            "weights": {"0.weight": first.tolist(), "0.bias": [0.]*16, "2.weight": last.tolist(), "2.bias": [0.]}}
        predictions = spec.bounded.predict_many(rows, model, batch=(raw, context))
        self.assertLess(predictions[0]["predicted_feasibility_score"][1], 0.)
        self.assertGreater(predictions[1]["predicted_feasibility_score"][1], 0.)
        self.assertLessEqual(float(np.max(np.abs(spec.bounded.logits_many(rows, model, batch=(raw, context))-.2))), 1.)

    def test_worker_no_outer_label_leakage_shared_inner_GLMS_and_frozen_controls(self):
        DecisionWaitTest.setUpClass()
        bank = DecisionWaitTest.bank
        registered = spec.manifest(spec.previous.manifest(spec.previous.previous.manifest(spec.source.manifest(DecisionWaitTest.registered))))
        keys = {spec.state_key(r) for r in registered["mean_states"]}
        targets = [r for r in bank["training_data"] if spec.state_key(r) in keys]
        masses = {spec.state_key(r): r for r in spec.source.build_masses(bank["training_data"], {})}
        for k in keys:
            masses[k]["draw_count"] = 8
        fold = registered["folds"][0]
        train, test, _ = spec.source.source.coverage.split_rows(bank, fold["split"], fold["index"])
        test = [r for r in test if spec.state_key(r) in keys]
        ref = reference(train, masses)
        prior = {"frozen_cost_fit": {"training_groups": ref["training_groups"]}, "predictions": {v: prediction_rows(test) for v in spec.previous.VARIANTS}}
        original = copy.deepcopy(prior)
        changed = copy.deepcopy(masses)
        for r in test:
            changed[spec.state_key(r)]["positive_wait_min"] = [999.]*6
        with patch.object(spec.stopping, "CHECKPOINTS", (0, 2, 4)), patch.object(spec.source.feasibility, "fit_wait", wraps=spec.source.feasibility.fit_wait) as glm:
            registered = {**registered, "contract": spec.contract()}
            cell, models = spec.fit_fold(bank, targets, masses, fold, prior, ref, registered)
            spec.qualify(cell, fold, registered)
            other, refit = spec.fit_fold(bank, targets, changed, fold, prior, ref, registered)
            self.assertEqual(glm.call_count, 8)  # Four per call, shared by both corrections.
            self.assertEqual(models, refit)
            self.assertEqual(cell["selections"], other["selections"])
            self.assertEqual(cell["predictions"], other["predictions"])
            bad = copy.deepcopy(cell)
            bad["selections"]["shrunk_bounded"]["inner_fits"][0]["glm_fit"]["training_groups"].append(fold["held_out"])
            with self.assertRaises(ValueError):
                spec.qualify(bad, fold, registered)
        self.assertEqual(prior, original)
        for v in spec.previous.VARIANTS:
            self.assertEqual(cell["predictions"][v], prior["predictions"][v])
        for v in spec.bounded.FUNCTIONAL_RIDGES:
            for r, p in zip(prior["predictions"]["symmetric_feasibility"], cell["predictions"][v]):
                self.assertEqual(r["predicted_heads"], p["predicted_heads"])

    def test_all47_zero_step_folds_cannot_be_promoted_and_accounting_does_not_double_GLMS(self):
        DecisionWaitTest.setUpClass()
        bank = DecisionWaitTest.bank
        registered = spec.manifest(spec.previous.manifest(spec.previous.previous.manifest(spec.source.manifest(DecisionWaitTest.registered))))
        keys = {spec.state_key(r) for r in registered["mean_states"]}
        panels = {}
        for r in bank["training_data"]:
            if spec.state_key(r) in keys:
                values = np.zeros((8, 6, 4))
                waits = {l["candidate"]: l["wait_delta_min"] for l in r["labels"]}
                values[..., 3] = [waits[i] for i in range(6)]
                panels[spec.state_key(r)] = {"original_choice": r["original_choice"], "values": values}
        masses = {spec.state_key(r): r for r in spec.source.build_masses(bank["training_data"], panels)}
        cells, models = {}, {}
        with patch.object(spec.neural, "batches", side_effect=lambda rs: (np.zeros((len(rs), 6, 198)), np.zeros((len(rs), 12)))):
            for fold in registered["folds"]:
                train, test, _ = spec.source.source.coverage.split_rows(bank, fold["split"], fold["index"])
                test = [r for r in test if spec.state_key(r) in keys]
                ref = reference(train, masses)
                groups = [g for g in registered["groups"] if spec.source.source.credit.group_key(g, fold["split"]) != fold["held_out"]]
                partition = spec.stopping.grouping.partition_groups(groups, {spec.source.source.paired.scene_group(r) for r in train if spec.state_key(r) in keys})
                unbounded, _ = spec.stopping.fit_path(train, masses, ref, (0,))
                new = {v: spec.bounded.fit_path(train, masses, ref, v, (0,))[0] for v in spec.bounded.FUNCTIONAL_RIDGES}
                inner = [{"held_groups": bucket, "glm_fit": {"training_groups": [g for g in ref["training_groups"] if g not in bucket]},
                    "neural_fit": {"training_groups": [g for g in ref["training_groups"] if g not in bucket]},
                    "curve": [{"step": s, "loss_sum_min": 1., "mass_sum_min": 2.} for s in spec.stopping.CHECKPOINTS]} for bucket in partition]
                _, curve = spec.stopping.select_checkpoint([i["curve"] for i in inner])
                selections = {v: {"partition": partition, "inner_fits": copy.deepcopy(inner), "selected_step": 0,
                    "pooled_curve": curve, "inner_glm_iterations": 0} for v in spec.bounded.FUNCTIONAL_RIDGES}
                index = fold["split"], fold["index"]
                models[index] = {"symmetric_feasibility": ref, "state_conditioned_pair": unbounded, spec.previous.PRIMARY: unbounded, **new}
                cells[index] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
                    "fold": fold, "frozen_cost_fit": {"training_groups": ref["training_groups"]},
                    "fits": {v: spec.stopping.metadata(m) for v, m in new.items()}, "selections": selections,
                    "predictions": {v: prediction_rows(test) for v in spec.VARIANTS}, "inner_glm_fits": 4, "inner_neural_paths": 8,
                    "outer_model_assemblies": 2, "adamw_steps": 4800, "new_cost_fits": 0, "actor_updates": 0, "new_native_episodes": 0}
            result = spec.summarize(cells, registered, bank, masses, panels, lambda f: models[f["split"], f["index"]])
        self.assertEqual((result["new_inner_glm_fits"], result["inner_neural_paths"], result["adamw_steps"]), (188, 376, 225600))
        self.assertFalse(result["development_gate"] or result["third_scene_screen_eligible"] or result["native_deployment_eligible"])
        for v in spec.bounded.FUNCTIONAL_RIDGES:
            self.assertEqual(result["checkpoint_counts"][v], {"0": 47})
            self.assertFalse(result["variants"][v]["gates"]["cost"])
        with self.assertRaises(ValueError):
            spec.summarize({}, registered, bank, masses, panels, lambda f: None)

    def test_scheduler_unbound_small_results_and_python310(self):
        for t in (task_specification("unit_bound"), task_specification("unit_bound", {"split": "scene", "index": 0})):
            self.assertIsNone(t["require_node"])
            self.assertEqual(t["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((t["cpu"], t["ram_mb"]), (1, 3072))
            self.assertNotIn("_raw", t["result_dir"])
        for p in (spec.ROOT/"freq_hrl/domains/transit/native_bounded_wait.py", spec.ROOT/"scripts/run_native_transit_bounded_wait_stage211.py",
                spec.ROOT/"scripts/submit_native_transit_bounded_wait_stage211_scheduleurm.py"):
            ast.parse(p.read_text(), feature_version=(3, 10))


if __name__ == "__main__":
    unittest.main()
