import ast
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import run_native_transit_early_stop_wait_stage209 as spec
from scripts.submit_native_transit_early_stop_wait_stage209_scheduleurm import task_specification
from tests.test_native_transit_decision_wait import DecisionWaitTest
from tests.test_native_transit_nonlinear_wait import reference, prediction_rows
from tests.test_native_transit_sequence_value import fixture


class EarlyStopWaitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.rows = fixture()[0]["training_data"]
        cls.masses = {spec.state_key(r): r for r in spec.source.build_masses(cls.rows, {})}
        for m in cls.masses.values():
            m["draw_count"] = 8

    def test_step0_exact_GLM_and_checkpoint_path_preserves_fixed_optimizer(self):
        rows = self.rows[:12]
        ref = reference(rows, self.masses)
        rng = torch.random.get_rng_state().clone()
        zero, curve = spec.stopping.fit_path(rows, self.masses, ref, (0,), validation=rows)
        logits, _, _ = spec.transfer.model_scores(spec.neural.batches(rows)[0], [r["original_choice"] for r in rows], ref)
        np.testing.assert_array_equal(spec.neural.logits_many(rows, zero), logits)
        self.assertEqual((zero["steps"], zero["parameter_update_l2"], zero["hidden_gradient_l2_max"]), (0, 0., 0.))
        with patch.object(spec.neural, "STEPS", 4):
            control = spec.neural.fit_wait(rows, self.masses, ref, "state_conditioned_pair")
        path, curve = spec.stopping.fit_path(rows, self.masses, ref, (0, 2, 4), validation=rows)
        self.assertEqual(path["weights"], control["weights"])
        self.assertEqual(path["final_mass_loss"], control["final_mass_loss"])
        self.assertEqual(curve[-1], {"step": 4, **spec.stopping.validation_loss(rows, self.masses, path)})
        torch.testing.assert_close(rng, torch.random.get_rng_state(), rtol=0, atol=0)

    def test_pooled_loss_selects_step_and_exact_ties_keep_the_earlier_step(self):
        curves = [[{"step": s, "loss_sum_min": 10.-(s == 100), "mass_sum_min": 20.} for s in spec.stopping.CHECKPOINTS]]
        self.assertEqual(spec.stopping.select_checkpoint(curves)[0], 100)
        for p in curves[0]:
            p["loss_sum_min"] = 10.
        self.assertEqual(spec.stopping.select_checkpoint(curves)[0], 0)

    def test_inner_held_labels_never_change_own_GLM_scales_or_training_path(self):
        groups = list({(r["root"], r["scene_seed"]): {k: r[k] for k in ("root", "scene_seed", "scenario")} for r in self.rows}.values())
        keys = set(self.masses)
        partition = spec.stopping.grouping.partition_groups(groups, {spec.source.source.paired.scene_group(r) for r in self.rows})
        held = set(map(tuple, partition[0]))
        train = [r for r in self.rows if spec.source.source.paired.scene_group(r) not in held]
        validation = [r for r in self.rows if spec.source.source.paired.scene_group(r) in held]
        changed = copy.deepcopy(self.masses)
        for r in validation:
            changed[spec.state_key(r)]["positive_wait_min"] = [0. if i == r["original_choice"] else 999. for i in range(6)]
        a = spec.source.feasibility.fit_wait(train, self.masses, "symmetric_feasibility")
        b = spec.source.feasibility.fit_wait(train, changed, "symmetric_feasibility")
        self.assertEqual(a, b)
        model, curve = spec.stopping.fit_path(train, self.masses, a, (0, 4), validation=validation)
        other, changed_curve = spec.stopping.fit_path(train, changed, b, (0, 4), validation=validation)
        self.assertEqual(model, other)
        self.assertNotEqual(curve, changed_curve)
        self.assertEqual(len([g for bucket in partition for g in bucket]), len(groups))
        self.assertEqual({tuple(g) for bucket in partition for g in bucket}, {spec.source.source.paired.scene_group(r) for r in self.rows})
        self.assertEqual(len(keys), len(self.rows))

    def test_outer_held_mass_cannot_choose_checkpoint_and_controls_costs_unchanged(self):
        DecisionWaitTest.setUpClass()
        bank = DecisionWaitTest.bank
        old = spec.previous.manifest(spec.source.manifest(DecisionWaitTest.registered))
        registered = spec.manifest(old)
        keys = {spec.state_key(r) for r in registered["mean_states"]}
        targets = [r for r in bank["training_data"] if spec.state_key(r) in keys]
        masses = {spec.state_key(r): r for r in spec.source.build_masses(bank["training_data"], {})}
        for k in keys:
            masses[k]["draw_count"] = 8
        fold = registered["folds"][0]
        train, test, _ = spec.source.source.coverage.split_rows(bank, fold["split"], fold["index"])
        test = [r for r in test if spec.state_key(r) in keys]
        ref = reference(train, masses)
        prior = {"frozen_cost_fit": {"training_groups": ref["training_groups"]},
            "predictions": {v: prediction_rows(test) for v in spec.VARIANTS[:-1]}}
        before = copy.deepcopy(prior)
        changed = copy.deepcopy(masses)
        for r in test:
            changed[spec.state_key(r)]["positive_wait_min"] = [999.]*6
        with patch.object(spec.stopping, "CHECKPOINTS", (0, 2, 4)):
            cell, model = spec.fit_fold(bank, targets, masses, fold, prior, ref, registered)
            other, refit = spec.fit_fold(bank, targets, changed, fold, prior, ref, registered)
            spec.qualify(cell, fold, registered)
            self.assertEqual(model, refit)
            self.assertEqual(cell["selection"], other["selection"])
            self.assertEqual(cell["predictions"], other["predictions"])
            bad = copy.deepcopy(cell)
            bad["selection"]["inner_fits"][0]["glm_fit"]["training_groups"].append(bad["selection"]["partition"][0][0])
            with self.assertRaises(ValueError):
                spec.qualify(bad, fold, registered)
        self.assertEqual(prior, before)
        for v in spec.VARIANTS[:-1]:
            self.assertEqual(prior["predictions"][v], cell["predictions"][v])
        for original, current in zip(prior["predictions"]["symmetric_feasibility"], cell["predictions"][spec.PRIMARY]):
            self.assertEqual(original["predicted_heads"], current["predicted_heads"])

    def test_all47_zero_step_folds_have_exact_accounting_and_cannot_pass_gate(self):
        DecisionWaitTest.setUpClass()
        bank = DecisionWaitTest.bank
        registered = spec.manifest(spec.previous.manifest(spec.source.manifest(DecisionWaitTest.registered)))
        keys = {spec.state_key(r) for r in registered["mean_states"]}
        panels = {}
        for r in bank["training_data"]:
            if spec.state_key(r) in keys:
                values = np.zeros((8, 6, 4))
                labels = {l["candidate"]: l["wait_delta_min"] for l in r["labels"]}
                values[..., 3] = [labels[i] for i in range(6)]
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
                model, _ = spec.stopping.fit_path(train, masses, ref, (0,))
                inner = [{"held_groups": bucket, "glm_fit": {"training_groups": [g for g in ref["training_groups"] if g not in bucket]},
                    "neural_fit": {"training_groups": [g for g in ref["training_groups"] if g not in bucket]},
                    "curve": [{"step": s, "loss_sum_min": 1., "mass_sum_min": 2.} for s in spec.stopping.CHECKPOINTS]} for bucket in partition]
                step, curve = spec.stopping.select_checkpoint([i["curve"] for i in inner])
                selection = {"partition": partition, "inner_fits": inner, "selected_step": step, "pooled_curve": curve,
                    "validation_mean_states": 240-len(test), "inner_glm_iterations": 0}
                index = fold["split"], fold["index"]
                models[index] = {"symmetric_feasibility": ref, "state_conditioned_pair": model, spec.PRIMARY: model}
                cells[index] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
                    "fold": fold, "frozen_cost_fit": {"training_groups": ref["training_groups"]}, "fit": spec.stopping.metadata(model),
                    "selection": selection, "predictions": {v: prediction_rows(test) for v in spec.VARIANTS},
                    "inner_glm_fits": 4, "inner_neural_paths": 4, "outer_model_assemblies": 1, "adamw_steps": 2400,
                    "new_cost_fits": 0, "actor_updates": 0, "new_native_episodes": 0}
            result = spec.summarize(cells, registered, bank, masses, panels, lambda f: models[f["split"], f["index"]])
        self.assertEqual((result["adamw_steps"], result["outer_optimized_fits"], result["checkpoint_counts"]), (112800, 0, {"0": 47}))
        self.assertFalse(result["development_gate"] or result["third_scene_screen_eligible"] or result["native_deployment_eligible"])
        for v in result["variants"].values():
            self.assertFalse(v["gates"]["cost"])
            for s in v["validation"].values():
                self.assertEqual(s["overall"]["states"], 240)
        with self.assertRaises(ValueError):
            spec.summarize({}, registered, bank, masses, panels, lambda f: None)

    def test_scheduler_unbound_small_results_and_python310(self):
        for task in (task_specification("unit_early_stop"), task_specification("unit_early_stop", {"split": "scene", "index": 0})):
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertNotIn("_raw", task["result_dir"])
        for p in (spec.ROOT/"freq_hrl/domains/transit/native_early_stop_wait.py",
                spec.ROOT/"scripts/run_native_transit_early_stop_wait_stage209.py",
                spec.ROOT/"scripts/submit_native_transit_early_stop_wait_stage209_scheduleurm.py"):
            ast.parse(p.read_text(), feature_version=(3, 10))


if __name__ == "__main__":
    unittest.main()
