import ast
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import run_native_transit_future_transfer_stage210 as spec
from scripts.submit_native_transit_future_transfer_stage210_scheduleurm import task_specification
from tests.test_native_transit_decision_wait import DecisionWaitTest


class FutureTransferTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        DecisionWaitTest.setUpClass()
        cls.bank = DecisionWaitTest.bank
        cls.registered = spec.manifest(spec.previous.manifest(spec.previous.previous.manifest(spec.source.manifest(DecisionWaitTest.registered))))
        cls.rows = cls.bank["training_data"]
        keys = {spec.state_key(r) for r in cls.registered["mean_states"]}
        cls.mean_rows = [r for r in cls.rows if spec.state_key(r) in keys]
        panels = {}
        for r in cls.mean_rows:
            values = np.zeros((8, 6, 4))
            labels = {l["candidate"]: l["wait_delta_min"] for l in r["labels"]}
            values[..., 3] = np.asarray([labels[i] for i in range(6)])[None, :]
            values[:, :, 3] += np.arange(8)[:, None]*.005*(np.arange(6) != r["original_choice"])[None, :]
            panels[spec.state_key(r)] = {"original_choice": r["original_choice"], "values": values}
        cls.full = {spec.state_key(r): r for r in spec.source.build_masses(cls.rows, panels)}
        prepared = spec.half_masses(cls.rows, cls.full, panels)
        cls.tables = {p: {h: {spec.state_key(r): r for r in rs} for h, rs in hs.items()} for p, hs in prepared.items()}
        cls.fold = cls.registered["folds"][0]
        cls.train, _, _ = spec.source.source.coverage.split_rows(cls.bank, cls.fold["split"], cls.fold["index"])

    def test_half_targets_reproduce_full_masses_and_singletons_are_unchanged(self):
        mean_keys = {spec.state_key(r) for r in self.mean_rows}
        for tables in self.tables.values():
            for k, r in self.full.items():
                for h in ("A", "B"):
                    self.assertEqual(tables[h][k]["draw_count"], 4 if k in mean_keys else 1)
                    self.assertEqual(tables[h][k]["positive_wait_min"][r["original_choice"]], 0.)
                    if k not in mean_keys:
                        self.assertEqual(tables[h][k], r)
                for field in ("positive_wait_min", "negative_wait_min"):
                    np.testing.assert_allclose((np.asarray(tables["A"][k][field])+tables["B"][k][field])/2, r[field], atol=1e-12, rtol=0)

    def test_checkpoint_observation_does_not_change_the_training_path(self):
        rows = self.train[:12]
        masses = self.tables["first_last"]["A"]
        glm = spec.source.feasibility.fit_wait(rows, masses, "symmetric_feasibility")
        seen = []
        a, _ = spec.stopping.fit_path(rows, masses, glm, (0, 2, 4))
        b, _ = spec.stopping.fit_path(rows, masses, glm, (0, 2, 4), on_checkpoint=lambda m: seen.append(
            (m["steps"], spec.neural.logits_many(rows, m))))
        self.assertEqual(a, b)
        self.assertEqual([s for s, _ in seen], [0, 2, 4])
        np.testing.assert_array_equal(seen[-1][1], spec.neural.logits_many(rows, a))

    def assert_no_fit_leakage(self, tables):
        models = []
        original = spec.stopping.fit_path
        def capture(*args, **kwargs):
            model, curve = original(*args, **kwargs)
            models.append(model)
            return model, curve
        with patch.object(spec.stopping, "CHECKPOINTS", (0, 2, 4)), patch.object(spec.stopping, "fit_path", side_effect=capture):
            a = spec.fit_case(self.train, self.mean_rows, self.tables["first_last"], self.fold, "A")
            b = spec.fit_case(self.train, self.mean_rows, tables, self.fold, "A")
        self.assertEqual(models[0], models[1])
        self.assertEqual(a["glm_fit"], b["glm_fit"])
        self.assertEqual(a["neural_fit"], b["neural_fit"])
        return a, b

    def test_opposite_future_labels_cannot_enter_GLM_normalization_or_neural_updates(self):
        tables = copy.deepcopy(self.tables["first_last"])
        for r in tables["B"].values():
            r["positive_wait_min"] = [0. if i == r["original_choice"] else 999. for i in range(6)]
        a, b = self.assert_no_fit_leakage(tables)
        for step in ("0", "2", "4"):
            same = "same_prefix_train_future"
            other = "same_prefix_other_future"
            self.assertEqual([r["weighted_loss_sum_min"] for r in a["checkpoints"][step][same]],
                [r["weighted_loss_sum_min"] for r in b["checkpoints"][step][same]])
            self.assertNotEqual(a["checkpoints"][step][other], b["checkpoints"][step][other])

    def test_outer_held_training_half_labels_cannot_enter_fits_either(self):
        tables = copy.deepcopy(self.tables["first_last"])
        _, held = spec.population_keys(self.mean_rows, self.fold)
        for k in held:
            r = tables["A"][k]
            r["positive_wait_min"] = [0. if i == r["original_choice"] else 999. for i in range(6)]
        a, b = self.assert_no_fit_leakage(tables)
        self.assertNotEqual(a["checkpoints"]["4"]["held_prefix_train_future"], b["checkpoints"]["4"]["held_prefix_train_future"])

    def test_fold_keeps_four_pairs_and_matched_populations_without_duplicates(self):
        with patch.object(spec.stopping, "CHECKPOINTS", (0, 2, 4)):
            registered = {**self.registered, "contract": spec.contract()}
            cell = spec.fit_fold(self.bank, self.tables, self.fold, registered)
            spec.qualify(cell, self.fold, registered)
            self.assertEqual((cell["new_wait_glm_fits"], cell["neural_paths"], cell["adamw_steps"]), (4, 4, 16))
            for case in cell["cases"].values():
                self.assertEqual(len(case["checkpoints"]["4"]["same_prefix_other_future"]), 18)
                self.assertEqual(len(case["checkpoints"]["4"]["held_prefix_other_future"]), 6)
            bad = copy.deepcopy(cell)
            bad["cases"]["first_last:A"]["glm_fit"]["training_groups"].append(self.fold["held_out"])
            with self.assertRaises(ValueError):
                spec.qualify(bad, self.fold, registered)

    def test_all47_aggregation_has240_unique_held_states_and_no_policy_claim(self):
        logits, scores = np.zeros((240, 6)), np.zeros((240, 6))
        records = spec.transfer.records(self.mean_rows, self.tables["first_last"]["A"], logits, scores, 1., {})
        for r in records:
            r.update(half_sign_disagreement_candidates=0, both_halves_false_safe=0, both_halves_false_unsafe=0)
        cells = {}
        for fold in self.registered["folds"]:
            groups = [g for g in self.registered["groups"] if spec.source.source.credit.group_key(g, fold["split"]) != fold["held_out"]]
            train, held = spec.population_keys(self.registered["mean_states"], fold)
            common = {"training_groups": sorted([[g["root"], g["scene_seed"]] for g in groups]), "training_states": 6*len(groups),
                "training_half4_states": 240-len(held), "training_mean8_states": 0, "sample_weight_sum": 6*len(groups)+3*(240-len(held)), "false_safe_cost": 1.}
            populations = {p: [r for r in records if spec.state_key(r) in (train if p.startswith("same_prefix") else held)] for p in spec.POPULATIONS}
            cases = {p+":"+h: {"train_half": h, "evaluation_half": "B" if h == "A" else "A",
                "glm_fit": {**common, "parameter_count": 199, "solver_success": True, "optimizer_iterations": 0},
                "neural_fit": {**common, "parameter_count": 3393, "steps": 600, "initialization_seed": 208,
                    "parameter_update_l2": 1., "hidden_gradient_l2_max": 1.},
                "checkpoints": {str(s): populations for s in spec.stopping.CHECKPOINTS}} for p in spec.PARTITIONS for h in ("A", "B")}
            cells[fold["split"], fold["index"]] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
                "fold": fold, "cases": cases, "new_wait_glm_fits": 4, "neural_paths": 4, "adamw_steps": 2400,
                "new_cost_fits": 0, "actor_updates": 0, "new_native_episodes": 0}
        result = spec.summarize(cells, self.registered)
        self.assertEqual((result["new_wait_glm_fits"], result["neural_paths"], result["adamw_steps"]), (188, 188, 112800))
        self.assertFalse(result["third_scene_screen_eligible"] or result["native_deployment_eligible"])
        for splits in result["validation"].values():
            for directions in splits.values():
                for d in directions.values():
                    held = d["steps"]["600"]["populations"]["held_prefix_other_future"]
                    self.assertEqual((held["state_model_evaluations"], held["unique_states"], held["unique_prefix_groups"]), (240, 240, 40))
                    self.assertEqual(d["steps"]["600"]["contrasts"]["mass_weighted_excess_nats"]["held_prefix_gap_change_vs_GLM"], 0.)
        with self.assertRaises(ValueError):
            spec.summarize({}, self.registered)

    def test_scheduler_remote_small_receipts_and_python310(self):
        for task in (task_specification("unit_future", operation="prepare"), task_specification("unit_future", operation="report"),
                task_specification("unit_future", {"split": "scene", "index": 0})):
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertNotIn("_raw", task["result_dir"])
        for p in (spec.ROOT/"scripts/run_native_transit_future_transfer_stage210.py",
                spec.ROOT/"scripts/submit_native_transit_future_transfer_stage210_scheduleurm.py"):
            ast.parse(p.read_text(), feature_version=(3, 10))


if __name__ == "__main__":
    unittest.main()
