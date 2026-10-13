import ast
import copy
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from scripts import run_native_transit_training_means_stage212 as spec
from scripts.submit_native_transit_training_means_stage212_scheduleurm import task_specification, require_completed
from tests.test_native_transit_sequence_value import fixture


def original_manifest():
    groups = []
    for root in spec.ROOTS:
        for scenario in spec.SCENARIOS:
            for index in range(2):
                groups.append({"root": root, "scenario": scenario, "origin": "existing",
                    "scene_seed": 900000000+root*100+spec.SCENARIOS.index(scenario)*10+index})
            groups.extend({"root": root, "scenario": scenario, "origin": "fresh",
                "scene_seed": spec.source.scene_seed(root, scenario, i)} for i in range(8))
    means = [{"root": r, "scenario": s, "scene_seed": spec.source.scene_seed(r, s, i), "probe_index": p}
        for r, s, i, p in spec.mean_source.states()]
    return {"protocol": spec.previous.EXPERIMENT_PROTOCOL, "contract": spec.previous.contract(),
        "groups": groups, "mean_states": means, "folds": [{"split": "scene", "index": i} for i in range(47)]}


def draw_fixture(row, draw, wait=1.):
    labels = [{"candidate": i, **{f: 0. for f in spec.mean_source.expected.LABEL_FIELDS}}
        for i in range(6)]
    for label in labels:
        label["future_fleet_cost"] = 2.
        if label["candidate"] != 0:
            label["wait_delta_min"] = wait
    return {**row, "draw": draw, "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "software_qualified": True, "continuation_seed": spec.continuation_seed(row, draw),
        "original_choice": 0, "labels": labels, "existing_labels": copy.deepcopy(labels),
        "original_source_reproductions": 1, "exact_resampled_identities": 1, "prefix_state_rng_matches": 7,
        "frozen_weight_qualified_episodes": 8, "window_ledger_qualified_episodes": 8,
        "continuation_qualified": True, "native_clock_ticks": 8*61380, "training_updates": 0,
        "peak_process_rss_mb": 730.}


class TrainingMeansTest(unittest.TestCase):
    def setUp(self):
        self.old = original_manifest()
        self.registered = spec.manifest(self.old)

    def test_complement_sources_evaluation_and_outer_folds_are_fixed(self):
        states = self.registered["training_states"]
        self.assertEqual(len(states), 360)
        self.assertEqual(sum(r["origin"] == "existing" for r in states), 120)
        self.assertEqual({r["scene_index"] for r in states if r["origin"] == "existing"}, {0, 1})
        self.assertEqual({r["scene_index"] for r in states if r["origin"] == "fresh"}, {4, 5, 6, 7})
        self.assertFalse({spec.state_key(r) for r in states} & {spec.state_key(r) for r in self.old["mean_states"]})
        for name in ("groups", "mean_states", "folds"):
            self.assertEqual(self.registered[name], self.old[name])
        bad = copy.deepcopy(self.old)
        bad["groups"].append({**bad["groups"][0], "scene_seed": 123})
        with self.assertRaises(ValueError):
            spec.manifest(bad)

    def test_preflight_balanced_and_seeds_disjoint_from_all_evaluation_draws(self):
        first = spec.cells(self.registered, preflight=True)
        self.assertEqual((len(first), len(spec.cells(self.registered))), (20, 2880))
        self.assertEqual({k[:3] for k in first}, {(r, s, o) for r in spec.ROOTS for s in spec.SCENARIOS for o in ("existing", "fresh")})
        self.assertTrue(all(k[-2:] == (8, 0) for k in first))
        seeds = [spec.continuation_seed(r, d) for r in self.registered["training_states"] for d in spec.DRAWS]
        old_seeds = {spec.mean_source.continuation_seed(*k, d) for k in spec.mean_source.states() for d in spec.DRAWS}
        self.assertEqual(len(set(seeds)), 2880)
        self.assertFalse(set(seeds) & old_seeds)

    def test_mean_wait_mass_keeps_old_weight_and_identity(self):
        row = self.registered["training_states"][0]
        draws = [draw_fixture(row, d, (-1.)**d*(d+1)) for d in spec.DRAWS]
        bank_row = {**row, "labels": draws[0]["labels"], "original_choice": 0}
        for cell in draws:
            cell["existing_labels"] = copy.deepcopy(bank_row["labels"])
        mass = {**row, "original_choice": 0, "draw_count": 1,
            "positive_wait_min": [0., 1., 1., 1., 1., 1.], "negative_wait_min": [0.]*6}
        original = copy.deepcopy(mass)
        target, upgraded = spec.aggregate_state(row, draws, bank_row, mass)
        self.assertEqual((upgraded["draw_count"], upgraded["sample_weight"]), (8, 1))
        np.testing.assert_array_equal(upgraded["positive_wait_min"], [0., 2., 2., 2., 2., 2.])
        np.testing.assert_array_equal(upgraded["negative_wait_min"], [0., 2.5, 2.5, 2.5, 2.5, 2.5])
        self.assertEqual(target["labels"][1]["wait_delta_min"], -.5)
        self.assertEqual(mass, original)
        with self.assertRaises(ValueError):
            spec.aggregate_state(row, draws[:7], bank_row, mass)
        with self.assertRaises(ValueError):
            spec.aggregate_state(row, draws, bank_row, {**mass, "draw_count": 8})

    def test_draw_qualification_rejects_changed_prefix_seed_units_and_identity(self):
        row = self.registered["training_states"][0]
        cell = draw_fixture(row, 0)
        spec.qualify_draw(cell, row, 0)
        changes = [("prefix_state_rng_matches", 6), ("scene_seed", 123), ("continuation_seed", 123), ("training_updates", 1)]
        for field, value in changes:
            with self.assertRaises(ValueError):
                spec.qualify_draw({**cell, field: value}, row, 0)
        bad = copy.deepcopy(cell)
        bad["labels"][0]["wait_delta_min"] = .1
        with self.assertRaises(ValueError):
            spec.qualify_draw(bad, row, 0)
        bad = copy.deepcopy(cell)
        bad["labels"][1]["actual_cost_delta"] = 1.
        with self.assertRaises(ValueError):
            spec.qualify_draw(bad, row, 0)

    def test_each_origin_uses_its_own_original_source(self):
        existing = next(r for r in self.registered["training_states"] if r["origin"] == "existing")
        fresh = next(r for r in self.registered["training_states"] if r["origin"] == "fresh")
        with patch.object(spec.source.labels, "source_record", return_value={"scene_seed": existing["scene_seed"]}) as old, \
                patch.object(spec.source.shared, "qualify") as qualify, patch.object(spec.source, "qualify_observer"):
            spec.qualify_original({"probe": {}}, existing)
            old.assert_called_once_with(existing["root"], existing["scenario"], scene_index=0)
            self.assertEqual(qualify.call_count, 1)
        original = {k: 1 for k in spec.source.OUTCOME_KEYS}
        original["supply"] = {}
        prior = {"observer": {k: original[k] for k in spec.source.OUTCOME_KEYS}, "policy_trace": []}
        path = Mock(read_text=Mock(return_value=json.dumps(prior)))
        with patch.object(spec.noise, "source_path", return_value=path) as new, \
                patch.object(spec.source, "qualify_cell"), patch.object(spec.source, "qualify_observer"), \
                patch.object(spec.source.shared, "policy_trace", return_value=[]), patch.object(spec.source.labels, "source_record") as old:
            spec.qualify_original(original, fresh)
            self.assertEqual(new.call_args.args[-2:], (4, 8))
            old.assert_not_called()

    def test_preflight_summary_and_failed_tasks_cannot_release_full_run(self):
        rows = {spec.cell_key(r, d): r for r in self.registered["training_states"] for d in spec.DRAWS}
        summary = spec.summarize_draws(self.registered, lambda k: draw_fixture(rows[k], k[-1]), preflight=True)
        self.assertEqual((summary["draws"], summary["native_episodes"]), (20, 160))
        self.assertEqual(summary["by_origin"], {"existing": 10, "fresh": 10})
        require_completed([{"signature": "a", "status": "done"}], {"a"})
        with self.assertRaises(SystemExit):
            require_completed([{"signature": "a", "status": "failed"}], {"a"})

    def test_scheduler_only_unbound_cpu_small_results_python310(self):
        key = spec.cells(self.registered, preflight=True)[0]
        for task in (task_specification("unit_means", key), task_specification("unit_means", operation="preflight_report")):
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["cpu"], 1)
            self.assertNotIn("_raw", task["result_dir"])
        self.assertEqual(task_specification("unit_means", key)["ram_mb"], 1536)
        for path in (spec.ROOT/"scripts/run_native_transit_training_means_stage212.py",
                spec.ROOT/"scripts/submit_native_transit_training_means_stage212_scheduleurm.py"):
            ast.parse(path.read_text(), feature_version=(3, 10))

    def test_glm_bounded_loss_and_scaling_preserve_old_weights_when_depth_changes(self):
        torch.set_num_threads(1)
        rows = fixture()[0]["training_data"][:12]
        masses = {spec.state_key(r): r for r in spec.previous.source.build_masses(rows, {})}
        for r in rows[:6]:
            masses[spec.state_key(r)]["draw_count"] = 8
        upgraded = {k: {**m, "draw_count": 8, "sample_weight": m["draw_count"]} for k, m in masses.items()}
        api = spec.previous.source.feasibility
        old_glm = api.fit_wait(rows, masses, "symmetric_feasibility")
        new_glm = api.fit_wait(rows, upgraded, "symmetric_feasibility")
        self.assertEqual(new_glm["training_mean_states"], 12)
        self.assertEqual(new_glm["sample_weight_sum"], 54.)
        self.assertEqual({k: v for k, v in old_glm.items() if k != "training_mean_states"},
            {k: v for k, v in new_glm.items() if k != "training_mean_states"})
        for variant in spec.previous.bounded.FUNCTIONAL_RIDGES:
            old, curve = spec.previous.bounded.fit_path(rows, masses, old_glm, variant, (0, 2), validation=rows)
            new, other = spec.previous.bounded.fit_path(rows, upgraded, new_glm, variant, (0, 2), validation=rows)
            self.assertEqual(curve, other)
            self.assertEqual({k: v for k, v in old.items() if k not in ("reference", "training_mean_states")},
                {k: v for k, v in new.items() if k not in ("reference", "training_mean_states")})
            self.assertEqual(new["training_mean_states"], 12)


if __name__ == "__main__":
    unittest.main()
