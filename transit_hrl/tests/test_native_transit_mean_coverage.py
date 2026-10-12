import ast
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import run_native_transit_mean_coverage_stage195 as spec
from scripts.analyze_native_transit_mean_coverage_stage195 import summarize
from scripts.analyze_native_transit_expected_value_stage192 import qualify
from scripts.submit_native_transit_mean_coverage_stage195_scheduleurm import task_specification, require_completed
from tests.test_native_transit_continuation_noise import fixture as noise_fixture
from tests.test_native_transit_coverage_value import bank_fixture, fake_model, fake_predictions


def draws_fixture(bank):
    rows = {spec.expected.state_key(r): r for r in bank["training_data"]}
    new, old = {}, {}
    for root, scenario, replica, probe in spec.states():
        row = rows[root, spec.noise.source.scene_seed(root, scenario, replica), probe]
        for draw in spec.DRAWS:
            cell = noise_fixture((root, scenario, probe, draw))
            cell.update(scene_seed=row["scene_seed"], original_choice=row["original_choice"],
                existing_labels=copy.deepcopy(row["labels"]), labels=copy.deepcopy(row["labels"]))
            for label in cell["labels"]:
                label["future_fleet_cost"] = label["fleet_cost_delta"]+(draw % 2)/12.
                if label["candidate"] != row["original_choice"]:
                    label["wait_delta_min"] += .001*draw
            if spec.reused((root, scenario, replica, probe)):
                old[root, scenario, probe, draw] = cell
            else:
                key = root, scenario, replica, probe, draw
                cell.update(protocol=spec.EXPERIMENT_PROTOCOL, contract=spec.contract(),
                    replica=replica, continuation_seed=spec.continuation_seed(*key), peak_process_rss_mb=500.)
                cell.pop("frozen_choices")
                new[key] = cell
    return new, old


class MeanCoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank, _, _, _ = bank_fixture()
        cls.registered = spec.manifest(cls.bank["manifest"])
        cls.new, cls.old = draws_fixture(cls.bank)
        cls.targets, _ = spec.build_targets(cls.bank, cls.new, cls.old)

    def test_balanced_complete_groups_reuse_draws_without_seed_collision(self):
        self.assertEqual((len(spec.states()), len(spec.cells()), len(spec.cells(preflight=True))), (240, 1680, 10))
        all_seeds = [spec.continuation_seed(*k, d) for k in spec.states() for d in spec.DRAWS]
        self.assertEqual(len(set(all_seeds)), 1920)
        for r, s, p, d in spec.noise.cells():
            self.assertEqual(spec.continuation_seed(r, s, 0, p, d), spec.noise.continuation_seed(r, s, p, d))
        for root in spec.ROOTS:
            for scenario in spec.SCENARIOS:
                self.assertEqual(sum(k[:2] == (root, scenario) for k in spec.states()), 24)
        self.assertEqual(spec.contract()["budget"]["new_native_episodes"], 13440)

    def test_physical_mean_and_identity_preserve_all600_input_rows(self):
        self.assertEqual((len(self.targets), len(self.new), len(self.old)), (240, 1680, 240))
        rows = {spec.expected.state_key(r): r for r in self.bank["training_data"]}
        for target in self.targets:
            row = rows[spec.expected.state_key(target)]
            self.assertAlmostEqual(target["labels"][row["original_choice"]]["future_fleet_cost"], 1/24.)
            self.assertEqual(target["labels"][row["original_choice"]]["wait_delta_min"], 0.)
        replaced = spec.expected.replace_targets(self.bank["training_data"], self.targets)
        self.assertEqual(sum(a["labels"] != b["labels"] for a, b in zip(replaced, self.bank["training_data"])), 240)
        for a, b in zip(replaced, self.bank["training_data"]):
            self.assertEqual({k: v for k, v in a.items() if k != "labels"}, {k: v for k, v in b.items() if k != "labels"})

    def test_grouped_folds_hold_out_every_callback(self):
        self.assertEqual(len(self.registered["folds"]), 47)
        keys = {spec.expected.state_key(r) for r in self.targets}
        for fold in self.registered["folds"]:
            train, test, _ = spec.source.split_rows(self.bank, fold["split"], fold["index"])
            self.assertFalse({spec.source.paired.scene_group(r) for r in train} & {spec.source.paired.scene_group(r) for r in test})
            self.assertEqual(sum(spec.expected.state_key(r) in keys for r in test),
                {"scene": 6, "regime": 48, "lower_root": 120}[fold["split"]])

    def test_changed_seed_prefix_floor_or_identity_fails_before_collection_release(self):
        cell = next(iter(self.new.values()))
        spec.qualify_draw(cell)
        for field in ("continuation_seed", "scene_seed", "prefix_state_rng_matches", "native_clock_ticks"):
            changed = copy.deepcopy(cell)
            changed[field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError): spec.qualify_draw(changed)
        changed = copy.deepcopy(cell)
        changed["labels"][cell["original_choice"]]["wait_delta_min"] = .1
        with self.assertRaises(ValueError): spec.qualify_draw(changed)
        changed = copy.deepcopy(cell)
        changed["labels"][0]["future_fleet_cost"] += .1
        with self.assertRaises(ValueError): spec.qualify_draw(changed)
        missing = dict(self.new)
        missing.pop(next(iter(missing)))
        with self.assertRaises(ValueError): spec.build_targets(self.bank, missing, self.old)

    def test_fit_refits_all_heads_training_only_under_unchanged_budget(self):
        fold = self.registered["folds"][0]
        train, _, _ = spec.source.split_rows(self.bank, fold["split"], fold["index"])
        reference = fake_model(train, "tail_summary")
        with patch.object(spec.source.value, "fit_value", side_effect=lambda rows, *a, **k: fake_model(rows, "tail_summary")) as fit, \
                patch.object(spec.source.floor, "evaluate", side_effect=lambda rows, *a: fake_predictions(rows)):
            result, _ = spec.fit_fold(self.bank, self.targets, fold, reference)
        self.assertEqual(fit.call_count, 1)
        self.assertEqual(fit.call_args.args[1], "floor_absolute")
        self.assertEqual(fit.call_args.kwargs, {"steps": 100, "objective": "proper_mse", "representation": "tail_summary"})
        self.assertNotIn(tuple(fold["held_out"]), {spec.source.paired.scene_group(r) for r in fit.call_args.args[0]})
        self.assertEqual((result["training_mean_states"], result["evaluation_mean_states"]), (234, 6))
        qualify(result, "tail_summary", fold, self.registered, protocol_spec=spec)

    def test_report_counts_prefixes_not_draws_and_does_not_open_reserved_scenes(self):
        cells = {}
        keys = {spec.expected.state_key(r) for r in self.targets}
        for fold in self.registered["folds"]:
            train, test, _ = spec.source.split_rows(self.bank, fold["split"], fold["index"])
            predictions = fake_predictions([r for r in test if spec.expected.state_key(r) in keys])
            cells["tail_summary", fold["split"], fold["index"]] = {
                "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
                "mode": "tail_summary", "fold": fold, "fits": {v: fake_model(train, "tail_summary") for v in spec.VARIANTS},
                "predictions": {v: {label: {p: predictions for p in spec.source.value.POLICIES}
                    for label in ("original", "mean8")} for v in spec.VARIANTS},
                "training_mean_states": 240-len(predictions), "evaluation_mean_states": len(predictions),
                "critic_updates": 100, "actor_updates": 0, "new_native_steps": 0, "fit_wall_s": 1.}
        prepared = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
            "input_states": 600, "mean_states": 240, "prefix_groups": 40, "new_draws": 1680,
            "reused_draws": 240, "new_native_episodes": 13440, "mean_noise": {}}
        result = summarize(cells, self.registered, prepared)
        self.assertEqual((result["new_fits"], result["critic_updates"], result["evaluation_prefix_scene_groups"]), (47, 4700, 40))
        self.assertFalse(result["expected_target_development_gate"] or result["third_scene_screen_eligible"] or result["native_deployment_eligible"])

    def test_scheduler_is_unbound_small_sync_and_qualification_precedes_more_tasks(self):
        tasks = [task_specification("unit_mean", key=k) for k in spec.cells()]
        self.assertEqual(len({t["signature"] for t in tasks}), 1680)
        self.assertEqual(len({t["result_dir"] for t in tasks}), 1680)
        for task in [*tasks, task_specification("unit_mean"), task_specification("unit_mean", fold=self.registered["folds"][0]),
                task_specification("unit_mean", operation="report")]:
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertNotIn("_raw", task["result_dir"])
            self.assertNotIn("bank.json", task["cmd"])
        first = [dict(t, status="done") for t in tasks[:10]]
        require_completed(first, {t["signature"] for t in first})
        first[0]["status"] = "running"
        with self.assertRaises(SystemExit): require_completed(first, {t["signature"] for t in first})

    def test_changed_scripts_parse_on_remote_python310(self):
        for name in ("run_native_transit_mean_coverage_stage195.py", "submit_native_transit_mean_coverage_stage195_scheduleurm.py",
                "analyze_native_transit_mean_coverage_stage195.py", "run_native_transit_continuation_noise_stage191.py"):
            ast.parse((spec.ROOT/"scripts"/name).read_text(), feature_version=(3, 10))


if __name__ == "__main__":
    unittest.main()
