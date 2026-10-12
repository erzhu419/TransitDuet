import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import run_native_transit_coverage_value_stage188 as spec
from scripts.analyze_native_transit_coverage_value_stage188 import qualify_fold, summarize
from scripts.submit_native_transit_coverage_value_stage188_scheduleurm import task_specification
from tests.test_native_transit_value_expansion import fixture as fresh_fixture
from tests.test_native_transit_timetable_context import fixture as old_fixture
from scripts.analyze_native_transit_value_expansion_stage187 import summarize as summarize_fresh


def bank_fixture():
    old_cells, old, _ = old_fixture()
    for row in old:
        key = row["root"], row["scenario"], 1
        row["timetable_context"] = copy.deepcopy(next(c["timetable_context"] for c in old_cells[key]["contexts"]
            if c["index"] == row["probe_index"]))
    fresh_cells, _ = fresh_fixture(preflight=False)
    fresh = summarize_fresh(fresh_cells)
    for row in [*old, *fresh["training_data"]]:
        for i, candidate in enumerate(row["candidates"]):
            candidate.update(wait_cost=i*.01, headway_cost=i*.02)
    registered = spec.manifest(old)
    bank, receipt = spec.assemble_bank(old, fresh, registered)
    return bank, receipt, old, fresh


def fake_model(rows, mode):
    model = {"training_groups": sorted(map(list, {spec.paired.scene_group(r) for r in rows})),
        "training_states": len(rows), "steps": 100, "parameter_count": spec.contract()["parameters"][mode],
        "center": [0.]*98, "scale": [1.]*98, "target_scale": [1.]*3, "regret_cost_scale": 1.,
        "parameter_update_l2": 1., "initialization_seed": spec.floor.occupancy.SEED,
        "loss_ledger": [{"step": 1}, {"step": 100}], "objective": "proper_mse", "representation": mode}
    if mode == "timetable_sequence":
        model.update(encoder_parameter_update_l2=1., sequence_projection_update_l2=1., encoder_gradient_l2_max=1.)
    return model


def fake_predictions(rows):
    return [{**{k: r[k] for k in ("root", "scenario", "scene_seed", "probe_index", "direction", "original_choice")},
        "selected": r["original_choice"], "predicted_heads": [[0., 0.]]*6,
        "predicted_wait_delta_min": [0.]*6, "absolute_fleet_cost": [1.]*6, "causal_fleet_floor": .5,
        "actual_cost_delta": 0., "fleet_cost_delta": 0., "nonfleet_cost_delta": 0., "wait_delta_min": 0.,
        "episode_reward_delta": 0., "best_realized_cost_delta": -.1, "regret": .1,
        "strict_pairs": 15, "concordant_pairs": 0, "tied_predicted_pairs": 15} for r in rows]


def fold_fixture(bank, mode, fold):
    train, test, _ = spec.split_rows(bank, fold["split"], fold["index"])
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "mode": mode, "fold": fold, "fit": fake_model(train, mode),
        "predictions": {p: fake_predictions(test) for p in spec.value.POLICIES},
        "critic_updates": 100, "actor_updates": 0, "new_native_steps": 0,
        "fit_wall_s": 1., "evaluation_wall_s": .1}


class CoverageValueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank, cls.receipt, cls.old, cls.fresh = bank_fixture()

    def test_bank_keeps_all_old_labels_and_new_groups_without_transforming_inputs(self):
        bank, receipt = self.bank, self.receipt
        self.assertEqual(bank["training_data"][:120], self.old)
        self.assertEqual(bank["training_data"][120:], self.fresh["training_data"])
        self.assertEqual((receipt["training_states"], receipt["independent_scene_groups"], receipt["paired_labels"]), (600, 100, 3600))
        self.assertEqual(receipt["event_count_range"], [12, 12])
        self.assertFalse(receipt["third_scene_screen_eligible"])
        broken = copy.deepcopy(self.fresh)
        broken["training_data"][0] = copy.deepcopy(broken["training_data"][1])
        with self.assertRaises(ValueError): spec.assemble_bank(self.old, broken, bank["manifest"])

    def test_all_callbacks_remain_together_in_all_108_folds(self):
        registered = self.bank["manifest"]
        self.assertEqual(len(registered["folds"]), 108)
        for fold in registered["folds"]:
            train, test, _ = spec.split_rows(self.bank, fold["split"], fold["index"])
            if fold["split"] == "full":
                self.assertEqual((len(train), len(test)), (600, 600))
            else:
                self.assertFalse({spec.paired.scene_group(r) for r in train} & {spec.paired.scene_group(r) for r in test})
                self.assertEqual(len(train)+len(test), 600)
                self.assertEqual(len(test), {"scene": 6, "regime": 120, "lower_root": 300}[fold["split"]])

    def test_worker_fits_only_training_rows_and_reuses_model_for_both_selectors(self):
        captured = []
        def fit(rows, method, **kwargs):
            captured.append([spec.paired.scene_group(r) for r in rows])
            self.assertEqual((method, kwargs), ("floor_absolute", {"steps": 100, "objective": "proper_mse", "representation": "timetable_sequence"}))
            return fake_model(rows, kwargs["representation"])
        with patch.object(spec.value, "fit_value", side_effect=fit), \
                patch.object(spec.floor, "evaluate", side_effect=lambda rows, model, policy: fake_predictions(rows)) as evaluate:
            result, model = spec.fit_fold(self.bank, "timetable_sequence", "scene", 0)
        held = tuple(result["fold"]["held_out"])
        self.assertNotIn(held, captured[0])
        self.assertEqual(evaluate.call_count, 2)
        self.assertTrue(all(c.args[1] is model for c in evaluate.call_args_list))
        qualify_fold(result, "timetable_sequence", result["fold"], self.bank["manifest"])

    def test_qualification_rejects_leaking_groups_dead_encoder_and_changed_budget(self):
        fold = self.bank["manifest"]["folds"][0]
        cell = fold_fixture(self.bank, "timetable_sequence", fold)
        qualify_fold(cell, "timetable_sequence", fold, self.bank["manifest"])
        for failure in ("groups", "gradient", "steps", "labels", "selector", "shared_model"):
            broken = copy.deepcopy(cell)
            if failure == "groups": broken["fit"]["training_groups"].append(fold["held_out"])
            elif failure == "gradient": broken["fit"]["encoder_gradient_l2_max"] = 0
            elif failure == "steps": broken["fit"]["steps"] = 101
            elif failure == "labels": broken["predictions"]["wait_feasible"].pop()
            elif failure == "selector": broken["predictions"]["wait_feasible"][0]["selected"] = 0
            else: broken["predictions"]["cost_only"][0]["predicted_wait_delta_min"][0] = 1.
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                qualify_fold(broken, "timetable_sequence", fold, self.bank["manifest"])

    def test_216_matched_fits_report_same_old_states_not_unmatched_sample_means(self):
        registered = self.bank["manifest"]
        cells = {(m, f["split"], f["index"]): fold_fixture(self.bank, m, f) for m in spec.MODES for f in registered["folds"]}
        old_predictions = fake_predictions(self.old)
        old_cells = {m: {"validation": {p: {s: {"predictions": old_predictions,
            "overall": spec.floor.summary(old_predictions), "by_root": {str(r): spec.floor.summary([x for x in old_predictions if x["root"] == r])
                for r in spec.collection.ROOTS}} for s in spec.SPLITS} for p in spec.value.POLICIES}} for m in spec.MODES}
        result = summarize(cells, registered, old_cells)
        self.assertEqual((result["fits"], result["critic_updates"]), (216, 21600))
        self.assertFalse(result["third_scene_screen_eligible"] or result["native_deployment_eligible"])
        for variant in result["variants"].values():
            for split in spec.SPLITS:
                self.assertEqual(variant["validation"][split]["overall"]["states"], 600)
                self.assertEqual(variant["old_state_subset"][split]["overall"]["states"], 120)
        self.assertEqual(result["same_old_state_coverage_effect"]["timetable_sequence:wait_feasible"]["scene"]["overall"]["mean_actual_cost_delta"], 0.)
        cells.pop(next(iter(cells)))
        with self.assertRaises(ValueError): summarize(cells, registered, old_cells)

    def test_scheduler_bank_stays_server_only_and_all_fit_tasks_are_unbound(self):
        tasks = [task_specification("unit_coverage", mode=m, fold=f) for m in spec.MODES for f in self.bank["manifest"]["folds"]]
        self.assertEqual(len({t["signature"] for t in tasks}), 216)
        self.assertEqual(len({t["result_dir"] for t in tasks}), 216)
        for task in [task_specification("unit_coverage"), *tasks]:
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertNotIn("bank.json", task["cmd"])
            self.assertNotIn("_raw", task["result_dir"])
            self.assertEqual([Path(p).name for p in task["stage_input_paths"]], ["scripts", "freq_hrl", "native_freqduet", "inputs"])


if __name__ == "__main__":
    unittest.main()
