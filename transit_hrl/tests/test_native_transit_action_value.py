import copy
import json
import unittest

import numpy as np

from freq_hrl.domains.transit import native_action_value as value
from scripts import run_native_transit_action_value_stage176 as spec
from scripts.submit_native_transit_action_value_stage176_scheduleurm import task_specification


def summary_fixture():
    rows, groups = [], []
    for root in (397, 401):
        for scenario in spec.source.SCENARIOS:
            scene = root*1000+spec.source.SCENARIOS.index(scenario)
            probes = (8, 9, 22, 23, 36, 37)
            groups.append({"root": root, "scenario": scenario, "scene_seed": scene, "probes": list(probes)})
            for index in probes:
                direction = index % 2 == 0
                candidates = []
                for i in range(6):
                    gap = 260.+i*35.
                    candidates.append({"fleet_cost": i*.01, "commitment": {
                        "prefix_planned_s": (8640.+np.arange(7)*gap).tolist(),
                        "prefix_wait_cost": gap/1200., "prefix_fleet_cost": i*.005}})
                state = [0.]*16
                state[9], state[12], state[11] = float(direction), .5, .01
                labels = [{"candidate": i, "fleet_cost_delta": (i-2)*.003,
                    "nonfleet_cost_delta": (i-2)*.001, "actual_cost_delta": (i-2)*.004,
                    "wait_delta_min": (i-2)*.01, "episode_reward_delta": (2-i)*10.} for i in range(6)]
                rows.append({"root": root, "scenario": scenario, "scene_seed": scene, "probe_index": index,
                    "direction": direction, "time_s": 8520.+index*100., "native_state_v2": state,
                    "context": {"queues": {"True": 5, "False": 10}, "onboard": {"True": 30, "False": 40},
                        "active": {"True": 5, "False": 6}}, "nominal_s": (8640.+np.arange(7)*360.).tolist(),
                    "original_choice": 2, "candidates": candidates, "labels": labels})
    return {"protocol": spec.source.EXPERIMENT_PROTOCOL, "contract": spec.source.contract(),
        "software_qualified": True, "training_states": 60, "training_updates": 0,
        "native_steps": 420*61380, "source_control_reproductions": 120,
        "window_ledger_qualified_episodes": 420, "training_data": rows, "fitting_groups": groups}


class ActionValueTest(unittest.TestCase):
    def test_features_are_causal_and_do_not_use_outcomes_or_group_identifiers(self):
        row = summary_fixture()["training_data"][0]
        x = value.action_features(row)
        self.assertEqual(x.shape, (6, 72))
        changed = copy.deepcopy(row)
        changed.update(root=999, scenario="other", scene_seed=0, probe_index=0, labels=[])
        changed["native_state_v2"][13] += 100  # Instantaneous HF is not an upper input.
        np.testing.assert_array_equal(value.action_features(changed), x)
        changed["candidates"][0]["commitment"]["prefix_planned_s"][1] += 20
        self.assertFalse(np.array_equal(value.action_features(changed), x))

    def test_fit_keeps_identity_zero_and_cost_heads_select_known_action(self):
        rows = summary_fixture()["training_data"]
        model = value.fit_action_value(rows)
        self.assertEqual(model["pairwise_examples"], 900)
        self.assertEqual(len(model["training_groups"]), 10)
        for row in rows:
            predicted = value.predict_action_value(row, model)
            np.testing.assert_array_equal(predicted[row["original_choice"]], [0., 0.])
            self.assertEqual(value.choose_action(row, model), 0)
            self.assertTrue(np.all(np.diff(predicted, axis=0) > 0))
        zero = {**model, "coefficients": np.zeros((72, 2)).tolist()}
        self.assertEqual(value.choose_action(rows[0], zero), rows[0]["original_choice"])

    def test_action_order_does_not_change_learned_values_or_choice(self):
        rows = summary_fixture()["training_data"]
        model = value.fit_action_value(rows)
        permutation = [5, 1, 0, 4, 2, 3]
        permuted = copy.deepcopy(rows)
        for row in permuted:
            row["candidates"] = [row["candidates"][i] for i in permutation]
            labels = row["labels"]
            row["labels"] = [{**labels[old], "candidate": new} for new, old in enumerate(permutation)]
            row["original_choice"] = permutation.index(row["original_choice"])
        other = value.fit_action_value(permuted)
        np.testing.assert_allclose(other["coefficients"], model["coefficients"], atol=1e-12)
        self.assertEqual(permutation[value.choose_action(permuted[0], other)], 0)

    def test_whole_scene_and_regime_and_lower_root_stay_out_of_fitting(self):
        rows = summary_fixture()["training_data"]
        for split, expected in (("scene", 10), ("regime", 5), ("lower_root", 2)):
            result = value.grouped_validation(rows, split)
            self.assertEqual(len(result["folds"]), expected)
            self.assertEqual(len(result["predictions"]), 60)
            for fold in result["folds"]:
                held = fold["held_out"]
                if split == "scene": self.assertNotIn(list(held), fold["training_groups"])
                elif split == "lower_root": self.assertTrue(all(g[0] != held for g in fold["training_groups"]))
                else:
                    excluded = {value.scene_group(r) for r in rows if r["scenario"] == held}
                    self.assertTrue(excluded.isdisjoint(map(tuple, fold["training_groups"])))
        changed = copy.deepcopy(rows)
        first_group = value.scene_group(rows[0])
        for row in changed:
            if value.scene_group(row) == first_group:
                for l in row["labels"]: l["actual_cost_delta"] *= -100
        a, b = (value.grouped_validation(data, "scene")["predictions"][:6] for data in (rows, changed))
        self.assertEqual([p["predicted_heads"] for p in a], [p["predicted_heads"] for p in b])

    def test_training_requires_qualified_unique_states_and_physical_label_decomposition(self):
        summary = summary_fixture()
        self.assertEqual(len(spec.training_rows(summary)), 60)
        for failure in ("qualified", "duplicate", "cost", "identity", "direction"):
            changed = copy.deepcopy(summary)
            row = changed["training_data"][0]
            if failure == "qualified": changed["software_qualified"] = False
            elif failure == "duplicate": changed["training_data"][1] = copy.deepcopy(row)
            elif failure == "cost": row["labels"][0]["actual_cost_delta"] += 1
            elif failure == "direction": row["direction"] = not row["direction"]
            else:
                row["labels"][2].update(actual_cost_delta=1, fleet_cost_delta=1)
            with self.subTest(failure=failure), self.assertRaises(ValueError): spec.training_rows(changed)
        result = spec.fit_and_validate(summary)
        json.dumps(result)
        self.assertTrue(result["native_screen_eligible"])
        self.assertEqual((result["fitted_models"], result["new_native_steps"]), (18, 0))

    def test_fit_task_is_unpinned_single_cpu_and_stages_no_historic_results(self):
        task = task_specification("unit_value_fit")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual(task["cpu"], 1)
        self.assertTrue(task["result_dir"].endswith("unit_value_fit/fit"))
        self.assertTrue(task["stage_input_paths"][-1].endswith("unit_value_fit/inputs"))
        self.assertEqual(len(task["stage_input_paths"]), 4)
        self.assertIn("OPENBLAS_NUM_THREADS=1", task["cmd"])


if __name__ == "__main__":
    unittest.main()
