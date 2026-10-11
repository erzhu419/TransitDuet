import contextlib
import copy
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.domains.transit import native_timetable_encoder as encoder
from freq_hrl.domains.transit.native_timetable_context import event_features
from scripts import run_native_transit_sequence_value_stage186 as spec
from scripts.analyze_native_transit_sequence_value_stage186 import summarize
from scripts.submit_native_transit_sequence_value_stage186_scheduleurm import task_specification
from tests.test_native_transit_timetable_context import fixture as context_fixture


def fixture():
    cells, rows, _ = context_fixture()
    by_key = {(c["seed"], c["scene_seed"], ctx["index"]): ctx["timetable_context"]
        for c in cells.values() for ctx in c["contexts"]}
    for r in rows:
        for i, c in enumerate(r["candidates"]):
            c.update(wait_cost=.3+i*.01, headway_cost=i*.02)
    data = {"protocol": spec.source.EXPERIMENT_PROTOCOL, "contract": spec.source.contract(),
        "software_qualified": True, "source_control_reproductions": 20, "training_states": 120,
        "paired_labels_reused": 720, "new_native_episodes": 20, "new_native_steps": 1227600, "training_updates": 0,
        "training_data": [{**r, "timetable_context": by_key[r["root"], r["scene_seed"], r["probe_index"]]} for r in rows]}
    return data, rows


class SequenceValueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_masked_encoding_and_numpy_serialization_match_for_ragged_sequences(self):
        data, _ = fixture()
        rows = copy.deepcopy(data["training_data"][:2])
        rows[1]["timetable_context"]["pending_trips"] += [{"trip_id": 101, "direction": False,
            "nominal_s": 13000., "scheduled_s": 13000., "committed": True, "target_headway_s": 360.}]
        center, scale = encoder.training_scales(rows)
        module = encoder.TimetableEncoder()
        model = {"center": center.tolist(), "scale": scale.tolist(),
            "weights": {k: v.detach().numpy().tolist() for k, v in module.state_dict().items()}}
        with torch.no_grad():
            together = module(*encoder.batch_events(rows, center, scale)).numpy().reshape(2, 6, 16)
            alone = module(*encoder.batch_events(rows[:1], center, scale)).numpy()
        np.testing.assert_allclose(together[0], alone, atol=1e-7, rtol=1e-6)
        for row, encoded in zip(rows, together):
            np.testing.assert_allclose(encoder.encode_numpy(row, json.loads(json.dumps(model))), encoded, atol=1e-7, rtol=1e-5)

    def test_zero_projection_preserves_control_and_encoder_gets_loss_gradients(self):
        rows = fixture()[0]["training_data"][:6]
        rng = torch.get_rng_state().clone()
        models = [spec.value.fit_value(rows, "floor_absolute", steps=0, objective="proper_mse", representation=r)
            for r in ("tail_summary", "timetable_sequence")]
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        old, new = models
        self.assertEqual((old["parameter_count"], new["parameter_count"]), (1635, 2147))
        np.testing.assert_array_equal(np.asarray(new["weights"]["0.weight"])[:, :98], old["weights"]["0.weight"])
        np.testing.assert_array_equal(np.asarray(new["weights"]["0.weight"])[:, 98:], np.zeros((16, 16)))
        for key in ("predicted_heads", "predicted_wait_delta_min", "absolute_fleet_cost"):
            np.testing.assert_allclose(spec.value.predict(rows[0], old)[key], spec.value.predict(rows[0], new)[key], atol=1e-12)
        one = spec.value.fit_value(rows, "floor_absolute", steps=1, objective="proper_mse", representation="timetable_sequence")
        self.assertEqual(one["encoder_gradient_l2_max"], 0.)
        self.assertGreater(one["encoder_parameter_update_l2"], 0.)  # Weight decay is not proof of learning.
        trained = spec.value.fit_value(rows, "floor_absolute", steps=4, objective="proper_mse", representation="timetable_sequence")
        self.assertGreater(trained["encoder_gradient_l2_max"], 0.)
        self.assertGreater(trained["sequence_projection_update_l2"], 0.)

    def test_event_scales_are_train_only_and_inference_ignores_labels_group_ids(self):
        rows = fixture()[0]["training_data"][:12]
        group = spec.paired.scene_group(rows[0])
        train = [r for r in rows if spec.paired.scene_group(r) != group]
        model = spec.value.fit_value(train, "floor_absolute", steps=3, objective="proper_mse", representation="timetable_sequence")
        center, scale = encoder.training_scales(train)
        np.testing.assert_array_equal(model["event_encoder"]["center"], center)
        np.testing.assert_array_equal(model["event_encoder"]["scale"], scale)
        changed = copy.deepcopy(rows[0])
        changed.update(labels=[], root=0, scenario="unused", scene_seed=0)
        prediction = spec.value.predict(rows[0], model)
        self.assertEqual(prediction, spec.value.predict(changed, json.loads(json.dumps(model))))
        self.assertEqual(prediction["predicted_heads"][rows[0]["original_choice"]], [0., 0.])
        self.assertTrue(all(c >= prediction["causal_fleet_floor"] for c in prediction["absolute_fleet_cost"]))
        changed["timetable_context"]["candidate_planned_s"][0][-2] += 1000.
        self.assertEqual(model["event_encoder"]["center"], center.tolist())
        self.assertNotEqual(spec.value.predict(changed, model), prediction)

    def test_complete_sequence_reaches_trained_value_beyond_equal_base_summaries(self):
        rows = fixture()[0]["training_data"][:6]
        row = copy.deepcopy(rows[0])
        row["candidates"][4] = copy.deepcopy(row["candidates"][3])
        table = row["timetable_context"]
        table["candidate_planned_s"][4] = table["candidate_planned_s"][3].copy()
        table["candidate_planned_s"][4][8] += 60.
        np.testing.assert_array_equal(spec.value.features(row, "timetable_sequence")[3], spec.value.features(row, "timetable_sequence")[4])
        self.assertFalse(np.array_equal(event_features(row)[3], event_features(row)[4]))
        model = spec.value.fit_value(rows, "floor_absolute", steps=4, objective="proper_mse", representation="timetable_sequence")
        prediction = spec.value.predict(row, model)
        self.assertFalse(np.array_equal(prediction["predicted_heads"][3], prediction["predicted_heads"][4]))

    def test_grouped_fit_is_matched_to_control_and_cannot_swap_primary(self):
        data, original = fixture()
        def control_fit(rows, method):
            return spec.value.fit_value(rows, method, steps=2, objective="proper_mse", representation="tail_summary")
        with contextlib.redirect_stdout(io.StringIO()):
            control = spec.floor_source.fit_rows_and_validate(original, "floor_absolute", fitter=control_fit)
            control.update(protocol=spec.prior.EXPERIMENT_PROTOCOL, contract=spec.prior.contract(), training_mode="tail_summary")
            cell = spec.fit_and_validate(data, original, control)
        result = summarize(cell, control)
        self.assertEqual(result["critic_updates"], 56)
        self.assertFalse(result["native_deployment_eligible"])
        for failure in ("gradients", "groups", "steps", "scales", "primary"):
            changed = copy.deepcopy(cell)
            if failure == "gradients": changed["fit_ledger"][0]["encoder_gradient_l2_max"] = 0.
            elif failure == "groups": changed["fit_ledger"][0]["training_groups"].pop()
            elif failure == "steps": changed["fit_ledger"][0]["steps"] += 1
            elif failure == "scales": changed["fit_ledger"][0]["target_scale"][0] += 1.
            else: changed["third_scene_screen_eligible"] = not changed["third_scene_screen_eligible"]
            with self.subTest(failure=failure), self.assertRaises(ValueError): summarize(changed, control)
        changed = copy.deepcopy(data)
        changed["training_data"][0]["labels"][0]["wait_delta_min"] += 1.
        with self.assertRaises(ValueError): spec.training_rows(changed, original)

    def test_scheduler_keeps_real_fitting_off_local_and_unbound(self):
        task = task_specification("unit_sequence_value")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
        self.assertEqual(len(task["stage_input_paths"]), 4)
        self.assertIn("run_native_transit_sequence_value_stage186.py", task["cmd"])


if __name__ == "__main__":
    unittest.main()
