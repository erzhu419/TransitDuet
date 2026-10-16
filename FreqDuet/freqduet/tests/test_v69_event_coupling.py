import copy
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v69_event_coupling as event
from lower.resac_lagrangian import CategoricalPolicy


class EventCouplingTest(unittest.TestCase):
    def test_panel_rotates_windows_without_selecting_outcomes(self):
        self.assertEqual((len(event.CONTEXTS), len(event.ITEMS)), (32, 256))
        self.assertEqual(len(event.ITEMS) * len(event.joint.ACTIONS), 1792)
        for reference, (_, day) in enumerate(event.labels.REFERENCES):
            selected = [window for c in event.CONTEXTS for ref, window in [event.labels.CONTEXTS[c]]
                        if ref == reference]
            self.assertEqual(selected, [day % 3])
        seeds = {event.labels.context_spec(event.ITEMS[i][0])["train_seed"] for i in event.SMOKE_ITEMS}
        self.assertEqual(len(seeds), 2)

    def test_inverse_cdf_preserves_masses_and_skips_masked_actions(self):
        probabilities = np.array([0., 1., 3., 0., 6., 0., 0.])
        counts = np.bincount([event.categorical_index(probabilities, u)
                              for u in np.arange(.0005, 1., .001)], minlength=7)
        np.testing.assert_array_equal(counts, [0, 100, 300, 0, 600, 0, 0])
        self.assertEqual(event.categorical_index(probabilities, 0.), 1)
        self.assertEqual(event.categorical_index(probabilities, .1), 2)
        self.assertEqual(event.categorical_index(probabilities, .4), 4)

    def test_event_uniforms_ignore_global_order_but_not_local_occurrence_or_seed(self):
        left, right = event.EventUniforms(69), event.EventUniforms(69)
        first = left.draw(1, 4, True)
        other = left.draw(2, 3, False)
        self.assertEqual(right.draw(2, 3, False), other)
        self.assertEqual(right.draw(1, 4, True), first)
        repeated = left.draw(1, 4, True)
        self.assertEqual(repeated[0], (1, 4, 1, 1))
        self.assertNotEqual(first[1], repeated[1])
        self.assertEqual(repeated, right.draw(1, 4, True))
        self.assertNotEqual(first, event.EventUniforms(70).draw(1, 4, True))

    def test_sampler_keeps_weights_mask_probabilities_and_all_external_rngs(self):
        policy = CategoricalPolicy(3, event.joint.ACTIONS, hidden_dim=8, sample_seed=42,
                                   action_limit_feature_index=2)
        state = np.array([0., 1., 10. / 45.], dtype=np.float32)
        bus = SimpleNamespace(trip_id=7, last_station=SimpleNamespace(station_id=2), direction=True)
        runner = SimpleNamespace(_label_agent=1, _bus_for_agent=lambda _: bus,
                                 env=SimpleNamespace(current_time=100.))
        weights, private = copy.deepcopy(policy.state_dict()), policy.sampling_state().clone()
        with torch.no_grad():
            probabilities = policy.dist_info(torch.as_tensor(state)[None, :])[0].clone()
        numpy_before, python_before, torch_before = np.random.get_state(), random.getstate(), torch.get_rng_state()
        uniforms = event.EventUniforms(69)
        uniforms.install(runner, policy)
        action = policy.get_action(state)
        self.assertIn(float(action), (0., 5., 10.))
        self.assertEqual(action.shape, ())
        self.assertEqual(action.dtype, np.float32)
        record = uniforms.records[(7, 2, 1, 0)]
        expected = event.joint.ACTIONS[event.categorical_index(probabilities[0].numpy(), record["uniform"])]
        self.assertEqual(float(action), expected)
        self.assertEqual(float(policy.get_action(state, deterministic=True)),
                         event.joint.ACTIONS[int(probabilities.argmax())])
        self.assertEqual(len(uniforms.records), 1)
        event.joint.assert_same_state(weights, policy.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(probabilities, policy.dist_info(torch.as_tensor(state)[None, :])[0], rtol=0, atol=0)
        event.joint.assert_same_state(numpy_before, np.random.get_state())
        self.assertEqual(python_before, random.getstate())
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        self.assertTrue(torch.equal(private, policy.sampling_state()))

    def test_alignment_records_reordering_and_rejects_different_common_uniforms(self):
        record = dict(uniform=.4, order=0, time_s=10., action_s=5.)
        key = (1, 2, 0, 0)
        native = {key: record}
        alternate = {key: dict(record, order=1, time_s=15., action_s=10.),
                     (2, 3, 0, 0): dict(record)}
        aligned = event.alignment(native, alternate)
        self.assertEqual(aligned, dict(common_events=1, uniform_mismatches=0, order_shifted=1,
                                      time_shifted=1, action_differences=1, native_only=0, alternative_only=1))
        alternate[key]["uniform"] = .5
        with self.assertRaises(AssertionError):
            event.alignment(native, alternate)

    def test_variance_ratio_uses_eight_original_day_clusters(self):
        contexts = [dict(scenario_seed=day, noise=dict(event=dict(metric=i + 1.), stream=dict(metric=4. * (i + 1.))))
                    for i, day in enumerate(event.labels.DEMAND_SEEDS) for _ in range(4)]
        result = event.variance_comparison(contexts, "metric")
        self.assertEqual(result["ratio"], .25)
        self.assertEqual(result["ratio_ci95"], [.25, .25])
        self.assertEqual(result["difference"]["n_demand_days"], 8)
        self.assertEqual(result["event_noise_variance"], 4.5)

    def test_aggregate_includes_all_action_constraints_and_excludes_native_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root, v67, v68 = [Path(directory) / name for name in ("event", "labels", "stream")]
            (v67 / "labels").mkdir(parents=True)
            for index, (context, future) in enumerate(event.ITEMS):
                spec = event.labels.context_spec(context)
                stream_rows = []
                for ai, action in enumerate(event.joint.ACTIONS):
                    row = {metric: float(ai * (future + 1)) for metric in event.repeated.RESPONSE_METRICS}
                    row.update(action_s=action, trips_unlaunched=0., trips_incomplete=0., conservation_max_error=0.,
                               alignment=None if ai == 0 else dict(common_events=1, uniform_mismatches=0),
                               repeat_trip_stop_calls=0)
                    stream_rows.append(row)
                    if future == 0:
                        (v67 / "labels" / f"c{context}_a{action}.json").write_text(json.dumps(row))
                group = dict(**spec, future_index=future, native_action_s=0, exact_prefix=True,
                             common_future_start=True, forced_native_verified=index in event.SMOKE_ITEMS, rows=stream_rows)
                old_path = event.old_group_path(v68, context, future)
                old_path.parent.mkdir(parents=True, exist_ok=True)
                old_path.write_text(json.dumps(group))
                event_rows = copy.deepcopy(stream_rows)
                for row in event_rows:
                    for metric in event.METRICS:
                        row[metric] /= 2
                path = event.group_path(root, index)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(group, rows=event_rows)))
            event.aggregate(root, v67, v68)
            result = json.loads((root / "summary" / "status.json").read_text())
            self.assertEqual(result["alignment"]["common_events"], 256 * 6)
            self.assertEqual(result["forced_native_smoke_groups"], 2)
            self.assertEqual(result["variance"]["passenger_person_s"]["ratio"], .25)
            self.assertEqual(result["cross_future"]["event"]["passenger_person_s"]["mean"], 0.)
            self.assertFalse(result["main_policy_changed"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
