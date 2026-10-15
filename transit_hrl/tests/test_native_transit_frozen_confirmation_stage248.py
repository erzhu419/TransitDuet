import copy
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_frozen_confirmation as learning
from scripts import run_native_transit_frozen_confirmation_stage248 as spec
from scripts.submit_native_transit_frozen_confirmation_stage248_scheduleurm import task_specification
from scripts import submit_native_transit_frozen_confirmation_stage248_scheduleurm as submit


class FrozenConfirmationTest(unittest.TestCase):
    def setUp(self):
        rows = [{"root": 397, "scene_seed": 1, "probe_index": p, "scenario": "low_noise"} for p in (8, 9)]
        choices = {v: [0, 3] for v in learning.VARIANTS}
        choices[learning.PRIMARY] = [2, 3]
        choices["pair_gls_pooled_safe"] = [1, 3]
        self.cells = [{"key": ["mixed_a", 0], "blocks": [{"states": rows, "choices": choices}]}]

    def test_union_and_zero_states_preserve_full_denominator(self):
        self.assertEqual(learning.action_union(self.cells), {(397, 1, 8): [0, 1, 2]})
        def load(key, draw):
            self.assertEqual(key, (397, 1, 8))
            return [{"candidate": c, **dict(zip(learning.FIELDS, [-c, 0., -c, -2*c]))} for c in range(3)]
        panels = learning.sparse_panels(self.cells, range(64), load)
        stats = learning.summarize(self.cells, panels)["variants"][learning.PRIMARY]
        self.assertEqual(stats["fields"]["cost"]["mean"], -1.)
        self.assertEqual(stats["fields"]["wait_min"]["mean"], -2.)
        self.assertEqual(stats["changed_states"], 1)
        with self.assertRaisesRegex(ValueError, "Missing frozen-action"):
            learning.sparse_panels(self.cells, range(64), lambda k, d: load(k, d)[:2])

    def test_paired_sampling_deduplicates_repeated_exposures(self):
        values = np.zeros((64, 6, 4))
        values[:, 1, 0] = np.arange(64)
        values[:, 2, 0] = np.arange(64)+4.
        panels = {(397, 1, 8): {"values": values}, (397, 1, 9): {"values": np.zeros_like(values)}}
        one = learning.summarize(self.cells, panels)
        repeated = learning.summarize(self.cells*4, panels)
        self.assertEqual(one["variants"], repeated["variants"])
        paired = learning.summarize(self.cells, panels, baseline="pair_gls_pooled_safe")
        self.assertNotIn("raw", paired["variants"])
        self.assertEqual(paired["variants"][learning.PRIMARY]["fields"]["cost"]["mean"], 2.)
        self.assertEqual(paired["variants"][learning.PRIMARY]["fields"]["cost"]["conditional_mc_se"], 0.)

    def test_future_labels_do_not_reselect_actions(self):
        before = copy.deepcopy(self.cells)
        values = np.zeros((64, 6, 4))
        panels = {(397, 1, 8): {"values": values}, (397, 1, 9): {"values": values.copy()}}
        with patch.object(spec.previous.learning, "select", side_effect=AssertionError("Reselection")):
            a = learning.summarize(self.cells, panels)
            values[:, 2, 0] = 10.
            b = learning.summarize(self.cells, panels)
        self.assertNotEqual(a, b)
        self.assertEqual(self.cells, before)

    def test_64_new_seeds_no_probe_collisions_or_old_overlap(self):
        keys = [(*st, d) for st in spec.coverage.states() for d in spec.DRAWS]
        seeds = {spec.continuation_seed(*k) for k in keys}
        old = {spec.old_future.continuation_seed(*st, d) for st in spec.coverage.states() for d in range(8)}
        old.update(spec.coverage.continuation_seed(r, s, i, p, d) for r in spec.ROOTS for s in spec.SCENARIOS
            for i in range(10) for p in spec.coverage.PROBES for d in range(8))
        self.assertEqual(len(seeds), 240*64)
        self.assertFalse(seeds & old)
        self.assertEqual(spec.contract(), json.loads(json.dumps(spec.contract())))

    def test_preflight_subset_and_scheduler_stay_on_linux_unbound(self):
        sealed = {"active": [{"key": list(st)} for st in spec.coverage.states()]}
        self.assertEqual(len(spec.cells(sealed, preflight=True)), 10)
        self.assertEqual(len(spec.cells(sealed)), 240*64)
        for operation, key in (("seal", None), ("preflight_report", None), ("collect", spec.cells(sealed)[0])):
            task = task_specification("unit_confirmation", operation=operation, key=key)
            self.assertEqual(task["cpu"], 1)
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertNotIn("_raw", task["result_dir"])
            self.assertFalse(any("/results/" in p for p in task["stage_input_paths"]))

    def test_sparse_physical_labels_and_confirmation_gate(self):
        base = {"service_cost_restricted": 2., "peak_fleet": 13,
            "restricted_wait_horizon_min": 4., "ep_reward": -100.}
        branches = [{**base, "probe": {"selected": 1}}, {**base, "probe": {"selected": 4},
            "peak_fleet": 12, "service_cost_restricted": 1., "restricted_wait_horizon_min": 3., "ep_reward": -90.}]
        labels = spec.labels_from_rollouts(base, branches, 12)
        self.assertEqual([r["candidate"] for r in labels], [1, 4])
        self.assertAlmostEqual(labels[1]["fleet_cost_delta"], -1/12)
        self.assertEqual(labels[1]["physical_return_delta"], 100.)
        key = (*spec.coverage.states()[0], 0)
        item = {"key": list(key[:4]), "candidates": [1, 4]}
        r = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "key": list(key),
            "software_qualified": True, "continuation_seed": spec.continuation_seed(*key),
            "candidates": [1, 4], "original_choice": 1, "labels": labels, "original_source_reproductions": 1,
            "exact_resampled_identities": 1, "prefix_state_rng_matches": 3, "frozen_weight_qualified_episodes": 4,
            "window_ledger_qualified_episodes": 4, "continuation_qualified": True, "native_clock_ticks": 4*61380, "training_updates": 0}
        spec.qualify_draw(r, key, item)
        r["labels"][1]["fleet_cost_delta"] = 0.
        with self.assertRaisesRegex(ValueError, "physical target"): spec.qualify_draw(r, key, item)
        primary = {"fields": {f: {"conditional_mc_normal_interval95": [-2., -1.]} for f in ("cost", "wait_min")}}
        half = {"variants": {learning.PRIMARY: {"fields": {f: {"mean": -1.} for f in ("cost", "wait_min")}}}}
        self.assertTrue(all(learning.confirm(primary, [half, half]).values()))
        half["variants"][learning.PRIMARY]["fields"]["cost"]["mean"] = .1
        self.assertFalse(learning.confirm(primary, [half, half])["cost"])

    def test_remaining_fast_submit_still_requires_exact_completed_phase(self):
        registered = {"contract": spec.contract(), "coverage_qualified": True,
            "active": [{"key": list(st)} for st in spec.coverage.states() if st[1:] == ("low_noise", 0, 8)]}
        initial = spec.cells(registered, preflight=True)
        previous = [task_specification("unit_confirmation"),
            *[task_specification("unit_confirmation", key=k) for k in initial],
            task_specification("unit_confirmation", operation="preflight_report")]
        records = [{**t, "id": f"t{i}", "status": "done", "exit_code": 0} for i, t in enumerate(previous)]
        runtime = SimpleNamespace(load_state=lambda: {"tasks": records})
        with patch.object(sys, "argv", ["submit", "--run-name", "unit_confirmation", "--phase", "remaining"]), \
                patch.object(submit, "inventory", return_value=records), \
                patch.object(submit.Path, "read_text", return_value=json.dumps(registered)), \
                patch.dict(sys.modules, {"scheduler": runtime}), patch.object(submit, "execute_bulk") as bulk:
            submit.main()
            tasks = bulk.call_args.args[0]
            self.assertEqual(len(tasks), 126)
            self.assertEqual(len({t["signature"] for t in tasks}), len(tasks))
            self.assertFalse({t["signature"] for t in tasks} & {t["signature"] for t in records})
            self.assertTrue(all(t["allow_duplicate"] for t in tasks))
            bulk.reset_mock()
            records[-1]["exit_code"] = 1
            with self.assertRaises(SystemExit): submit.main()
            bulk.assert_not_called()
            records[-1]["exit_code"] = 0
            records.append({**tasks[0], "id": "already_submitted", "status": "queued"})
            with self.assertRaises(SystemExit): submit.main()
            bulk.assert_not_called()


if __name__ == "__main__": unittest.main()
