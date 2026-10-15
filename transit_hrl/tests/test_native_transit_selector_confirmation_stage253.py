import copy
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_selector_confirmation as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory
from scripts import run_native_transit_selector_confirmation_stage253 as spec
from scripts import submit_native_transit_selector_confirmation_stage253_scheduleurm as submit


class SelectorConfirmationTest(unittest.TestCase):
    def setUp(self):
        self.st = spec.coverage.states()[0]
        root, scenario, replica, probe = self.st
        scene = spec.source.scene_seed(root, scenario, replica)
        self.rows = [{"root": root, "scene_seed": scene, "probe_index": p, "scenario": scenario} for p in (probe, probe+1)]
        choices = {v: [0, 0] for v in learning.VARIANTS}
        for m in learning.MODELS: choices[f"{m}_expanded8_pooled_safe"] = [1, 0]
        self.cells = [{"key": ["mixed_a", 0, "even_odd", 0, b, scope],
            "blocks": [{"states": self.rows, "choices": copy.deepcopy(choices)}]}
            for b in (0, 1) for scope in ("train60", "held60")]

    def test_sparse_pairs_shared_events_and_full_denominator(self):
        def load(key, draw):
            return [{"candidate": c, **dict(zip(learning.frozen.FIELDS, [-c, -c if draw % 2 else c, -c-(-c if draw % 2 else c), -2*c]))}
                for c in (0, 1)]
        panels = learning.frozen.sparse_panels(self.cells, range(64), load, variants=learning.VARIANTS)
        before = copy.deepcopy(self.cells)
        with patch.object(spec.previous.learning, "select", side_effect=AssertionError("Reselection")):
            stats = learning.summarize(self.cells, panels)
        self.assertEqual(self.cells, before)
        primary = stats["primary"]["free_old"]
        self.assertEqual(primary["fields"]["cost"]["mean"], -.5)
        self.assertEqual(primary["fields"]["wait_min"]["mean"], -1.)
        a = learning.attribution(self.cells, panels, "free_old_expanded8_pooled_safe", "free_old_matched2_pooled_safe")
        self.assertEqual(a, learning.attribution(self.cells*4, panels, "free_old_expanded8_pooled_safe", "free_old_matched2_pooled_safe"))
        self.assertEqual(a["prefixes"][0]["fleet_events_beneficial"], 32)
        self.assertEqual(a["prefixes"][0]["fleet_events_harmful"], 32)
        self.assertEqual(a["prefixes"][0]["contribution"]["cost"], -.5)
        with self.assertRaisesRegex(ValueError, "Missing frozen-action"):
            learning.frozen.sparse_panels(self.cells, range(64), lambda k, d: load(k, d)[:1], variants=learning.VARIANTS)

    def test_seal_copies_actions_without_future_reads_or_inference(self):
        states = spec.coverage.states()[:60]
        rows = [{"root": r, "scene_seed": spec.source.scene_seed(r, s, i), "probe_index": p,
            "scenario": s, "original_choice": 0} for r, s, i, p in states]
        blocks = [{"bank": b, "scope": scope, "states": [list(learning.frozen.state_key(r)) for r in rows],
            "choices": {v: [int(j == 0 and v == 'free_old_expanded8_pooled_safe') for j in range(60)] for v in learning.VARIANTS}}
            for b in (0, 1) for scope in ("train60", "held60")]
        bank_path = spec.coverage.source.bank_path(spec.coverage.SOURCE_RUN)
        def read(path):
            if path == bank_path: return json.dumps({"training_data": rows})
            if path.name == "actions.json": return json.dumps(blocks)
            if path.name == "aggregate.json": return json.dumps({"protocol": spec.previous.EXPERIMENT_PROTOCOL,
                "contract": spec.previous.contract(), "coverage_qualified": True})
            raise AssertionError(f"Unexpected read {path}")
        with patch.object(spec.coverage, "states", return_value=states), patch.object(spec.previous, "KEYS", [self.cells[0]["key"][:4]]), \
                patch.object(spec.Path, "read_text", autospec=True, side_effect=read), patch.object(spec.Path, "is_file", return_value=True), \
                patch.object(spec.source.native, "load_source", return_value=(None, SimpleNamespace(is_dir=lambda: True))), \
                patch.object(spec.replay, "replay_draw", side_effect=AssertionError("Future sampling before seal")), \
                patch.object(spec, "write_json") as write:
            spec.seal("unit_selector_confirmation")
        sealed, receipt = [c.args[1] for c in write.call_args_list]
        self.assertEqual(sealed["cells"][0]["blocks"][0]["choices"], blocks[0]["choices"])
        self.assertEqual(receipt["active_states"], 1)
        self.assertEqual(receipt["native_episodes"], 4*64)

    def test_new_seed_range_and_chunk_partition(self):
        keys = [(*st, d) for st in spec.coverage.states() for d in spec.DRAWS]
        seeds = {spec.continuation_seed(*k) for k in keys}
        old = {spec.replay.continuation_seed(*k) for k in keys}
        training = spec.previous.information
        old.update(training.continuation_seed(*st, d) for st in spec.coverage.states() for d in training.DRAWS)
        old.update(spec.replay.old_future.continuation_seed(*st, d) for st in spec.coverage.states() for d in range(8))
        self.assertEqual(len(seeds), 240*64)
        self.assertFalse(seeds & old)
        sealed = {"active": [{"key": list(st)} for st in spec.coverage.states()[:2]]}
        draws = spec.preflight_keys(sealed)+[d for k in spec.chunk_keys(sealed) for d in spec.chunk_draws(sealed, k)]
        self.assertEqual(len(draws), 128)
        self.assertEqual(len(set(draws)), 128)
        self.assertEqual(spec.contract(), json.loads(json.dumps(spec.contract())))

    def test_report_and_joint_gate_do_not_pick_a_winner(self):
        item = {"key": list(self.st), "state": list(learning.frozen.state_key(self.rows[0])), "candidates": [0, 1]}
        sealed = {"cells": self.cells, "active": [item]}
        outcomes = {}
        for d in spec.DRAWS:
            key = (*self.st, d)
            labels = [{"candidate": c, "actual_cost_delta": -.001*c, "fleet_cost_delta": 0.,
                "nonfleet_cost_delta": -.001*c, "wait_delta_min": -.002*c,
                "physical_return_delta": .1*c, "episode_reward_delta": .1*c, "future_fleet_cost": 0.} for c in (0, 1)]
            r = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "key": list(key),
                "software_qualified": True, "continuation_seed": spec.continuation_seed(*key), "original_choice": 0,
                "candidates": [0, 1], "labels": labels, "original_source_reproductions": 1, "exact_resampled_identities": 1,
                "prefix_state_rng_matches": 3, "frozen_weight_qualified_episodes": 4, "window_ledger_qualified_episodes": 4,
                "continuation_qualified": True, "native_clock_ticks": 4*61380, "training_updates": 0,
                "wall_seconds": 1., "peak_process_rss_mb": 100.}
            spec.qualify_draw(r, key, item)
            outcomes[raw_directory(spec.draw_path("unit_selector_confirmation", *key))/"result.json"] = r
        with patch.object(spec, "load_seal", return_value=sealed), patch.object(spec, "VIEWS", ("mixed_a",)), \
                patch.object(spec.Path, "read_text", autospec=True, side_effect=lambda p: json.dumps(outcomes[p])), \
                patch.object(spec, "write_json") as write:
            spec.report("unit_selector_confirmation")
        result = write.call_args.args[1]
        self.assertTrue(result["confirmation"]["joint_pass"])
        self.assertEqual(result["native_episodes"], 256)
        banks = result["groups"]["mixed_pooled"]["held60"]
        self.assertAlmostEqual(banks["pooled"]["fresh64"]["primary"]["free_old"]["fields"]["cost"]["mean"], -.0005)
        banks["bank1"]["fresh64"]["primary"]["shrunk"]["fields"]["wait_min"]["mean"] = .01
        self.assertFalse(learning.confirm(banks)["joint_pass"])
        self.assertFalse(result["default_changed"])
        bad = copy.deepcopy(next(iter(outcomes.values())))
        bad["continuation_seed"] -= 1
        with self.assertRaisesRegex(ValueError, "physical budget"): spec.qualify_draw(bad, tuple(bad["key"]), item)

    def test_linux_resources_and_remaining_phase_gate(self):
        sealed = {"contract": spec.contract(), "coverage_qualified": True, "active": [{"key": list(self.st)}]}
        initial = [submit.task_specification("unit_selector_confirmation", operation="preflight", key=k) for k in spec.preflight_keys(sealed)]
        previous = [submit.task_specification("unit_selector_confirmation"), *initial,
            submit.task_specification("unit_selector_confirmation", operation="preflight_report")]
        records = [{**t, "id": f"t{i}", "status": "done", "exit_code": 0} for i, t in enumerate(previous)]
        for task in previous:
            self.assertEqual(task["cpu"], 1)
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertNotIn("_raw", task["result_dir"])
            self.assertFalse(any("/results/" in p for p in task["stage_input_paths"]))
        runtime = SimpleNamespace(load_state=lambda: {"tasks": records})
        with patch.object(sys, "argv", ["submit", "--run-name", "unit_selector_confirmation", "--phase", "remaining"]), \
                patch.object(submit, "inventory", return_value=records), patch.object(submit.Path, "read_text", return_value=json.dumps(sealed)), \
                patch.dict(sys.modules, {"scheduler": runtime}), patch.object(submit, "execute_bulk") as bulk:
            submit.main()
            tasks = bulk.call_args.args[0]
            self.assertEqual(len(tasks), 16)
            self.assertTrue(all(t["allow_duplicate"] for t in tasks))
            bulk.reset_mock()
            records[-1]["exit_code"] = 1
            with self.assertRaises(SystemExit): submit.main()
            bulk.assert_not_called()
            records[-1]["exit_code"] = 0
            records.append({**tasks[0], "id": "already_queued", "status": "queued"})
            with self.assertRaises(SystemExit): submit.main()
            bulk.assert_not_called()


if __name__ == "__main__": unittest.main()
