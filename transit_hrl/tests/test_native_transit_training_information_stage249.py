import copy
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_training_information as learning
from scripts import run_native_transit_training_information_stage249 as spec
from scripts import run_native_transit_frozen_confirmation_stage248 as confirmation
from scripts import submit_native_transit_training_information_stage249_scheduleurm as submit
from tests import test_native_transit_balanced_critic_stage247 as fixtures


class TrainingInformationTest(unittest.TestCase):
    def test_nested_training_banks_exclude_evaluation_and_keep_counts(self):
        for bank in (0, 1):
            for arm, inner, full in (("matched", 2, 4), ("expanded", 8, 16)):
                train, evaluation = learning.draw_assignment(bank, arm)
                self.assertEqual((len(train), len(evaluation)), (full, 16))
                self.assertFalse(set(train) & set(evaluation))
                halves = [learning.draw_assignment(bank, arm, h)[0] for h in (0, 1)]
                self.assertEqual([len(h) for h in halves], [inner, inner])
                self.assertEqual(train, halves[0]+halves[1])
            for half in (None, 0, 1):
                self.assertTrue(set(learning.draw_assignment(bank, "matched", half)[0]) <=
                    set(learning.draw_assignment(bank, "expanded", half)[0]))
        self.assertEqual(set(learning.draw_assignment(0, "expanded")[0]), set(learning.draw_assignment(1, "expanded")[1]))

    def test_seed_disjointness_and_chunk_budget_without_preflight_recollection(self):
        keys = [(*st, d) for st in spec.coverage.states() for d in spec.DRAWS]
        actual = spec.preflight_keys()+[k for chunk in spec.chunk_keys() for k in spec.chunk_draws(chunk)]
        self.assertEqual(len(actual), 7680)
        self.assertEqual(len(set(actual)), len(actual))
        self.assertEqual(set(actual), set(keys))
        seeds = {spec.continuation_seed(*k) for k in keys}
        old = {confirmation.continuation_seed(*st, d) for st in spec.coverage.states() for d in confirmation.DRAWS}
        old.update(spec.old_future.continuation_seed(*st, d) for st in spec.coverage.states() for d in range(8))
        old.update(spec.coverage.continuation_seed(r, s, i, p, d) for r in spec.ROOTS for s in spec.SCENARIOS
            for i in range(10) for p in spec.coverage.PROBES for d in range(8))
        self.assertEqual(len(seeds), 7680)
        self.assertFalse(seeds & old)
        self.assertEqual(len(spec.chunk_keys()), 1920)
        self.assertEqual(spec.contract()["budget"]["native_episodes"], 8*len(keys))

    def test_fitting_uses_only_assigned_futures_and_preserves_capacity(self):
        f = fixtures.BalancedCriticTest()
        f.setUp()
        cached, old_fit = f.cached, f.fit
        values = np.tile(f.draws, (1, 8, 1, 1))
        train, evaluation = learning.draw_assignment(0, "expanded")
        before = copy.deepcopy(cached)
        with patch.object(learning.previous.joint, "fit_reference", side_effect=AssertionError("Raw refit")):
            models, fit = learning.previous.fit_paths(f.rows, f.data, values[:, train], cached, old_fit)
            spec.previous.qualify_fit(fit)
            values[:, evaluation] += 999.
            other, other_fit = learning.previous.fit_paths(f.rows, f.data, values[:, train], cached, old_fit)
        self.assertEqual((models, fit), (other, other_fit))
        self.assertEqual(cached, before)
        for level in learning.previous.strength.lowrank.LEVELS:
            a, b = np.asarray(models["state_gls"]["weights"][level]), np.asarray(cached["all_mse"]["weights"][level])
            np.testing.assert_allclose(a[0]-a[1], b[0]-b[1], atol=1e-12)

    def test_unchanged_selector_and_diagnostic_units(self):
        f = fixtures.BalancedCriticTest()
        f.setUp()
        models, _ = learning.previous.fit_paths(f.rows, f.data, f.draws, f.cached, f.fit)
        paths = {arm: learning.previous.base.candidates(f.data, models["state_gls"]) for arm in learning.ARMS}
        curves = {arm: {c: learning.previous.base.curve(f.data, f.draws, p) for c, p in ps.items()} for arm, ps in paths.items()}
        blocks = [{"held_group": [r, 0], "half": h, "scales": [.1, .1], "curves": curves} for r in (397, 401) for h in (0, 1)]
        selected, _ = learning.select(blocks)
        self.assertEqual(selected["old"], selected["matched"])
        self.assertEqual(selected["matched"], selected["expanded"])
        predictions = learning.predict(paths, selected)
        stats = learning.information_error(f.draws, f.draws, f.data, predictions)
        self.assertTrue(all(x["nonanchor_mse"] == 0. for x in stats["target_mean_error"].values()))
        aggregate = learning.aggregate([learning.previous.metrics(f.data, f.draws, predictions)])
        self.assertEqual(set(aggregate["all"]["variants"]), set(learning.VARIANTS))
        for x in aggregate["all"]["variants"].values(): self.assertAlmostEqual(x["delta_cost"], x["delta_service"]+x["delta_risk"], places=12)

    def test_full_native_qualification_uses_real_training_seed(self):
        key = (*spec.coverage.states()[0], 31)
        labels = [{"candidate": i, **dict.fromkeys(spec.coverage.expected.LABEL_FIELDS, 0.)} for i in range(6)]
        r = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
            "root": key[0], "scenario": key[1], "replica": key[2], "probe_index": key[3], "draw": key[4],
            "scene_seed": spec.source.scene_seed(*key[:3]), "continuation_seed": spec.continuation_seed(*key),
            "original_choice": 0, "labels": labels, "existing_labels": copy.deepcopy(labels),
            "original_source_reproductions": 1, "exact_resampled_identities": 1, "prefix_state_rng_matches": 7,
            "frozen_weight_qualified_episodes": 8, "window_ledger_qualified_episodes": 8,
            "continuation_qualified": True, "native_clock_ticks": 8*61380, "training_updates": 0}
        spec.qualify_draw(r)
        r["continuation_seed"] = confirmation.continuation_seed(*key)
        with self.assertRaisesRegex(ValueError, "training seed"): spec.qualify_draw(r)
        r["continuation_seed"] = spec.continuation_seed(*key)
        r["native_clock_ticks"] -= 1
        with self.assertRaisesRegex(ValueError, "native budget"): spec.qualify_draw(r)

    def test_scheduler_is_unbound_cpu_and_never_syncs_raw_panels(self):
        for operation, key in (("preflight", spec.preflight_keys()[0]), ("collect", spec.chunk_keys()[0]),
                ("fit", spec.previous.PREFLIGHT), ("prepare", None)):
            task = submit.task_specification("unit_info", operation=operation, key=key)
            self.assertEqual(task["cpu"], 1)
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertNotIn("_raw", task["result_dir"])
            self.assertFalse(any("/results/" in p for p in task["stage_input_paths"]))
        self.assertEqual(spec.contract(), json.loads(json.dumps(spec.contract())))

    def test_bulk_fast_path_cannot_bypass_phase_or_resubmit_remaining(self):
        registered = {"contract": spec.contract(), "coverage_qualified": True}
        previous = [*[submit.task_specification("unit_info", operation="preflight", key=k) for k in spec.preflight_keys()],
            submit.task_specification("unit_info", operation="preflight_report")]
        records = [{**t, "id": f"t{i}", "status": "done", "exit_code": 0} for i, t in enumerate(previous)]
        runtime = SimpleNamespace(load_state=lambda: {"tasks": records})
        with patch.object(sys, "argv", ["submit", "--run-name", "unit_info", "--phase", "remaining"]), \
                patch.object(submit, "inventory", return_value=records), \
                patch.object(submit.Path, "read_text", return_value=json.dumps(registered)), \
                patch.dict(sys.modules, {"scheduler": runtime}), patch.object(submit, "execute_bulk") as bulk:
            submit.main()
            tasks = bulk.call_args.args[0]
            self.assertEqual(len(tasks), 1920)
            self.assertTrue(all(t["allow_duplicate"] for t in tasks))
            self.assertEqual(len({t["signature"] for t in tasks}), len(tasks))
            bulk.reset_mock()
            records[-1]["exit_code"] = 1
            with self.assertRaises(SystemExit): submit.main()
            bulk.assert_not_called()
            records[-1]["exit_code"] = 0
            records.append({**tasks[0], "id": "expanded", "status": "queued"})
            with self.assertRaises(SystemExit): submit.main()
            bulk.assert_not_called()


if __name__ == "__main__": unittest.main()
