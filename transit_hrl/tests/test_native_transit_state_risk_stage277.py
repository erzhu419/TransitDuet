import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_state_risk_stage277 as spec
from scripts import submit_native_transit_state_risk_stage277_scheduleurm as submit

learning = spec.learning


class StateRiskTest(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(277)
        z = np.concatenate((np.ones((12, 6, 1)), rng.normal(size=(12, 6, 8))), axis=-1)
        z[:, 0, 1:] = 0.
        observed = rng.normal(size=(12, 8))
        prior = rng.normal(size=(12, 6))*.01;prior[:, 0] = 0.
        target = prior+.01*z[..., 1]*observed[:, None, 0]
        mask = np.ones((12, 6), bool);mask[:, 0] = False;mask[0, 4:] = False
        return z, observed, target, prior, mask

    def test_context_uses_only_observed_fields_not_group_ids_or_labels(self):
        row = {"time_s": 7200., "native_state_v2": [0.]*13, "direction": True,
            "context": {"queues": {"True": 10., "False": 20.}, "onboard": {"True": 30.}},
            "occupancy_context": {"active_buses": [], "queue_by_station": []}}
        a = learning.observed_context([row])
        b = learning.observed_context([{**row, "root": 999, "scenario": "new", "probe_index": 37, "labels": "unused"}])
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, [[2., .2, .4, .3, 0., 0., 0., 0.]])

    def test_same_capacity_and_training_only_normalization(self):
        z, c, y, p, e = self.fixture();ix = np.arange(8)
        model = learning.fit(z[ix], c[ix], y[ix], p[ix], e[ix], .01)
        for m in model["maps"].values():
            self.assertEqual(m["weights"].shape, (73, 1))
        changed = c.copy();changed[8:] += 1e8
        repeat = learning.fit(z[ix], changed[ix], y[ix], p[ix], e[ix], .01)
        for kind in learning.MAPS:
            for f in ("center", "scale", "weights"):
                np.testing.assert_array_equal(model["maps"][kind][f], repeat["maps"][kind][f])
        before = learning.predict(model, z, c, p, e, np.zeros(12, int), .002)[0]
        after = learning.predict(model, z, changed, p, e, np.zeros(12, int), .002)[0]
        np.testing.assert_array_equal(before["action_context"], after["action_context"])
        self.assertGreater(np.max(np.abs(before["observed_context"]-after["observed_context"])), 1e-6)

    def test_normalization_weights_and_ineligible_context(self):
        z, c, _, _, e = self.fixture()
        a = learning.context(z, c, e, "action_context")
        changed = z.copy();changed[0, 4:, 1:] = 1e8
        np.testing.assert_array_equal(a, learning.context(changed, c, e, "action_context"))
        np.testing.assert_array_equal(c, learning.context(changed, c, e, "observed_context"))

    def test_pointwise_clip_and_original_anchor(self):
        z, c, y, p, e = self.fixture();model = learning.fit(z, c, y, p, e, .01)
        for radius in (0., .001):
            costs, clips = learning.predict(model, z*1e5, c, p, e, np.zeros(12, int), radius)
            for kind in learning.MAPS:
                np.testing.assert_array_equal(costs[kind][:, 0], 0.)
                self.assertLessEqual(np.max(np.abs(costs[kind+"_clip"]-p)), radius+1e-12)
                learning.base.metrics(costs[kind+"_clip"], p, y[:, None], e, radius, clips[kind+"_clip"])

    def test_one_cached_cell_no_prior_refit_and_freeze_before_evaluation(self):
        rng = np.random.default_rng(277)
        rows = [{"root": 397, "scene_seed": i//6, "probe_index": i%6} for i in range(60)]
        held = [{**r, "scene_seed": r["scene_seed"]+10} for r in rows]
        x = rng.normal(size=(60, 6, 198));x[:, 0] = 0.
        data = {"x": x, "originals": np.zeros(60, int), "raw_logits": -np.ones((60, 6))}
        mask = np.ones((60, 6), bool);mask[:, 0] = False
        values = rng.normal(size=(60, 32, 6, 2))*.02;values[:, :, 0] = 0.
        old, queries, errors, saved = [], [], [], {}
        folds = [(None, list(range(60)), [("train", list(range(60))), ("held", list(range(60)))])]
        folds += [(list(g), list(ix), [("retained", list(ix)), ("excluded", list(val))]) for g, ix, val in spec.cache.learning.joint.binding.policy.prefix_folds(rows)]
        linear = {"center": [0.]*8, "scale": [1.]*8, "weights": np.zeros((9, 3)).tolist()}
        for bank in (0, 1):
            for group, ids, scopes in folds:
                i = len(old)
                old.append({"bank": bank, "removed": group, "penalty": .01, "fit_indices": ids,
                    "training_draws": list(range(16*bank, 16*(bank+1))), "radius": .002, "maps": {"linear": linear}})
                for scope, ii in scopes:
                    queries.append({"index": i, "scope": scope, "label": "held" if scope == "held" else "train", "ids": ii, "eligible": mask[ii].tolist()})
                    for a in (*learning.ARMS[:3], "raw_bounded", "raw_linear"): saved[f"{i}_{scope}_{a}"] = np.zeros((len(ii), 6))
                    saved[f"{i}_{scope}_raw_bounded"][:, 1:] = .004
                    saved[f"{i}_{scope}_raw_bounded_clip"][:, 1:] = .002
                    for future, b in (("same", bank), ("opposite", 1-bank)):
                        stats = {a: learning.base.metrics(saved[f"{i}_{scope}_{a}"], np.zeros((len(ii), 6)), values[ii, b*16:(b+1)*16, :, 1], mask[ii], .002) for a in learning.ARMS[:3]}
                        errors.append({"bank": bank, "removed": group, "penalty": .01, "scope": scope+"_"+future, "metrics": stats})
        class Archive(dict):
            def __enter__(self): return self
            def __exit__(self, *args): return False
        files = {"models.json": old, "queries.json": queries, "errors.json": errors}
        frozen, training_calls = [], []
        def draws(rs, panels, ds):
            if not frozen:
                m = old[len(training_calls)]
                self.assertEqual(ds, m["training_draws"])
                self.assertEqual(rs, [rows[i] for i in m["fit_indices"]])
                training_calls.append(True)
            return values[[r["scene_seed"]%10*6+r["probe_index"] for r in rs]][:, ds]
        with tempfile.TemporaryDirectory() as tmp, patch.object(spec.Path, "read_text", autospec=True, side_effect=lambda p: json.dumps(files[p.name])), \
                patch.object(spec.np, "load", side_effect=[Archive(basis=np.eye(198)[:, :8]), Archive(saved), Archive({a: np.zeros((22, 60, 6)) for a in (*learning.ARMS[:3], "raw_bounded", "raw_linear")})]), \
                patch.object(spec, "read_cell", return_value={"reference": {}}), patch.object(spec, "write_json"), \
                patch.object(spec.np, "savez_compressed", side_effect=lambda *a, **k: frozen.append(True)), \
                patch.object(spec.cache.cached.training.cohort, "split_rows", return_value=(rows, held)), \
                patch.object(spec.cache.learning.base, "causal_data", return_value=data), \
                patch.object(learning, "observed_context", return_value=rng.normal(size=(60, 8))), \
                patch.object(spec.cache.learning.base, "training_draws", side_effect=draws), \
                patch.object(spec.cache.learning, "fit_inner", side_effect=AssertionError("Prior refit")), \
                patch.object(spec.cache.learning, "select", side_effect=AssertionError("Selection")):
            result = spec.fit_cell(spec.Path(tmp), spec.KEYS[0], {spec.KEYS[0][0]: []}, {})
        self.assertEqual(result["head_fits"], 44)
        self.assertEqual(len(training_calls), 22)
        self.assertEqual(len(result["records"]), 128)
        self.assertEqual(result["prediction_error"], 0.)
        self.assertLess(result["metric_error"], 1e-12)
        for r in result["records"]:
            if not r["scope"].startswith("matched_held"):
                stats = r["metrics"]["raw_bounded_clip"]
                self.assertEqual(stats["clipped_actions"], stats["action_exposures"])

    def test_scheduler_cpu_no_pin_small_pull_and_failed_parent_blocks(self):
        task = submit.task_specification("unit_state")
        self.assertEqual(task["cpu"], 1);self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertTrue(task["result_dir"].endswith("/report"))
        parent = {"protocol": submit.gate.EXPERIMENT_PROTOCOL, "contract": submit.gate.contract(), "coverage_qualified": True}
        receipt = {"scheduler": {"status_counts": {"done": 1}, "exit_code_counts": {"0": 1}}}
        with patch("sys.argv", ["submit", "--run-name", "unit_state"]), patch.object(submit, "inventory", return_value=[]), patch.object(submit.Path, "exists", return_value=False), \
                patch.object(submit.Path, "read_text", autospec=True, side_effect=lambda p: json.dumps(receipt if p.name == "completion_receipt.json" else parent)), \
                patch.object(submit.subprocess, "check_output", return_value="revision"), patch.object(submit, "write_json"), patch.object(submit, "execute_bulk") as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]), 1)
            bulk.reset_mock();receipt["scheduler"]["exit_code_counts"] = {"1": 1}
            with self.assertRaises(SystemExit): submit.main()
            bulk.assert_not_called()


if __name__ == "__main__": unittest.main()
