import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_effective_risk_stage278 as spec
from scripts import submit_native_transit_effective_risk_stage278_scheduleurm as submit

learning = spec.learning


class EffectiveRiskTest(unittest.TestCase):
    def fixture(self, n=16):
        rng = np.random.default_rng(278)
        z = np.concatenate((np.ones((n, 6, 1)), rng.normal(size=(n, 6, 8))), axis=-1);z[:, 0, 1:] = 0.
        c = rng.normal(size=(n, 8));p = np.zeros((n, 6));y = rng.normal(size=(n, 6))*.01;y[:, 0] = 0.
        e = np.ones((n, 6), bool);e[:, 0] = False;e[0, 4:] = False
        return z, c, y, p, e

    def cached(self, z, c, y, p, e):
        m = learning.previous.fit(z, c, y, p, e, .01)
        linear = learning.previous.base.normalization(z[..., 1:], e)
        return m, linear

    def test_spectral_df_equals_weighted_hat_trace(self):
        z, _, _, _, e = self.fixture()
        stats = learning.previous.base.normalization(z[..., 1:], e)
        design = learning.previous.base.transform(z[..., 1:], stats)
        x = design[e];w = learning.previous.base.base.state_weights(e)[e];w /= w.sum()
        gram = x.T@(w[:, None]*x);regularizer = np.eye(9)*.01;regularizer[0, 0] = 0.
        actual = np.trace(np.linalg.solve(gram+regularizer, gram))
        self.assertAlmostEqual(learning.effective_df(learning.spectrum(design, e), .01), actual)

    def test_feature_only_match_reuses_maps_and_is_independent_of_labels(self):
        z, c, y, p, e = self.fixture();old, linear = self.cached(z, c, y, p, e)
        one = learning.fit(old, linear, z, c, y, p, e)
        two = learning.fit(old, linear, z, c, y*100., p, e)
        self.assertEqual(one["complexity"], two["complexity"])
        for kind, d in one["complexity"].items():
            self.assertAlmostEqual(d["target_df"], d["matched_df"], places=8)
            self.assertGreater(d["old_df"], d["target_df"])
            self.assertGreater(d["penalty"], .01)
            self.assertEqual(one["maps"][kind]["weights"].shape, (73, 1))
            for f in ("center", "scale", "context_stats"):
                self.assertIs(one["maps"][kind][f], old["maps"][kind][f])

    def test_bounds_and_original_anchor_survive_query_shift(self):
        z, c, y, p, e = self.fixture();old, linear = self.cached(z, c, y, p, e)
        m = learning.fit(old, linear, z, c, y, p, e)
        for radius in (.001, 0.):
            costs, clips = learning.predict(m, z*1e5, c*1e5, p, e, np.zeros(len(z), int), radius)
            for kind in learning.MAPS:
                a = kind+"_df_clip"
                self.assertLessEqual(np.max(np.abs(costs[a])), radius+1e-12)
                np.testing.assert_array_equal(costs[a][:, 0], 0.)
                learning.previous.base.metrics(costs[a], p, y[:, None], e, radius, clips[a])

    def test_complete_cell_and_group_replay_with_freeze_before_evaluation(self):
        z, c, _, p, e = self.fixture(60);e[:] = True;e[:, 0] = False
        rows = [{"root": 397, "scene_seed": i//6, "probe_index": i%6} for i in range(60)]
        held = [{**r, "scene_seed": r["scene_seed"]+10} for r in rows]
        x = np.zeros((60, 6, 198));x[..., :8] = z[..., 1:]
        data = {"x": x, "originals": np.zeros(60, int), "raw_logits": -np.ones((60, 6))}
        values = np.random.default_rng(279).normal(size=(60, 32, 6, 2))*.02;values[:, :, 0] = 0.
        folds = [(None, list(range(60)), [("train", list(range(60)), "train"), ("held", list(range(60)), "held")])]
        folds += [(list(g), list(ix), [("retained", list(ix), "train"), ("excluded", list(val), "train"), ("matched_held", list(range(60)), "held")]) for g, ix, val in spec.cache.learning.joint.binding.policy.prefix_folds(rows)]
        old, linears, queries, errors, saved = [], [], [], [], {}
        for bank in (0, 1):
            ds = list(range(bank*16, (bank+1)*16))
            for group, ids, scopes in folds:
                i = len(old);target = values[ids][:, ds][..., 1].mean(axis=1)
                m, linear = self.cached(z[ids], c[ids], target, p[ids], e[ids]);linear["weights"] = np.zeros((9, 3))
                meta = {"index": i, "bank": bank, "removed": group, "training_draws": ds, "fit_indices": ids, "radius": .002}
                old.append({**meta, **m});linears.append({**meta, "penalty": .01, "maps": {"linear": linear}})
                for scope, ii, label in scopes:
                    q = {"index": i, "bank": bank, "removed": group, "label": label, "ids": ii, "scope": scope, "radius": .002, "eligible": e[ii].tolist()};queries.append(q)
                    costs = {a: np.zeros((len(ii), 6)) for a in learning.previous.ARMS[:3]}
                    new, clips = learning.previous.predict(m, z[ii], c[ii], p[ii], e[ii], np.zeros(len(ii), int), .002);costs.update(new)
                    saved.update({f"{i}_{scope}_{a}": v for a, v in costs.items()})
                    for future, b in (("same", bank), ("opposite", 1-bank)):
                        stats = {a: learning.previous.base.metrics(v, p[ii], values[ii, b*16:(b+1)*16, :, 1], e[ii], .002, clips.get(a)) for a, v in costs.items()}
                        errors.append({"bank": bank, "scope": scope+"_"+future, "metrics": stats})
        class Archive(dict):
            def __enter__(self): return self
            def __exit__(self, *args): return False
        def convert(v):
            if isinstance(v, np.ndarray): return v.tolist()
            raise TypeError(type(v))
        def read(path):
            v = linears if spec.source.SOURCE_RUN in str(path) else {"models.json": old, "queries.json": queries, "errors.json": errors}[path.name]
            return json.dumps(v, default=convert)
        frozen, training = [], []
        def draws(rs, panels, ds):
            if not frozen:
                m = old[len(training)];self.assertEqual(ds, m["training_draws"])
                self.assertEqual(rs, [rows[j] for j in m["fit_indices"]]);training.append(True)
            return values[[r["scene_seed"]%10*6+r["probe_index"] for r in rs]][:, ds]
        with tempfile.TemporaryDirectory() as tmp, patch.object(spec.Path, "read_text", autospec=True, side_effect=read), \
                patch.object(spec.np, "load", side_effect=[Archive(basis=np.eye(198)[:, :8]), Archive(saved)]), \
                patch.object(spec.source, "read_cell", return_value={"reference": {}}), patch.object(spec, "write_json"), \
                patch.object(spec.np, "savez_compressed", side_effect=lambda *a, **kw: frozen.append(True)), \
                patch.object(spec.cache.cached.training.cohort, "split_rows", return_value=(rows, held)), \
                patch.object(spec.cache.learning.base, "causal_data", return_value=data), patch.object(learning.previous, "observed_context", return_value=c), \
                patch.object(spec.cache.learning.base, "training_draws", side_effect=draws), \
                patch.object(spec.cache.learning, "fit_inner", side_effect=AssertionError("Prior refit")), \
                patch.object(spec.cache.learning, "select", side_effect=AssertionError("Policy selection")):
            result = spec.fit_cell(spec.Path(tmp), spec.KEYS[0], {spec.KEYS[0][0]: []}, {})
        self.assertEqual(len(training), 22);self.assertEqual(result["head_fits"], 44)
        self.assertEqual(result["prediction_error"], 0.);self.assertLess(result["metric_error"], 1e-12)
        cells = [{**result, "key": list(k)} for k in spec.KEYS]
        before = spec.source.report(cells)
        report = spec.report(cells, before)
        self.assertTrue(report["coverage_qualified"])
        self.assertEqual(report["reproduction"]["group_metric_error"], 0.)

    def test_scheduler_unpinned_and_parent_exit_required(self):
        t = submit.task_specification("unit_edf")
        self.assertEqual(t["cpu"], 1);self.assertIsNone(t["require_node"])
        self.assertEqual(t["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertTrue(t["result_dir"].endswith("/report"))
        parent = {"protocol": spec.source.EXPERIMENT_PROTOCOL, "contract": spec.source.contract(), "coverage_qualified": True, "fits": {"residual_head_fits": 1408}}
        receipt = {"scheduler": {"status_counts": {"done": 1}, "exit_code_counts": {"0": 1}}}
        with patch("sys.argv", ["submit", "--run-name", "unit_edf"]), patch.object(submit, "inventory", return_value=[]), patch.object(submit.Path, "exists", return_value=False), \
                patch.object(submit.Path, "read_text", autospec=True, side_effect=lambda p: json.dumps(receipt if p.name == "completion_receipt.json" else parent)), \
                patch.object(submit.subprocess, "check_output", return_value="revision"), patch.object(submit, "write_json"), patch.object(submit, "execute_bulk") as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]), 1)
            bulk.reset_mock();receipt["scheduler"]["exit_code_counts"] = {"1": 1}
            with self.assertRaises(SystemExit): submit.main()
            bulk.assert_not_called()


if __name__ == "__main__": unittest.main()
