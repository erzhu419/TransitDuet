import copy
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_service_transfer_stage282 as spec
from scripts import submit_native_transit_service_transfer_stage282_scheduleurm as submit

learning = spec.learning


class ServiceTransferTest(unittest.TestCase):
    def fixture(self, n=20):
        rng = np.random.default_rng(282)
        x = rng.normal(size=(n, 6, 198))
        service = np.tile([0., .02, .01, .05, .05, .05], (n, 1))
        risk = np.tile([0., 0., -.005, 0., 0., 0.], (n, 1))
        logits = -np.ones((n, 6))
        logits[:, 5] = 1.
        data = {"x": x, "originals": np.zeros(n, int), "components": np.stack((service, risk), axis=-1),
            "heads": np.stack((service+risk, np.zeros_like(service)), axis=-1),
            "raw_logits": logits, "raw_choices": np.zeros(n, int)}
        risks = {a: risk.copy() for a in learning.probe.policy.ARMS}
        risks["raw_linear_clip"][:, 2] = -.03
        risks["observed_context_df_clip"][:, 1] = -.04
        values = np.zeros((n, 32, 6, 4))
        values[:, :, 1, 1] = .02
        values[:, :, 2, 1] = -.01
        values[:, :, 1, 2] = .004
        values[:, :, 2, 2] = -.002
        values[:, :, 1, 3] = .03
        values[:, :, 2, 3] = -.04
        values[..., 0] = values[..., 1]+values[..., 2]
        z = np.concatenate((np.ones((n, 6, 1)), (x-x[:, :1])[..., :8]), axis=-1)
        observed = rng.normal(size=(n, 8))
        eligible = (logits <= 0)&(np.arange(6)[None] != 0)
        prior = learning.probe.policy.joint.relative(data["heads"], data["originals"])[..., 0]-risks["raw_prior"]
        return data, risks, values, z, observed, eligible, prior

    def geometry(self, z, observed, eligible):
        base = learning.base
        linear = {**base.normalization(z[..., 1:], eligible), "weights": np.zeros((9, 1))}
        df = spec.geometry.learning.effective_df(spec.geometry.learning.spectrum(base.transform(z[..., 1:], linear), eligible), .01)
        maps, complexity = {}, {}
        for k in learning.state.MAPS:
            c = learning.state.context(z, observed, eligible, k)
            context = base.normalization(np.broadcast_to(c[:, None], (*z.shape[:2], 8)), eligible)
            lift = learning.state.lift(z, c, context)
            m = {"context_stats": context, **base.normalization(lift, eligible), "weights": np.zeros((73, 1))}
            design = base.transform(lift, m)
            penalty = spec.geometry.learning.matched_penalty(spec.geometry.learning.spectrum(design, eligible), df)
            maps[k] = m
            complexity[k] = {"penalty": penalty, "target_df": df}
        return {"maps": maps, "complexity": complexity}, linear

    def test_training_only_radius_weights_and_frozen_geometry(self):
        data, risks, values, z, observed, eligible, prior = self.fixture()
        cached, linear = self.geometry(z, observed, eligible)
        before = copy.deepcopy(cached)
        target = prior+.003*z[..., 1]
        model = learning.fit(cached, linear, z, observed, target, prior, eligible)
        weights = learning.base.base.state_weights(eligible)[eligible]
        self.assertAlmostEqual(model["radius"], np.sqrt(np.average((target-prior)[eligible]**2, weights=weights)))
        design = learning.base.transform(z[..., 1:], linear)[eligible]
        w = weights/weights.sum()
        reg = np.eye(9)*.01
        reg[0, 0] = 0.
        expected = np.linalg.solve(design.T@(w[:, None]*design)+reg, design.T@(w*(target-prior)[eligible]))
        np.testing.assert_allclose(model["maps"]["linear"]["weights"][:, 0], expected, atol=1e-14)
        for k in learning.state.MAPS:
            self.assertEqual(model["maps"][k+"_df"]["penalty"], cached["complexity"][k]["penalty"])
            for f in ("center", "scale", "weights"):
                np.testing.assert_array_equal(cached["maps"][k][f], before["maps"][k][f])
        self.assertEqual(model["head_fits"], 3)
        self.assertLess(model["normal_error"], 1e-12)

    def test_zero_update_exact_controls_query_clip_and_benchmark_isolation(self):
        data, risks, values, z, observed, eligible, prior = self.fixture()
        cached, linear = self.geometry(z, observed, eligible)
        zero = learning.fit(cached, linear, z, observed, prior, prior, eligible)
        services, _ = learning.predict(zero, z, observed, prior, eligible, data["originals"])
        for v in services.values():
            np.testing.assert_array_equal(v, prior)
        services["mc"] = prior.copy()
        choices = learning.decisions(data, risks, services)
        frozen, _ = learning.probe.policy.decisions(data, risks)
        for s in learning.SERVICES:
            for r in learning.CONTROLS:
                np.testing.assert_array_equal(choices[r if s == "raw" else f"{s}__{r}"], frozen[r])
        model = learning.fit(cached, linear, z, observed, prior+.01*z[..., 1], prior, eligible)
        predictions, _ = learning.predict(model, z*100, observed*100, prior, eligible, data["originals"])
        for k in learning.MAPS:
            self.assertLessEqual(np.max(np.abs(predictions[k+"_clip"]-prior)), model["radius"]+1e-15)
            self.assertTrue(np.all(predictions[k][:, 0] == 0))
        predictions["mc"] = prior.copy()
        a = learning.decisions(data, risks, predictions)
        predictions["mc"] = -np.ones_like(prior)
        b = learning.decisions(data, risks, predictions)
        for name in a:
            if not name.startswith("mc__"):
                np.testing.assert_array_equal(a[name], b[name])

    def test_service_metrics_ignore_fleet_and_use_state_balanced_population(self):
        data, risks, values, z, observed, eligible, prior = self.fixture(2)
        eligible[1] = False
        stat = learning.metrics({"raw": prior}, {}, prior, values[:, :16], data["originals"], eligible, .1)["raw"]
        self.assertEqual(stat["state_exposures"], 1)
        truth = values[:, :16, :, 2].mean(axis=1)
        expected = np.mean((prior[0, eligible[0]]-truth[0, eligible[0]])**2)
        self.assertAlmostEqual(stat["squared_error_sum"], expected)
        values[..., 1] += 100.
        values[..., 0] = values[..., 1]+values[..., 2]
        replay = learning.metrics({"raw": prior}, {}, prior, values[:, :16], data["originals"], eligible, .1)["raw"]
        self.assertEqual(replay, stat)

    def test_full_cell_sparse_model_indices_fit_isolation_and_group_replay(self):
        data, risks, values, z, observed, eligible, prior = self.fixture(60)
        rows = [{"root": 397, "scene_seed": j//6, "probe_index": j%6} for j in range(60)]
        held = [{**r, "scene_seed": r["scene_seed"]+10} for r in rows]
        folds = [(None, list(range(60)), [("train", list(range(60)), "train"), ("held", list(range(60)), "held")])]
        folds += [(list(g), list(ix), [("retained", list(ix), "train"), ("excluded", list(val), "train"), ("matched_held", list(range(60)), "held")])
            for g, ix, val in spec.cache.learning.joint.binding.policy.prefix_folds(rows)]
        models, queries, saved, benchmarks, previous = [], [], {}, {}, []
        linear_models = [None]*66
        for bank in (0, 1):
            draws, _ = spec.cache.sampling.learning.draw_assignment(bank, "expanded")
            for group, ix, scopes in folds:
                i = 3*len(models)+1
                geom, linear = self.geometry(z[ix], observed[ix], eligible[ix])
                context = {"index": i, "bank": bank, "removed": group, "fit_indices": ix, "training_draws": draws}
                models.append({**context, **geom})
                linear_models[i] = {**context, "penalty": .01, "maps": {"linear": linear}}
                for scope, ids, label in scopes:
                    d = {k: v[ids] for k, v in data.items()}
                    r = {k: v[ids] for k, v in risks.items()}
                    choices, _ = learning.probe.policy.decisions(d, r)
                    q = {"index": i, "bank": bank, "removed": group, "scope": scope, "label": label,
                        "ids": ids, "eligible": eligible[ids].tolist(), "choices": {a: v.tolist() for a, v in choices.items()}}
                    qi = len(queries)
                    queries.append(q)
                    saved.update({f"{i}_{scope}_{a}": v for a, v in r.items()})
                    chosen, parts, _ = learning.probe.propose(d, r, values[ids][:, draws])
                    benchmarks[qi] = {**q, "choices": {a: v.tolist() for a, v in chosen.items()},
                        "components": {a: v.tolist() for a, v in parts.items()}}
                    for future, b in (("same", bank), ("opposite", 1-bank)):
                        v = values[ids, b*16:(b+1)*16]
                        previous.append({"bank": bank, "scope": scope+"_"+future,
                            "outcomes": learning.probe.policy.outcomes(v, chosen), "component_errors": learning.probe.component_errors(d, parts, v)})
        class Archive(dict):
            def __enter__(self): return self
            def __exit__(self, *args): return False
        frozen, fit_accesses, evaluation_accesses = [], [], []
        def read(path):
            if path.name == "models.json":
                return json.dumps(models if spec.RISK_RUN in path.parts else linear_models, default=lambda a: a.tolist())
            if path.name == "actions.json":
                return json.dumps(queries)
            if path.name.startswith("proposal_"):
                return json.dumps(benchmarks[int(path.stem.split("_")[1])])
            self.assertEqual(path.name, "outcomes.json")
            self.assertTrue(frozen)
            return json.dumps(previous)
        def write(path, payload):
            if path.name == "predictions.json":
                frozen.append(payload)
        def draw_values(rs, panels, ds):
            if not frozen:
                m = models[len(fit_accesses)]
                self.assertEqual(rs, [rows[j] for j in m["fit_indices"]])
                self.assertEqual(ds, m["training_draws"])
                fit_accesses.append(True)
            else:
                evaluation_accesses.append(True)
            return values[[r["scene_seed"]%10*6+r["probe_index"] for r in rs]][:, ds]
        def load(path):
            return Archive({"basis": np.eye(198, 8)} if spec.geometry.source.BASIS_RUN in path.parts else saved)
        with tempfile.TemporaryDirectory() as tmp, patch.object(spec.Path, "read_text", autospec=True, side_effect=read), \
                patch.object(spec.np, "load", side_effect=load), patch.object(spec.geometry.source, "read_cell", return_value={"reference": {}}), \
                patch.object(spec, "write_json", side_effect=write), patch.object(spec.cache.cached.training.cohort, "split_rows", return_value=(rows, held)), \
                patch.object(spec.cache.learning.base, "causal_data", return_value=data), patch.object(learning.state, "observed_context", return_value=observed), \
                patch.object(spec.cache.learning.base, "training_draws", side_effect=draw_values), \
                patch.object(spec.geometry.learning, "fit", side_effect=AssertionError("Risk refit")), \
                patch.object(spec.cache.learning, "fit_inner", side_effect=AssertionError("Prior refit")):
            result = spec.fit_cell(spec.Path(tmp), spec.KEYS[0], {spec.KEYS[0][0]: []}, {})
        self.assertEqual(len(fit_accesses), 22)
        self.assertEqual(len(evaluation_accesses), 128)
        self.assertEqual(result["head_fits"], 66)
        self.assertEqual(result["model_state_exposures"], 2640)
        self.assertLess(result["replay_error"], 1e-12)
        before = {"groups": {}}
        for g in ("mixed_pooled", *spec.VIEWS):
            keys = [k for k in spec.KEYS if k[0].startswith("mixed")] if g == "mixed_pooled" else [k for k in spec.KEYS if k[0] == g]
            before["groups"][g] = {}
            for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
                before["groups"][g][label] = {}
                for scope in spec.SCOPES:
                    rs = [r for _ in keys for r in previous if r["scope"] == scope and (bank is None or r["bank"] == bank)]
                    before["groups"][g][label][scope] = {"outcomes": {h: {a: learning.probe.policy.summarize([r["outcomes"][h][a] for r in rs])
                        for a in (*learning.CONTROLS, *learning.BENCHMARKS.values())} for h in learning.probe.policy.DRAW_GROUPS},
                        "component_errors": learning.probe.summarize_errors([r["component_errors"] for r in rs])}
        report = spec.report([{**result, "key": list(k)} for k in spec.KEYS], before)
        self.assertTrue(report["coverage_qualified"])
        self.assertLess(report["reproduction"]["group_replay_error"], 1e-12)
        self.assertEqual(set(report["groups"]["mixed_pooled"]["pooled"]["held_opposite"]["outcomes"]), {"all16", "first8", "last8"})
        json.dumps(report, allow_nan=False)

    def test_scheduler_parent_gate_and_no_node_pin(self):
        task = submit.task_specification("unit_service")
        self.assertEqual(task["cpu"], 1)
        self.assertEqual(task["ram_mb"], 3072)
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        parent = {"protocol": spec.source.EXPERIMENT_PROTOCOL, "contract": spec.source.contract(), "coverage_qualified": True}
        receipt = {"scheduler": {"status_counts": {"done": 1}, "exit_code_counts": {"0": 1}}}
        with patch("sys.argv", ["submit", "--run-name", "unit_service"]), patch.object(submit, "inventory", return_value=[]), \
                patch.object(submit.Path, "exists", return_value=False), patch.object(submit.Path, "read_text", autospec=True,
                    side_effect=lambda p: json.dumps(receipt if p.name == "completion_receipt.json" else parent)), \
                patch.object(submit.subprocess, "check_output", return_value="revision"), patch.object(submit, "write_json"), patch.object(submit, "execute_bulk") as bulk:
            submit.main()
            self.assertEqual(len(bulk.call_args.args[0]), 1)
            bulk.reset_mock()
            receipt["scheduler"]["exit_code_counts"] = {"1": 1}
            with self.assertRaises(SystemExit):
                submit.main()
            bulk.assert_not_called()


if __name__ == "__main__":
    unittest.main()
