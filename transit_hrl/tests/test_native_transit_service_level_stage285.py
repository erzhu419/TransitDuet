import copy
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_service_level_stage285 as spec
from scripts import submit_native_transit_service_level_stage285_scheduleurm as submit
import test_native_transit_service_centering_stage284 as parent_test

learning, policy = spec.learning, spec.policy


class ServiceLevelTest(unittest.TestCase):
    def fixture(self, n=60):
        data, risks, services, clips, cached, values, parts = parent_test.ServiceCenteringTest().fixture(n)
        rng = np.random.default_rng(285)
        data["x"] = rng.normal(size=(n, 6, 198))
        observed = rng.normal(size=(n, 8))
        eligible = (data["raw_logits"] <= 0)&(np.arange(6)[None] != 0)
        services, added = learning.previous.ablate(services, parts, eligible, data["originals"], cached["radius"])
        cached["maps"]["linear"].update(center=np.zeros(8), scale=np.ones(8))
        cached["maps"]["observed_context_df"]["context_stats"] = {"center": np.zeros(8), "scale": np.ones(8)}
        return data, risks, services, {**clips, **added}, cached, values, parts, observed, eligible

    def test_matched_feature_only_complexity_and_direct_level_target(self):
        data, risks, services, clips, cached, values, parts, observed, eligible = self.fixture()
        z = np.concatenate((np.ones((60, 6, 1)), data["x"][..., :8]), axis=-1)
        coords = learning.features(cached, z, observed, eligible)
        eligible[-1] = False
        residual = .001+.002*coords["observed_level"][:, 0]
        target = services["raw"]+residual[:, None]
        cached["maps"]["linear"]["weights"][0, 0] = residual[:-1].mean()
        m = learning.fit(cached, coords, target, services["raw"], eligible)
        self.assertEqual(m["head_fits"], 2)
        self.assertEqual(m["eligible_fit_states"], 59)
        self.assertLess(m["normal_error"], 1e-12)
        self.assertLess(m["df_error"], 1e-8)
        self.assertLess(m["uniform_error"], 1e-12)
        for k in learning.KINDS:
            self.assertEqual(m["maps"][k]["weights"].shape, (9, 1))
            self.assertAlmostEqual(m["maps"][k]["effective_df"], m["target_df"], places=8)
            self.assertGreaterEqual(m["maps"][k]["penalty"], .01)
        shifted = learning.fit(cached, coords, target+np.linspace(-1., 1., 6)[None]*eligible, services["raw"], eligible)
        for k in learning.KINDS:
            self.assertEqual(m["maps"][k]["penalty"], shifted["maps"][k]["penalty"])
        predicted, _ = learning.predict(m, coords, services, eligible, data["originals"])
        # The pure observed-state level is in the supplied coordinate span.
        self.assertLess(np.mean((predicted["observed_level_clip"][:-1, 1]-target[:-1, 1])**2), 1e-8)

    def test_frozen_contrasts_anchors_clip_and_level_ranking(self):
        data, risks, services, clips, cached, values, parts, observed, eligible = self.fixture(6)
        coords = {k: np.zeros((6, 8)) for k in learning.KINDS}
        model = {"radius": .1, "maps": {k: {"center": np.zeros(8), "scale": np.ones(8),
            "weights": np.r_[.003, np.zeros(8)][:, None]} for k in learning.KINDS}}
        before = {k: v.copy() for k, v in services.items()}
        output, _ = learning.predict(model, coords, services, eligible, data["originals"])
        for k in learning.KINDS:
            for m in learning.service.MAPS:
                delta = output[f"{k}_{m}_contrast_clip"]-services["raw"]
                centered = learning.previous.center(services[m]-services["raw"], eligible)[1]
                np.testing.assert_allclose(delta[eligible], (.003+centered)[eligible], atol=1e-15)
            level_delta = output[k+"_clip"]-services["raw"]
            self.assertTrue(np.allclose(level_delta[eligible], .003))
        for s in learning.NEW_SERVICES:
            np.testing.assert_array_equal(output[s][:, 0], services["raw"][:, 0])
        for s in before:
            np.testing.assert_array_equal(before[s], output[s])
        old = learning.service.decisions(data, risks, services)
        new = learning.service.decisions(data, risks, output)
        for a in learning.previous.ARMS:
            np.testing.assert_array_equal(old[a], new[a])
        model["radius"] = .001
        clipped, _ = learning.predict(model, coords, services, eligible, data["originals"])
        for s in learning.NEW_SERVICES:
            self.assertLessEqual(np.max(np.abs(clipped[s]-services["raw"])), .001+1e-15)

    def test_full_cell_assigned_only_fits_and_group_replay(self):
        data, risks, services, clips, cached, values, parts, observed, eligible = self.fixture()
        rows = [{"root": 397, "scene_seed": i//6, "probe_index": i%6} for i in range(60)]
        held = [{**r, "scene_seed": r["scene_seed"]+10} for r in rows]
        folds = [(None, np.arange(60), [("train", list(range(60)), "train"), ("held", list(range(60)), "held")])]
        folds += [(list(g), ix, [("retained", list(ix), "train"), ("excluded", list(val), "train"), ("matched_held", list(range(60)), "held")])
            for g, ix, val in spec.cache.learning.joint.binding.policy.prefix_folds(rows)]
        queries, models, previous, saved = [], [], [], {}
        for bank in (0, 1):
            for group, fit_indices, scopes in folds:
                i = 3*len(models)+1
                draws, _ = spec.cache.sampling.learning.draw_assignment(bank, "expanded")
                m = copy.deepcopy(cached)
                target = values[fit_indices][:, draws][..., 2].mean(axis=1)
                level = learning.previous.center(target-services["raw"][fit_indices], eligible[fit_indices])[0]
                m["maps"]["linear"]["weights"][0, 0] = level.mean()
                models.append({**m, "index": i, "bank": bank, "removed": group, "fit_indices": list(fit_indices), "training_draws": draws})
                for scope, ids, label in scopes:
                    d = {k: v[ids] for k, v in data.items()}
                    r = {a: v[ids] for a, v in risks.items()}
                    s = {a: v[ids] for a, v in services.items()}
                    c = {a: v[ids] for a, v in clips.items()}
                    choices = learning.service.decisions(d, r, s)
                    queries.append({"index": i, "bank": bank, "removed": group, "scope": scope, "label": label, "ids": ids,
                        "eligible": eligible[ids].tolist(), "radius": m["radius"], "services": s, "clips": c, "choices": choices})
                    saved.update({f"{i}_{scope}_{a}": v for a, v in r.items()})
                    for future, b in (("same", bank), ("opposite", 1-bank)):
                        ds, _ = spec.cache.sampling.learning.draw_assignment(b, "expanded")
                        v = values[ids][:, ds]
                        geometry, _ = learning.previous.error_geometry(v, s, d["originals"], eligible[ids])
                        previous.append({"bank": bank, "scope": scope+"_"+future, "outcomes": policy.outcomes(v, choices), "geometry": geometry,
                            "metrics": learning.service.metrics(s, c, s["raw"], v, d["originals"], eligible[ids], m["radius"])})
        class Archive(dict):
            def __enter__(self): return self
            def __exit__(self, *args): return False
        frozen, fit_calls, evaluation_calls = [], [], []
        def read(path):
            if path.name == "models.json": payload = models
            elif path.name == "predictions.json": payload = queries
            else:
                self.assertEqual(path.name, "outcomes.json")
                self.assertTrue(frozen)
                payload = previous
            return json.dumps(payload, default=lambda a: a.tolist() if hasattr(a, "tolist") else int(a))
        def write(path, payload):
            if path.name == "predictions.json": frozen.append(True)
        def draws(rs, panels, ds):
            if not frozen:
                m = models[len(fit_calls)]
                self.assertTrue(all(r["scene_seed"] < 10 for r in rs))
                self.assertEqual(list(ds), m["training_draws"])
                self.assertEqual([r["scene_seed"]*6+r["probe_index"] for r in rs], m["fit_indices"])
                fit_calls.append(True)
            else: evaluation_calls.append(True)
            return values[[r["scene_seed"]%10*6+r["probe_index"] for r in rs]][:, ds]
        def context(rs): return observed[[r["scene_seed"]%10*6+r["probe_index"] for r in rs]]
        with tempfile.TemporaryDirectory() as tmp, patch.object(spec.Path, "read_text", autospec=True, side_effect=read), \
                patch.object(spec.np, "load", side_effect=[Archive(basis=np.eye(198, 8)), Archive(saved)]), \
                patch.object(spec.transfer.geometry.source, "read_cell", return_value={"reference": {}}), patch.object(spec, "write_json", side_effect=write), \
                patch.object(spec.cache.cached.training.cohort, "split_rows", return_value=(rows, held)), \
                patch.object(spec.cache.learning.base, "causal_data", return_value=data), patch.object(learning.service.state, "observed_context", side_effect=context), \
                patch.object(spec.cache.learning.base, "training_draws", side_effect=draws), \
                patch.object(spec.transfer.learning, "fit", side_effect=AssertionError("Old service/contrast refit")), \
                patch.object(spec.cache.learning, "fit_inner", side_effect=AssertionError("Raw refit")):
            result = spec.fit_cell(spec.Path(tmp), spec.KEYS[0], {spec.KEYS[0][0]: []}, {})
        self.assertEqual(len(fit_calls), 22)
        self.assertEqual(len(evaluation_calls), 128)
        self.assertEqual(result["head_fits"], 44)
        self.assertEqual(result["model_state_exposures"], 2640)
        before = {"groups": {}}
        for g in ("mixed_pooled", *spec.VIEWS):
            keys = [k for k in spec.KEYS if k[0].startswith("mixed")] if g == "mixed_pooled" else [k for k in spec.KEYS if k[0] == g]
            before["groups"][g] = {}
            for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
                before["groups"][g][label] = {}
                for scope in spec.SCOPES:
                    rs = [r for _ in keys for r in previous if r["scope"] == scope and (bank is None or r["bank"] == bank)]
                    halves = policy.DRAW_GROUPS if g == "mixed_pooled" and scope in (*spec.PRIMARY, "matched_held_opposite") else ("all16",)
                    before["groups"][g][label][scope] = {"service_metrics": {a: {f: m[f] for f in ("MSE", "bias", "clipped_action_fraction")}
                        for a in learning.previous.SERVICES for m in [learning.base.summarize([r["metrics"][a] for r in rs], True)]},
                        "outcomes": {h: {a: policy.summarize([r["outcomes"][h][a] for r in rs]) for a in learning.previous.REPORT_ARMS} for h in halves},
                        "error_geometry": {h: {a: learning.previous.summarize([r["geometry"][h][a] for r in rs]) for a in learning.previous.SERVICES} for h in halves}}
        report = spec.report([{**result, "key": list(k)} for k in spec.KEYS], before)
        self.assertTrue(report["coverage_qualified"])
        self.assertLess(report["reproduction"]["group_replay_error"], 1e-12)
        self.assertEqual(report["fits"]["level_head_fits"], 1408)
        primary = report["groups"]["mixed_pooled"]["pooled"]["held_opposite"]
        self.assertEqual(set(primary["error_geometry"]), {"all16", "first8", "last8"})
        self.assertEqual(set(primary["service_metrics"]), set(learning.REPORT_SERVICES))
        self.assertEqual(set(primary["switches"]), set(learning.REPORT_ARMS))
        json.dumps(report, allow_nan=False)

    def test_completed_parent_scheduler_and_unpinned_placement(self):
        task = submit.task_specification("unit_service_level")
        self.assertEqual(task["cpu"], 1)
        self.assertEqual(task["ram_mb"], 3072)
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        parent = {"protocol": spec.source.EXPERIMENT_PROTOCOL, "contract": spec.source.contract(), "coverage_qualified": True}
        receipt = {"scheduler": {"status_counts": {"done": 1}, "exit_code_counts": {"0": 1}}}
        with patch("sys.argv", ["submit", "--run-name", "unit_service_level"]), patch.object(submit, "inventory", return_value=[]), \
                patch.object(submit.Path, "exists", return_value=False), patch.object(submit.Path, "read_text", autospec=True,
                    side_effect=lambda p: json.dumps(receipt if p.name == "completion_receipt.json" else parent)), \
                patch.object(submit.subprocess, "check_output", return_value="revision"), patch.object(submit, "write_json"), patch.object(submit, "execute_bulk") as bulk:
            submit.main()
            self.assertEqual(len(bulk.call_args.args[0]), 1)
            receipt["scheduler"]["exit_code_counts"] = {"1": 1}
            bulk.reset_mock()
            with self.assertRaises(SystemExit): submit.main()
            bulk.assert_not_called()


if __name__ == "__main__":
    unittest.main()
