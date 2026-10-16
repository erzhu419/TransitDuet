import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_service_offset_stage283 as spec
from scripts import submit_native_transit_service_offset_stage283_scheduleurm as submit

learning, policy = spec.learning, spec.policy


class ServiceOffsetTest(unittest.TestCase):
    def fixture(self, n=4):
        raw = np.tile([0., .02, .01, .05, .05, .05], (n, 1))
        risk = np.tile([0., 0., -.005, 0., 0., 0.], (n, 1))
        logits = -np.ones((n, 6))
        logits[:, 5] = 1.
        data = {"originals": np.zeros(n, int), "heads": np.stack((raw+risk, np.zeros_like(raw)), axis=-1),
            "components": np.stack((raw, risk), axis=-1), "raw_logits": logits, "raw_choices": np.zeros(n, int)}
        risks = {a: risk.copy() for a in learning.CONTROLS}
        risks["raw_linear_clip"][:, 2] = -.03
        risks["observed_context_df_clip"][:, 1] = -.04
        services = {"raw": raw.copy(), "mc": raw*.1}
        maps, radius = {}, .005
        for j, k in enumerate(learning.MAPS):
            weights = np.zeros((9 if k == "linear" else 73, 1))
            weights[0, 0], weights[1, 0] = .004, 1.
            maps[k] = {"weights": weights}
            delta = np.tile([0., .008, -.005, .01, -.003, .004], (n, 1))*(1+j*.1)
            services[k] = raw+delta
            services[k+"_clip"] = raw+np.clip(delta, -radius, radius)
        services = {a: services[a] for a in learning.previous.SERVICES}
        clips = {k+"_clip": np.abs(services[k]-raw) > radius for k in learning.MAPS}
        values = np.zeros((n, 32, 6, 4))
        values[..., 2] = raw[:, None]*.1
        values[:, :8, 1, 1] = .01
        values[:, 16:24, 2, 1] = -.01
        values[..., 3] = values[..., 2]*10
        values[..., 0] = values[..., 1]+values[..., 2]
        return data, risks, services, clips, {"maps": maps, "radius": radius}, values

    def test_cached_split_original_anchor_and_clip_nonadditivity(self):
        data, risks, services, clips, model, values = self.fixture()
        before = {k: v.copy() for k, v in services.items()}
        augmented, new_clips, parts = learning.ablate(model, services, data["originals"])
        self.assertEqual(set(augmented), set(learning.SERVICES))
        for k in learning.MAPS:
            offset, slopes = parts[k]
            np.testing.assert_allclose(offset+slopes, services[k]-services["raw"], atol=1e-15)
            self.assertTrue(np.all(offset[:, 0] == 0))
            self.assertTrue(np.all(slopes[:, 0] == 0))
            for p in ("offset", "slopes"):
                self.assertLessEqual(np.max(np.abs(augmented[f"{k}_{p}_clip"]-services["raw"])), model["radius"]+1e-15)
        full_delta = services["linear_clip"]-services["raw"]
        sum_clipped = augmented["linear_offset_clip"]+augmented["linear_slopes_clip"]-2*services["raw"]
        self.assertGreater(np.max(np.abs(full_delta-sum_clipped)), 1e-4)
        for a in before:
            np.testing.assert_array_equal(services[a], before[a])
        choices = learning.previous.decisions(data, risks, augmented)
        old = learning.previous.decisions(data, risks, services)
        for a in learning.previous.ARMS:
            np.testing.assert_array_equal(choices[a], old[a])
        self.assertTrue(all(np.all(v != 5) for v in choices.values()))

    def test_all_signed_MSE_terms_and_eligible_state_denominator(self):
        data, risks, services, clips, model, values = self.fixture(2)
        augmented, _, parts = learning.ablate(model, services, data["originals"])
        eligible = (data["raw_logits"] <= 0)&(np.arange(6)[None] != 0)
        eligible[1] = False
        records, error = learning.error_parts(values[:, :16], data["originals"], services["raw"], eligible, parts)
        self.assertLess(error, 1e-12)
        result = learning.summarize_parts([records["all16"]["linear"]])
        self.assertEqual(result["eligible_state_exposures"], 1)
        self.assertAlmostEqual(result["full_MSE"], sum(result[f] for f in learning.PART_FIELDS[:6]))
        self.assertAlmostEqual(result["full_bias"], result["raw_bias"]+result["offset_mean"]+result["slopes_mean"])
        target = values[:, :16, :, 2].mean(axis=1)
        offset, slopes = parts["linear"]
        for key, delta in (("offset_only_MSE", offset), ("slopes_only_MSE", slopes)):
            expected = np.mean((services["raw"][0, eligible[0]]+delta[0, eligible[0]]-target[0, eligible[0]])**2)
            self.assertAlmostEqual(result[key], expected)

    def test_switch_types_preserve_all_states_and_own_risk_control(self):
        data, risks, services, clips, model, values = self.fixture()
        choices = {r: np.asarray([0, 2, 1, 1]) for r in learning.CONTROLS}
        name = "linear_offset_clip__raw_prior"
        choices[name] = np.asarray([1, 0, 2, 1])
        record = learning.switch_stats(values[:, :16], choices, data["originals"])["all16"][name]
        out = learning.summarize_switches([record])
        self.assertEqual([out[k]["cases"] for k in learning.TYPES], [1, 1, 1, 1])
        expected = (values[np.arange(4), :16, choices[name]]-values[np.arange(4), :16, choices["raw_prior"]]).mean(axis=(0, 1))
        for f, mean in zip(policy.FIELDS, expected):
            self.assertAlmostEqual(sum(out[k][f] for k in learning.TYPES), mean)
        self.assertEqual(out["unchanged"]["delta_cost"], 0.)

    def test_full_cached_cell_and_group_replay_without_fits_or_early_labels(self):
        data, risks, services, clips, model, values = self.fixture(60)
        rows = [{"root": 397, "scene_seed": i//6, "probe_index": i%6} for i in range(60)]
        held = [{**r, "scene_seed": r["scene_seed"]+10} for r in rows]
        folds = [(None, [("train", list(range(60)), "train"), ("held", list(range(60)), "held")])]
        folds += [(list(g), [("retained", list(ix), "train"), ("excluded", list(val), "train"), ("matched_held", list(range(60)), "held")])
            for g, ix, val in spec.cache.learning.joint.binding.policy.prefix_folds(rows)]
        models, queries, previous, saved = [], [], [], {}
        for bank in (0, 1):
            for group, scopes in folds:
                i = 3*len(models)+1
                models.append({**model, "index": i, "bank": bank, "removed": group})
                for scope, ids, label in scopes:
                    d = {k: v[ids] for k, v in data.items()}
                    r = {a: v[ids] for a, v in risks.items()}
                    s = {a: v[ids] for a, v in services.items()}
                    c = {a: v[ids] for a, v in clips.items()}
                    choices = learning.previous.decisions(d, r, s)
                    eligible = (d["raw_logits"] <= 0)&(np.arange(6)[None] != 0)
                    queries.append({"index": i, "bank": bank, "removed": group, "scope": scope, "label": label, "ids": ids,
                        "eligible": eligible.tolist(), "radius": model["radius"], "services": s, "clips": c, "choices": choices})
                    saved.update({f"{i}_{scope}_{a}": v for a, v in r.items()})
                    for future, b in (("same", bank), ("opposite", 1-bank)):
                        v = values[ids, b*16:(b+1)*16]
                        previous.append({"bank": bank, "scope": scope+"_"+future, "outcomes": policy.outcomes(v, choices),
                            "metrics": learning.previous.metrics(s, c, s["raw"], v, d["originals"], eligible, model["radius"])})
        class Archive(dict):
            def __enter__(self): return self
            def __exit__(self, *args): return False
        frozen, accesses = [], []
        def read(path):
            if path.name == "models.json": payload = models
            elif path.name == "predictions.json": payload = queries
            else:
                self.assertEqual(path.name, "outcomes.json")
                self.assertTrue(frozen)
                payload = previous
            return json.dumps(payload, default=lambda a: a.tolist())
        def write(path, payload):
            if path.name == "predictions.json": frozen.append(True)
        def draws(rs, panels, ds):
            self.assertTrue(frozen)
            accesses.append(True)
            return values[[r["scene_seed"]%10*6+r["probe_index"] for r in rs]][:, ds]
        with tempfile.TemporaryDirectory() as tmp, patch.object(spec.Path, "read_text", autospec=True, side_effect=read), \
                patch.object(spec.np, "load", return_value=Archive(saved)), patch.object(spec.source.geometry.source, "read_cell", return_value={"reference": {}}), \
                patch.object(spec, "write_json", side_effect=write), patch.object(spec.cache.cached.training.cohort, "split_rows", return_value=(rows, held)), \
                patch.object(spec.cache.learning.base, "causal_data", return_value=data), patch.object(spec.cache.learning.base, "training_draws", side_effect=draws), \
                patch.object(spec.source.learning, "fit", side_effect=AssertionError("Service refit")), \
                patch.object(spec.cache.learning, "fit_inner", side_effect=AssertionError("Prior refit")):
            result = spec.analyze_cell(spec.Path(tmp), spec.KEYS[0], {spec.KEYS[0][0]: []}, {})
        self.assertEqual(len(accesses), 128)
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
                    before["groups"][g][label][scope] = {"service_metrics": {a: learning.previous.base.summarize([r["metrics"][a] for r in rs], True) for a in learning.previous.SERVICES},
                        "outcomes": {h: {a: policy.summarize([r["outcomes"][h][a] for r in rs]) for a in learning.previous.ARMS} for h in policy.DRAW_GROUPS}}
        report = spec.report([{**result, "key": list(k)} for k in spec.KEYS], before)
        self.assertTrue(report["coverage_qualified"])
        self.assertLess(report["reproduction"]["group_replay_error"], 1e-12)
        primary = report["groups"]["mixed_pooled"]["pooled"]["held_opposite"]
        self.assertEqual(set(primary["outcomes"]), {"all16", "first8", "last8"})
        self.assertEqual(set(primary["switches"]), set(learning.REPORT_ARMS))
        self.assertEqual(set(report["groups"]["root_397"]["pooled"]["held_opposite"]["outcomes"]), {"all16"})
        json.dumps(report, allow_nan=False)

    def test_scheduler_unpinned_and_completed_parent_required(self):
        task = submit.task_specification("unit_offset")
        self.assertEqual(task["cpu"], 1)
        self.assertEqual(task["ram_mb"], 3072)
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        parent = {"protocol": spec.source.EXPERIMENT_PROTOCOL, "contract": spec.source.contract(), "coverage_qualified": True, "fits": {"service_head_fits": 2112}}
        receipt = {"scheduler": {"status_counts": {"done": 1}, "exit_code_counts": {"0": 1}}}
        with patch("sys.argv", ["submit", "--run-name", "unit_offset"]), patch.object(submit, "inventory", return_value=[]), \
                patch.object(submit.Path, "exists", return_value=False), patch.object(submit.Path, "read_text", autospec=True,
                    side_effect=lambda p: json.dumps(receipt if p.name == "completion_receipt.json" else parent)), \
                patch.object(submit.subprocess, "check_output", return_value="revision"), patch.object(submit, "write_json"), patch.object(submit, "execute_bulk") as bulk:
            submit.main()
            self.assertEqual(len(bulk.call_args.args[0]), 1)
            bulk.reset_mock()
            receipt["scheduler"]["exit_code_counts"] = {"1": 1}
            with self.assertRaises(SystemExit): submit.main()
            bulk.assert_not_called()


if __name__ == "__main__":
    unittest.main()
