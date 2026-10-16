import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_service_centering_stage284 as spec
from scripts import submit_native_transit_service_centering_stage284_scheduleurm as submit
from test_native_transit_service_offset_stage283 import ServiceOffsetTest

learning, policy = spec.learning, spec.policy


class ServiceCenteringTest(unittest.TestCase):
    def fixture(self, n=4):
        data, risks, services, clips, model, values = ServiceOffsetTest().fixture(n)
        services, added, parts = learning.previous.ablate(model, services, data["originals"])
        return data, risks, services, {**clips, **added}, model, values, parts

    def test_prediction_only_split_anchors_empty_and_single_action(self):
        data, risks, services, clips, model, values, parts = self.fixture()
        eligible = (data["raw_logits"] <= 0)&(np.arange(6)[None] != 0)
        eligible[1] = False
        eligible[2] = False
        eligible[2, 2] = True
        augmented, _ = learning.ablate(services, parts, eligible, data["originals"], 1.)
        for k in learning.MAPS:
            delta = services[k]-services["raw"]
            level = augmented[k+"_level_clip"]-services["raw"]
            centered = augmented[k+"_centered_clip"]-services["raw"]
            fixed = augmented[k+"_offset_centered_clip"]-services["raw"]
            np.testing.assert_allclose(level+centered, delta, atol=1e-15)
            np.testing.assert_allclose(fixed, centered+parts[k][0], atol=1e-15)
            np.testing.assert_allclose(np.sum(np.where(eligible, centered, 0.), axis=1), 0., atol=1e-15)
            self.assertEqual(level[1, 1], 0.)
            self.assertAlmostEqual(centered[2, 2], 0.)
            self.assertTrue(np.all(level[:, 0] == 0.))
            self.assertTrue(np.all(centered[:, 0] == 0.))
        original = learning.previous.previous.decisions(data, risks, services)
        choices = learning.previous.previous.decisions(data, risks, augmented)
        for a in learning.previous.ARMS:
            np.testing.assert_array_equal(original[a], choices[a])
        clipped, _ = learning.ablate(services, parts, eligible, data["originals"], model["radius"])
        for name in learning.NEW_SERVICES:
            self.assertLessEqual(np.max(np.abs(clipped[name]-services["raw"])), model["radius"]+1e-15)

    def test_level_contrast_and_pair_ranking_denominators(self):
        eligible = np.asarray([[False, True, True, False, False, False],
            [False, True, True, True, False, False], [False]*6])
        target = np.tile([0., 1., 2., 3., 0., 0.], (3, 1))
        values = np.zeros((3, 16, 6, 4))
        values[..., 2] = target[:, None]
        # A level shift changes MSE but not pair ranking; contrasts reverse rank.
        services = {"raw": target.copy(), "shift": target+2., "reverse": -target, "tie": np.zeros_like(target)}
        records, error = learning.error_geometry(values, services, np.zeros(3, int), eligible)
        self.assertLess(error, 1e-12)
        result = {s: learning.summarize([r]) for s, r in records["all16"].items()}
        self.assertEqual(result["shift"]["eligible_state_exposures"], 2)
        self.assertEqual(result["shift"]["rank_state_exposures"], 2)
        self.assertAlmostEqual(result["shift"]["MSE"], 4.)
        self.assertAlmostEqual(result["shift"]["contrast_MSE"], 0.)
        self.assertEqual(result["shift"]["pair_inversion_rate"], 0.)
        self.assertEqual(result["reverse"]["pair_inversion_rate"], 1.)
        self.assertEqual(result["tie"]["pair_inversion_rate"], .5)
        for r in result.values():
            self.assertAlmostEqual(r["MSE"], r["level_MSE"]+r["contrast_MSE"])
        values[:, :, 2, 2] = values[:, :, 1, 2]
        records, _ = learning.error_geometry(values, services, np.zeros(3, int), eligible)
        self.assertEqual(records["all16"]["shift"]["rank_cases"], 1)

    def test_full_cell_group_replay_no_fits_or_early_labels(self):
        data, risks, services, clips, model, values, parts = self.fixture(60)
        rows = [{"root": 397, "scene_seed": i//6, "probe_index": i%6} for i in range(60)]
        held = [{**r, "scene_seed": r["scene_seed"]+10} for r in rows]
        folds = [(None, [("train", list(range(60)), "train"), ("held", list(range(60)), "held")])]
        folds += [(list(g), [("retained", list(ix), "train"), ("excluded", list(val), "train"), ("matched_held", list(range(60)), "held")])
            for g, ix, val in spec.cache.learning.joint.binding.policy.prefix_folds(rows)]
        queries, previous, saved = [], [], {}
        for bank in (0, 1):
            for group, scopes in folds:
                i = 3*len(queries)+1
                for scope, ids, label in scopes:
                    d = {k: v[ids] for k, v in data.items()}
                    r = {a: v[ids] for a, v in risks.items()}
                    s = {a: v[ids] for a, v in services.items()}
                    c = {a: v[ids] for a, v in clips.items()}
                    p = {a: [v[ids] for v in pair] for a, pair in parts.items()}
                    choices = learning.previous.previous.decisions(d, r, s)
                    eligible = (d["raw_logits"] <= 0)&(np.arange(6)[None] != 0)
                    queries.append({"index": i, "bank": bank, "removed": group, "scope": scope, "label": label, "ids": ids,
                        "eligible": eligible.tolist(), "radius": model["radius"], "services": s, "clips": c, "choices": choices, "parts": p})
                    saved.update({f"{i}_{scope}_{a}": v for a, v in r.items()})
                    for future, b in (("same", bank), ("opposite", 1-bank)):
                        v = values[ids, b*16:(b+1)*16]
                        previous.append({"bank": bank, "scope": scope+"_"+future, "outcomes": policy.outcomes(v, choices),
                            "metrics": learning.previous.previous.metrics(s, c, s["raw"], v, d["originals"], eligible, model["radius"])})
        class Archive(dict):
            def __enter__(self): return self
            def __exit__(self, *args): return False
        frozen, accesses = [], []
        def read(path):
            if path.name == "predictions.json": payload = queries
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
                patch.object(spec.np, "load", return_value=Archive(saved)), patch.object(spec.source.source.geometry.source, "read_cell", return_value={"reference": {}}), \
                patch.object(spec, "write_json", side_effect=write), patch.object(spec.cache.cached.training.cohort, "split_rows", return_value=(rows, held)), \
                patch.object(spec.cache.learning.base, "causal_data", return_value=data), patch.object(spec.cache.learning.base, "training_draws", side_effect=draws), \
                patch.object(spec.source.source.learning, "fit", side_effect=AssertionError("Service refit")), \
                patch.object(spec.cache.learning, "fit_inner", side_effect=AssertionError("Prior refit")):
            result = spec.analyze_cell(spec.Path(tmp), spec.KEYS[0], {spec.KEYS[0][0]: []}, {})
        self.assertEqual(len(accesses), 128)
        self.assertEqual(result["model_state_exposures"], 2640)
        before = {"groups": {}}
        for g in ("mixed_pooled", *spec.VIEWS):
            keys = [k for k in spec.KEYS if k[0].startswith("mixed")] if g == "mixed_pooled" else [k for k in spec.KEYS if k[0] == g]
            before["groups"][g] = {}
            for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
                before["groups"][g][label] = {}
                for scope in spec.SCOPES:
                    rs = [r for _ in keys for r in previous if r["scope"] == scope and (bank is None or r["bank"] == bank)]
                    before["groups"][g][label][scope] = {"service_metrics": {a: {f: m[f] for f in ("MSE", "bias", "clipped_action_fraction")}
                        for a in learning.previous.SERVICES for m in [learning.previous.previous.base.summarize([r["metrics"][a] for r in rs], True)]},
                        "outcomes": {h: {a: policy.summarize([r["outcomes"][h][a] for r in rs]) for a in learning.previous.REPORT_ARMS} for h in policy.DRAW_GROUPS}}
        # The parent retains halves only on the primary mixed populations.
        for g, banks in before["groups"].items():
            for scopes in banks.values():
                for scope, q in scopes.items():
                    if g != "mixed_pooled" or scope not in (*spec.PRIMARY, "matched_held_opposite"):
                        q["outcomes"] = {"all16": q["outcomes"]["all16"]}
        report = spec.report([{**result, "key": list(k)} for k in spec.KEYS], before)
        self.assertTrue(report["coverage_qualified"])
        self.assertLess(report["reproduction"]["group_replay_error"], 1e-12)
        self.assertLess(report["reproduction"]["switch_partition_error"], 1e-12)
        primary = report["groups"]["mixed_pooled"]["pooled"]["held_opposite"]
        self.assertEqual(set(primary["error_geometry"]), {"all16", "first8", "last8"})
        self.assertEqual(set(primary["switches"]), set(learning.REPORT_ARMS))
        json.dumps(report, allow_nan=False)

    def test_completed_parent_scheduler_and_unpinned_placement(self):
        task = submit.task_specification("unit_centering")
        self.assertEqual(task["cpu"], 1)
        self.assertEqual(task["ram_mb"], 3072)
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        parent = {"protocol": spec.source.EXPERIMENT_PROTOCOL, "contract": spec.source.contract(), "coverage_qualified": True}
        receipt = {"scheduler": {"status_counts": {"done": 1}, "exit_code_counts": {"0": 1}}}
        with patch("sys.argv", ["submit", "--run-name", "unit_centering"]), patch.object(submit, "inventory", return_value=[]), \
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
