import copy
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_service_decision_error_stage286 as spec
from scripts import submit_native_transit_service_decision_error_stage286_scheduleurm as submit
import test_native_transit_service_level_stage285 as parent_test

learning, policy = spec.learning, spec.policy


class ServiceDecisionErrorTest(unittest.TestCase):
    def fixture(self, n=60):
        data, risks, services, clips, cached, values, parts, observed, eligible = parent_test.ServiceLevelTest().fixture(n)
        coords = {k: np.zeros((n, 8)) for k in learning.previous.KINDS}
        model = {"radius": .02, "maps": {k: {"center": np.zeros(8), "scale": np.ones(8),
            "weights": np.r_[-.015 if k == "observed_level" else -.01, np.zeros(8)][:, None]} for k in learning.previous.KINDS}}
        services, _ = learning.previous.predict(model, coords, services, eligible, data["originals"])
        return data, risks, services, values, eligible

    def test_frozen_candidate_menu_empty_state_and_MSE_identity(self):
        data, risks, services, values, eligible = self.fixture(3)
        data["raw_logits"][2] = 1.
        mask, candidates, pred = learning.candidates(data, risks, services)
        self.assertFalse(mask[2].any())
        for r, ids in candidates.items():
            self.assertEqual(ids[2], data["originals"][2])
            self.assertTrue(mask[0, ids[0]])
        self.assertEqual(candidates["raw_prior"][0], 2)
        self.assertEqual(candidates["observed_context_df_clip"][0], 1)
        before = {r: v.copy() for r, v in candidates.items()}
        changed = {s: v+100. for s, v in services.items()}
        _, after, _ = learning.candidates(data, risks, changed)
        for r in before: np.testing.assert_array_equal(before[r], after[r])
        records, error = learning.errors(values[:, :16], data["originals"], mask, candidates, pred)
        self.assertLess(error, 1e-12)
        for row in records["all16"].values():
            m = learning.summarize_errors([row])
            self.assertEqual(m["eligible_state_exposures"], 2)
            self.assertAlmostEqual(m["margin_MSE"], m["service_MSE"]+m["risk_MSE"]+m["service_risk_cross"])
            self.assertGreaterEqual(m["margin_sign_error"], 0.)
            self.assertLessEqual(m["margin_sign_error"], 1.)

    def test_shared_future_replication_does_not_shrink_MC_error(self):
        rng = np.random.default_rng(286)
        deltas = rng.normal(size=(2, len(learning.PAIRS), 16, 4))
        deltas[..., 0] = deltas[..., 1]+deltas[..., 2]
        states = [(397, 1, 0), (397, 2, 0)]
        bucket = learning.new_bucket()
        learning.add_units(bucket, states, deltas)
        first = learning.summarize_units(bucket)
        learning.add_units(bucket, states, deltas)
        doubled = learning.summarize_units(bucket)
        for name, a in first.items():
            b = doubled[name]
            self.assertEqual(a["unique_state_future_units"], b["unique_state_future_units"])
            self.assertEqual(b["state_exposures"], 2*a["state_exposures"])
            for f in policy.FIELDS:
                self.assertAlmostEqual(a["mean"][f], b["mean"][f])
                self.assertAlmostEqual(a["conditional_MC_se"][f], b["conditional_MC_se"][f])
                self.assertAlmostEqual(sum(p[f] for p in b["prefix_contributions"]), b["mean"][f])
            self.assertEqual(b["fleet_up_exposures"], 2*a["fleet_up_exposures"])
            self.assertEqual(b["net_fleet_up_unique_units"], a["net_fleet_up_unique_units"])
        expected_se = np.sqrt(deltas[:, 0, :, 0].var(axis=1, ddof=1).sum()/16)/2
        self.assertAlmostEqual(next(iter(first.values()))["conditional_MC_se"]["delta_cost"], expected_se)

    def test_complete_cell_and_group_replay_no_training_or_early_labels(self):
        data, risks, services, values, eligible = self.fixture()
        rows = [{"root": 397, "scene_seed": i//6, "probe_index": i%6} for i in range(60)]
        held = [{**r, "scene_seed": r["scene_seed"]+10} for r in rows]
        folds = [(None, [("train", list(range(60)), "train"), ("held", list(range(60)), "held")])]
        folds += [(list(g), [("retained", list(ix), "train"), ("excluded", list(val), "train"), ("matched_held", list(range(60)), "held")])
            for g, ix, val in spec.cache.learning.joint.binding.policy.prefix_folds(rows)]
        queries, old_records, saved = [], [], {}
        for bank in (0, 1):
            for group, scopes in folds:
                i = 3*len(queries)+1
                for scope, ids, label in scopes:
                    d = {k: v[ids] for k, v in data.items()}
                    r = {a: v[ids] for a, v in risks.items()}
                    s = {a: v[ids] for a, v in services.items()}
                    choices = learning.service.decisions(d, r, s)
                    queries.append({"index": i, "bank": bank, "removed": group, "scope": scope, "label": label, "ids": ids,
                        "eligible": eligible[ids].tolist(), "services": s, "choices": choices})
                    saved.update({f"{i}_{scope}_{a}": v for a, v in r.items()})
                    for future, b in (("same", bank), ("opposite", 1-bank)):
                        ds, _ = spec.cache.sampling.learning.draw_assignment(b, "expanded")
                        old_records.append({"bank": bank, "scope": scope+"_"+future, "outcomes": policy.outcomes(values[ids][:, ds], choices)})
        class Archive(dict):
            def __enter__(self): return self
            def __exit__(self, *args): return False
        frozen, accesses = [], []
        def read(path):
            if path.name == "predictions.json": payload = queries
            else:
                self.assertEqual(path.name, "outcomes.json")
                self.assertTrue(frozen)
                payload = old_records
            return json.dumps(payload, default=lambda a: a.tolist())
        def write(path, payload):
            if path.name == "candidate_predictions.json":
                self.assertEqual(len(payload), 44)
                frozen.append(True)
        def draws(rs, panels, ds):
            self.assertTrue(frozen)
            accesses.append(True)
            return values[[r["scene_seed"]%10*6+r["probe_index"] for r in rs]][:, ds]
        units = {}
        with tempfile.TemporaryDirectory() as tmp, patch.object(spec.Path, "read_text", autospec=True, side_effect=read), \
                patch.object(spec.np, "load", return_value=Archive(saved)), patch.object(spec.source.transfer.geometry.source, "read_cell", return_value={"reference": {}}), \
                patch.object(spec, "write_json", side_effect=write), patch.object(spec.cache.cached.training.cohort, "split_rows", return_value=(rows, held)), \
                patch.object(spec.cache.learning.base, "causal_data", return_value=data), patch.object(spec.cache.learning.base, "training_draws", side_effect=draws), \
                patch.object(spec.source.learning, "fit", side_effect=AssertionError("Head refit")), \
                patch.object(spec.cache.learning, "fit_inner", side_effect=AssertionError("Raw refit")), \
                patch.object(learning.service, "decisions", side_effect=AssertionError("New policy")):
            result = spec.analyze_cell(spec.Path(tmp), spec.KEYS[0], {spec.KEYS[0][0]: []}, {}, units)
        self.assertEqual(len(accesses), 44)
        self.assertEqual(result["model_state_exposures"], 1560)
        before, aggregate_units = {"groups": {}}, {}
        for g in spec.GROUPS:
            keys = [k for k in spec.KEYS if g in spec.group_names(k)]
            before["groups"][g] = {}
            for bank in (0, 1):
                before["groups"][g][f"bank{bank}"] = {}
                for scope in spec.SCOPES:
                    rs = [r for _ in keys for r in old_records if r["bank"] == bank and r["scope"] == scope]
                    before["groups"][g][f"bank{bank}"][scope] = {"outcomes": {h: {a: policy.summarize([r["outcomes"][h][a] for r in rs]) for a in learning.ARM_RISK} for h in policy.DRAW_GROUPS}}
                    b = copy.deepcopy(units["root_397", bank, scope])
                    b["cases"] *= len(keys)
                    b["fleet_up"] *= len(keys)
                    b["fleet_down"] *= len(keys)
                    for state in b["units"]:
                        b["units"][state] *= len(keys)
                        b["state_cases"][state] *= len(keys)
                    aggregate_units[g, bank, scope] = b
        report = spec.report([{**result, "key": list(k)} for k in spec.KEYS], before, aggregate_units)
        self.assertTrue(report["coverage_qualified"])
        self.assertLess(report["reproduction"]["group_replay_error"], 1e-12)
        self.assertLess(report["reproduction"]["unique_unit_or_prefix_sum_error"], 1e-12)
        self.assertEqual(set(report["groups"]["mixed_pooled"]), {"bank0", "bank1"})
        json.dumps(report, allow_nan=False)

    def test_scheduler_completed_parent_and_unpinned_cpu(self):
        task = submit.task_specification("unit_decision_error")
        self.assertEqual(task["cpu"], 1)
        self.assertEqual(task["ram_mb"], 3072)
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        parent = {"protocol": spec.source.EXPERIMENT_PROTOCOL, "contract": spec.source.contract(), "coverage_qualified": True,
            "fits": {"level_head_fits": 1408}}
        receipt = {"scheduler": {"status_counts": {"done": 1}, "exit_code_counts": {"0": 1}}}
        with patch("sys.argv", ["submit", "--run-name", "unit_decision_error"]), patch.object(submit, "inventory", return_value=[]), \
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
