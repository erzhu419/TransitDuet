import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_risk_policy_stage279 as spec
from scripts import submit_native_transit_risk_policy_stage279_scheduleurm as submit

learning = spec.learning


class RiskPolicyTest(unittest.TestCase):
    def fixture(self, n=12):
        rng = np.random.default_rng(279)
        originals = np.arange(n)%6
        parts = rng.normal(size=(n, 6, 2))*.01;parts[np.arange(n), originals] = 0.
        heads = np.stack((parts.sum(axis=-1), np.zeros((n, 6))), axis=-1)
        logits = rng.choice([-1., 1.], size=(n, 6));logits[np.arange(n), originals] = 0.
        data = {"components": parts, "heads": heads, "raw_logits": logits, "originals": originals,
            "raw_choices": learning.joint.binding.policy.selected(heads[..., 0], logits, originals)}
        raw = parts[..., 1]
        risks = {a: raw.copy() for a in learning.ARMS}
        for a in learning.ARMS[1:]:
            correction = rng.uniform(-.002, .002, size=(n, 6));correction[np.arange(n), originals] = 0.
            risks[a] += correction
        values = rng.normal(size=(n, 32, 6, 4))*.02
        values[..., 0] = values[..., 1]+values[..., 2]
        return data, risks, values

    def test_risk_only_swap_preserves_raw_service_gate_and_choices(self):
        data, risks, _ = self.fixture();choices, err = learning.decisions(data, risks)
        self.assertEqual(err, 0.)
        np.testing.assert_array_equal(choices["raw_prior"], data["raw_choices"])
        for a, chosen in choices.items():
            expected = data["components"][..., 0]+risks[a]
            np.testing.assert_array_equal(chosen, learning.joint.binding.policy.selected(expected, data["raw_logits"], data["originals"]))
        changed = {**risks, "raw_prior": risks["raw_prior"]+.1}
        with self.assertRaisesRegex(ValueError, "Frozen raw risk"): learning.decisions(data, changed)

    def test_identity_first_tie_and_blocked_low_cost(self):
        originals = np.array([3]);zero = np.zeros((1, 6))
        data = {"originals": originals, "components": np.zeros((1, 6, 2)), "heads": np.zeros((1, 6, 2)),
            "raw_logits": np.array([[1., -1., -1., 0., -1., -1.]]), "raw_choices": originals}
        risks = {a: zero.copy() for a in learning.ARMS};risks["observed_context_df_clip"][0, 0] = -1e6
        choices, _ = learning.decisions(data, risks)
        for chosen in choices.values(): np.testing.assert_array_equal(chosen, originals)

    def test_physical_outcomes_half_identity_and_paired_differences(self):
        data, risks, values = self.fixture();choices, _ = learning.decisions(data, risks)
        stats = learning.outcomes(values[:, :16], choices)
        for a in learning.ARMS:
            full = learning.summarize([stats["all16"][a]])
            for f in learning.FIELDS:
                halves = [learning.summarize([stats[h][a]])[f] for h in ("first8", "last8")]
                self.assertAlmostEqual(full[f], sum(halves)/2)
            self.assertAlmostEqual(full["delta_cost"], full["delta_fleet"]+full["delta_service"])
            if a == "raw_prior":
                for f in learning.FIELDS: self.assertEqual(full[f], 0.)
                self.assertEqual(full["changed_rate"], 0.)
        broken = values[:, :16].copy();broken[..., 0] += np.arange(6)[None, None, :]
        choices["observed_context_df_clip"] = (choices["raw_prior"]+1)%6
        with self.assertRaisesRegex(ValueError, "Physical total"): learning.outcomes(broken, choices)

    def test_full_cell_no_fits_and_actions_frozen_before_any_outcomes(self):
        data, risks, values = self.fixture(60)
        rows = [{"root": 397, "scene_seed": i//6, "probe_index": i%6} for i in range(60)]
        held = [{**r, "scene_seed": r["scene_seed"]+10} for r in rows]
        mask = (data["raw_logits"] <= 0)&(np.arange(6)[None] != data["originals"][:, None])
        folds = [(None, list(range(60)), [("train", list(range(60)), "train"), ("held", list(range(60)), "held")])]
        folds += [(list(g), list(ix), [("retained", list(ix), "train"), ("excluded", list(val), "train"), ("matched_held", list(range(60)), "held")]) for g, ix, val in spec.cache.learning.joint.binding.policy.prefix_folds(rows)]
        models, queries, errors, saved = [], [], [], {}
        for bank in (0, 1):
            for group, fit_indices, scopes in folds:
                i = len(models)
                models.append({"index": i, "bank": bank, "removed": group, "fit_indices": fit_indices,
                    "training_draws": list(range(bank*16, (bank+1)*16)), "radius": .002})
                for scope, ids, label in scopes:
                    queries.append({"index": i, "bank": bank, "removed": group, "scope": scope, "label": label, "ids": ids, "eligible": mask[ids].tolist(), "radius": .002})
                    saved.update({f"{i}_{scope}_{a}": v[ids] for a, v in risks.items()})
                    for future, b in (("same", bank), ("opposite", 1-bank)):
                        v = values[ids, b*16:(b+1)*16]
                        y = v[..., 1]-v[np.arange(len(ids)), :, data["originals"][ids], 1][:, :, None]
                        metrics = {a: spec.source.learning.previous.base.metrics(r[ids], risks["raw_prior"][ids], y, mask[ids], .002) for a, r in risks.items()}
                        errors.append({"bank": bank, "scope": scope+"_"+future, "metrics": metrics})
        class Archive(dict):
            def __enter__(self): return self
            def __exit__(self, *args): return False
        files = {"models.json": models, "queries.json": queries, "errors.json": errors}
        frozen, accesses = [], []
        def write(path, payload):
            if path.name == "actions.json": frozen.append(True)
        def draws(rs, panels, ds):
            self.assertTrue(frozen);accesses.append(True)
            return values[[r["scene_seed"]%10*6+r["probe_index"] for r in rs]][:, ds]
        with tempfile.TemporaryDirectory() as tmp, patch.object(spec.Path, "read_text", autospec=True, side_effect=lambda p: json.dumps(files[p.name])), \
                patch.object(spec.np, "load", return_value=Archive(saved)), patch.object(spec.source.source, "read_cell", return_value={"reference": {}}), \
                patch.object(spec, "write_json", side_effect=write), patch.object(spec.cache.cached.training.cohort, "split_rows", return_value=(rows, held)), \
                patch.object(spec.cache.learning.base, "causal_data", return_value=data), patch.object(spec.cache.learning.base, "training_draws", side_effect=draws), \
                patch.object(spec.source.learning, "fit", side_effect=AssertionError("Residual refit")), \
                patch.object(spec.cache.learning, "fit_inner", side_effect=AssertionError("Prior refit")), \
                patch.object(spec.cache.learning, "select", side_effect=AssertionError("OOF strength selection")):
            result = spec.analyze_cell(spec.Path(tmp), spec.KEYS[0], {spec.KEYS[0][0]: []}, {})
        self.assertEqual(len(accesses), 128);self.assertEqual(result["model_state_exposures"], 2640)
        self.assertEqual(result["risk_metric_error"], 0.);self.assertEqual(result["raw_risk_error"], 0.)
        cells = [{**result, "key": list(k)} for k in spec.KEYS]
        # The source uses eligible states; policy outcomes intentionally use all states.
        before = {"groups": {}}
        for g in ("mixed_pooled", *spec.VIEWS):
            before["groups"][g] = {}
            for b, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
                before["groups"][g][b] = {}
                for scope in spec.SCOPES:
                    rs = [r for r in result["records"] if r["scope"] == scope and (bank is None or r["bank"] == bank)]
                    arms = {a: {"MSE": sum(r["risk_metrics"][a]["squared_error_sum"] for r in rs)/sum(r["risk_metrics"][a]["state_exposures"] for r in rs)} for a in learning.ARMS}
                    before["groups"][g][b][scope] = {"arms": arms}
        report = spec.report(cells, before)
        self.assertTrue(report["coverage_qualified"]);self.assertLess(report["reproduction"]["group_risk_MSE_error"], 1e-12)
        self.assertEqual(report["groups"]["mixed_pooled"]["pooled"]["held_opposite"]["outcomes"]["all16"]["raw_prior"]["cases"], 1920)

    def test_scheduler_unpinned_small_pull_and_parent_exit_gate(self):
        t = submit.task_specification("unit_risk_policy")
        self.assertEqual(t["cpu"], 1);self.assertIsNone(t["require_node"])
        self.assertEqual(t["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertTrue(t["result_dir"].endswith("/report"))
        parent = {"protocol": spec.source.EXPERIMENT_PROTOCOL, "contract": spec.source.contract(), "coverage_qualified": True}
        receipt = {"scheduler": {"status_counts": {"done": 1}, "exit_code_counts": {"0": 1}}}
        with patch("sys.argv", ["submit", "--run-name", "unit_risk_policy"]), patch.object(submit, "inventory", return_value=[]), patch.object(submit.Path, "exists", return_value=False), \
                patch.object(submit.Path, "read_text", autospec=True, side_effect=lambda p: json.dumps(receipt if p.name == "completion_receipt.json" else parent)), \
                patch.object(submit.subprocess, "check_output", return_value="revision"), patch.object(submit, "write_json"), patch.object(submit, "execute_bulk") as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]), 1)
            bulk.reset_mock();receipt["scheduler"]["exit_code_counts"] = {"1": 1}
            with self.assertRaises(SystemExit): submit.main()
            bulk.assert_not_called()


if __name__ == "__main__": unittest.main()
