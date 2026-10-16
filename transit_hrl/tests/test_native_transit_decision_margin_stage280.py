import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_decision_margin_stage280 as spec
from scripts import submit_native_transit_decision_margin_stage280_scheduleurm as submit

learning = spec.learning


class DecisionMarginTest(unittest.TestCase):
    def fixture(self, n=1):
        service = np.tile([0., .02, 0., .05, .05, .05], (n, 1))
        risk = np.tile([0., 0., -.01, 0., 0., 0.], (n, 1))
        data = {"originals": np.zeros(n, int), "components": np.stack((service, risk), axis=-1),
            "heads": np.stack((service+risk, np.zeros_like(service)), axis=-1),
            "raw_logits": -np.ones((n, 6)), "raw_choices": np.full(n, 2)}
        risks = {a: risk.copy() for a in learning.policy.ARMS}
        risks["observed_context_df_clip"][:, 1] = -.06
        choices, _ = learning.policy.decisions(data, risks)
        values = np.zeros((n, 32, 6, 4));values[:, :, 1, 2] = .03;values[:, :, 2, 1] = .01
        values[..., 0] = values[..., 1]+values[..., 2]
        return data, risks, choices, values

    def test_head_errors_exactly_explain_bad_predicted_improvement(self):
        d, r, choices, v = self.fixture()
        p = learning.predictions(d, r, choices)
        stats, err = learning.audit(v[:, :16], p)
        result = learning.summarize([stats["all16"]["observed_vs_linear9"]])
        for key, expected in (("predicted_total", -.03), ("actual_cost", .02), ("fleet_error", .04), ("service_error", .01),
                ("risk_MSE_proposed", .04**2), ("risk_MSE_control", .02**2)):
            self.assertAlmostEqual(result[key], expected)
        self.assertEqual(result["fleet_save_draws"], 16);self.assertEqual(result["fleet_add_draws"], 0)
        self.assertEqual(result["disagreement_cases"], 1);self.assertLess(err, 1e-12)
        for f in learning.MEANS:
            halves = [learning.summarize([stats[h]["observed_vs_linear9"]])[f] for h in ("first8", "last8")]
            self.assertAlmostEqual(result[f], sum(halves)/2)

    def test_no_disagreement_has_no_conditional_mse_and_bad_minima_fail(self):
        d, r, choices, v = self.fixture()
        identical = {a: choices["raw_prior"].copy() for a in choices}
        # All critics equal the raw predictor in this no-disagreement control.
        raw_risks = {a: r["raw_prior"] for a in r}
        p = learning.predictions(d, raw_risks, identical)
        stats, _ = learning.audit(v[:, :16], p)
        for s in stats["all16"].values():
            result = learning.summarize([s])
            self.assertIsNone(result["risk_MSE_proposed"]);self.assertEqual(result["actual_cost"], 0.)
        wrong = {**choices, "observed_context_df_clip": np.zeros(1, int)}
        with self.assertRaisesRegex(ValueError, "controller minima"): learning.predictions(d, r, wrong)

    def test_complete_cell_group_replay_and_no_labels_before_freeze(self):
        data, risks, choices, values = self.fixture(60)
        rows = [{"root": 397, "scene_seed": i//6, "probe_index": i%6} for i in range(60)]
        held = [{**r, "scene_seed": r["scene_seed"]+10} for r in rows]
        folds = [(None, [("train", list(range(60)), "train"), ("held", list(range(60)), "held")])]
        folds += [(list(g), [("retained", list(ix), "train"), ("excluded", list(val), "train"), ("matched_held", list(range(60)), "held")]) for g, ix, val in spec.cache.learning.joint.binding.policy.prefix_folds(rows)]
        queries, previous, saved = [], [], {}
        for bank in (0, 1):
            for group, scopes in folds:
                i = len({q["index"] for q in queries})
                for scope, ids, label in scopes:
                    ch = {a: x[ids] for a, x in choices.items()}
                    queries.append({"index": i, "bank": bank, "scope": scope, "label": label, "ids": ids, "removed": group, "choices": {a: x.tolist() for a, x in ch.items()}})
                    saved.update({f"{i}_{scope}_{a}": v[ids] for a, v in risks.items()})
                    for future, b in (("same", bank), ("opposite", 1-bank)):
                        previous.append({"bank": bank, "scope": scope+"_"+future,
                            "outcomes": learning.policy.outcomes(values[ids, b*16:(b+1)*16], ch)})
        class Archive(dict):
            def __enter__(self): return self
            def __exit__(self, *args): return False
        frozen, accesses = [], []
        def read(path):
            if path.name == "outcomes.json": self.assertTrue(frozen)
            return json.dumps(queries if path.name == "actions.json" else previous)
        def write(path, payload):
            if path.name == "predicted_margins.json": frozen.append(True)
        def draws(rs, panels, ds):
            self.assertTrue(frozen);accesses.append(True)
            return values[[r["scene_seed"]%10*6+r["probe_index"] for r in rs]][:, ds]
        with tempfile.TemporaryDirectory() as tmp, patch.object(spec.Path, "read_text", autospec=True, side_effect=read), \
                patch.object(spec.np, "load", return_value=Archive(saved)), patch.object(spec, "read_cell", return_value={"reference": {}}), \
                patch.object(spec, "write_json", side_effect=write), patch.object(spec.cache.cached.training.cohort, "split_rows", return_value=(rows, held)), \
                patch.object(spec.cache.learning.base, "causal_data", return_value=data), patch.object(spec.cache.learning.base, "training_draws", side_effect=draws), \
                patch.object(spec.source.source.learning, "fit", side_effect=AssertionError("Residual refit")), \
                patch.object(spec.cache.learning, "fit_inner", side_effect=AssertionError("Prior refit")), \
                patch.object(spec.cache.learning, "select", side_effect=AssertionError("Strength reselection")):
            result = spec.analyze_cell(spec.Path(tmp), spec.KEYS[0], {spec.KEYS[0][0]: []}, {})
        self.assertEqual(len(accesses), 128);self.assertEqual(result["model_state_exposures"], 2640)
        self.assertLess(result["outcome_error"], 1e-12)
        before = {"groups": {}}
        for g in ("mixed_pooled", *spec.VIEWS):
            keys = [k for k in spec.KEYS if k[0].startswith("mixed")] if g == "mixed_pooled" else [k for k in spec.KEYS if k[0] == g]
            before["groups"][g] = {}
            for b, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
                before["groups"][g][b] = {}
                for scope in spec.SCOPES:
                    rs = [r for _ in keys for r in previous if r["scope"] == scope and (bank is None or r["bank"] == bank)]
                    before["groups"][g][b][scope] = {"outcomes": {h: {a: learning.policy.summarize([r["outcomes"][h][a] for r in rs]) for a in learning.policy.ARMS} for h in learning.policy.DRAW_GROUPS}}
        report = spec.report([{**result, "key": list(k)} for k in spec.KEYS], before)
        self.assertTrue(report["coverage_qualified"])
        self.assertLess(report["reproduction"]["group_outcome_error"], 1e-12)
        compact = spec.compact_report(report)
        self.assertEqual(set(compact["groups"]["mixed_pooled"]["pooled"]["held_opposite"]), {"all16", "first8", "last8"})
        self.assertEqual(set(compact["groups"]["root_397"]["pooled"]["held_opposite"]), {"all16"})
        json.dumps(compact, allow_nan=False)

    def test_scheduler_unpinned_and_parent_exit_required(self):
        task = submit.task_specification("unit_margin")
        self.assertEqual(task["cpu"], 1);self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertTrue(task["result_dir"].endswith("/report"))
        parent = {"protocol": spec.source.EXPERIMENT_PROTOCOL, "contract": spec.source.contract(), "coverage_qualified": True}
        receipt = {"scheduler": {"status_counts": {"done": 1}, "exit_code_counts": {"0": 1}}}
        with patch("sys.argv", ["submit", "--run-name", "unit_margin"]), patch.object(submit, "inventory", return_value=[]), patch.object(submit.Path, "exists", return_value=False), \
                patch.object(submit.Path, "read_text", autospec=True, side_effect=lambda p: json.dumps(receipt if p.name == "completion_receipt.json" else parent)), \
                patch.object(submit.subprocess, "check_output", return_value="revision"), patch.object(submit, "write_json"), patch.object(submit, "execute_bulk") as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]), 1)
            bulk.reset_mock();receipt["scheduler"]["exit_code_counts"] = {"1": 1}
            with self.assertRaises(SystemExit): submit.main()
            bulk.assert_not_called()


if __name__ == "__main__": unittest.main()
