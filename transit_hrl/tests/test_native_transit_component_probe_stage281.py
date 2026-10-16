import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_component_probe_stage281 as spec
from scripts import submit_native_transit_component_probe_stage281_scheduleurm as submit

learning = spec.learning


class ComponentProbeTest(unittest.TestCase):
    def fixture(self, n=1):
        service = np.tile([0., .02, .01, .05, .05, .05], (n, 1))
        risk = np.tile([0., 0., -.005, 0., 0., 0.], (n, 1))
        logits = -np.ones((n, 6))
        logits[:, 5] = 1.
        data = {"originals": np.zeros(n, int), "components": np.stack((service, risk), axis=-1),
            "heads": np.stack((service+risk, np.zeros_like(service)), axis=-1),
            "raw_logits": logits, "raw_choices": np.zeros(n, int)}
        risks = {a: risk.copy() for a in learning.policy.ARMS}
        risks["raw_linear_clip"][:, 2] = -.03
        risks["observed_context_df_clip"][:, 1] = -.04
        values = np.zeros((n, 32, 6, 4))
        values[:, :, 1, 1] = .02
        values[:, :, 2, 1] = -.01
        values[:, :, 1, 2] = .004
        values[:, :, 2, 2] = -.002
        values[:, :, 1, 3] = .03
        values[:, :, 2, 3] = -.04
        values[:, :, 5, 2] = -.4
        values[..., 1:3] += .3
        values[..., 0] = values[..., 1]+values[..., 2]
        return data, risks, values

    def test_components_controls_feasibility_and_original_anchors(self):
        data, risks, values = self.fixture()
        choices, parts, error = learning.propose(data, risks, values[:, :16])
        controls, _ = learning.policy.decisions(data, risks)
        self.assertEqual(tuple(choices), learning.ARMS)
        self.assertEqual(len(choices), 18)
        for a in learning.CONTROLS:
            np.testing.assert_array_equal(choices[a], controls[a])
        self.assertLess(error, 1e-12)
        self.assertEqual(choices["all16__mc_service_raw"][0], 2)
        self.assertEqual(choices["all16__mc_both"][0], 2)
        self.assertTrue(all(c[0] != 5 for c in choices.values()))
        for v in parts.values():
            self.assertEqual(v[0, 0], 0.)
        np.testing.assert_allclose(parts["service_mc_all16"][0, :3], [0., .004, -.002], atol=1e-15)
        np.testing.assert_allclose(parts["risk_mc_all16"][0, :3], [0., .02, -.01], atol=1e-15)

    def test_selection_halves_do_not_use_evaluation_and_preserve_ties(self):
        data, risks, values = self.fixture()
        values[:, :8, 1, 2] = .3-.04
        values[..., 0] = values[..., 1]+values[..., 2]
        chosen, _, _ = learning.propose(data, risks, values[:, :16])
        self.assertEqual(chosen["first8__mc_service_raw"][0], 1)
        self.assertEqual(chosen["last8__mc_service_raw"][0], 2)
        stability = learning.summarize_stability([learning.selection_stability(chosen)])
        self.assertEqual(stability["mc_service_raw"]["first8_vs_last8"], 1.)
        values[:, 16:] *= -100
        replay, _, _ = learning.propose(data, risks, values[:, :16])
        for a in learning.ARMS:
            np.testing.assert_array_equal(chosen[a], replay[a])
        zero = np.zeros_like(values[:, :16])
        tied, _, _ = learning.propose(data, risks, zero)
        self.assertEqual(tied["all16__mc_both"][0], 0)

    def test_physical_halves_error_denominator_and_no_eligible_actions(self):
        data, risks, values = self.fixture(2)
        data["raw_logits"][1] = 1.
        data["raw_choices"][1] = 0
        choices, parts, _ = learning.propose(data, risks, values[:, :16])
        result = learning.policy.outcomes(values[:, 16:], choices)
        for a in learning.ARMS:
            mean = learning.policy.summarize([result["all16"][a]])
            self.assertAlmostEqual(mean["delta_cost"], mean["delta_fleet"]+mean["delta_service"])
            for f in learning.policy.FIELDS:
                halves = [learning.policy.summarize([result[h][a]])[f] for h in ("first8", "last8")]
                self.assertAlmostEqual(mean[f], sum(halves)/2)
        errors = learning.component_errors(data, parts, values[:, 16:])
        self.assertEqual(errors["cases"], 1)
        self.assertAlmostEqual(errors["sums"]["service_mc_all16"], 0.)
        data["raw_logits"][:] = 1.
        errors = learning.component_errors(data, parts, values[:, 16:])
        self.assertEqual(learning.summarize_errors([errors])["MSE"]["service_raw"], None)

    def test_complete_cached_replay_proposals_before_opposite_access(self):
        data, risks, values = self.fixture(60)
        rows = [{"root": 397, "scene_seed": i//6, "probe_index": i%6} for i in range(60)]
        held = [{**r, "scene_seed": r["scene_seed"]+10} for r in rows]
        choices, _ = learning.policy.decisions(data, risks)
        folds = [[("train", list(range(60)), "train"), ("held", list(range(60)), "held")]]
        folds += [[("retained", list(ix), "train"), ("excluded", list(val), "train"), ("matched_held", list(range(60)), "held")]
            for _, ix, val in spec.cache.learning.joint.binding.policy.prefix_folds(rows)]
        queries, previous, saved = [], [], {}
        for bank in (0, 1):
            for scopes in folds:
                i = len({q["index"] for q in queries})
                for scope, ids, label in scopes:
                    ch = {a: v[ids] for a, v in choices.items()}
                    queries.append({"index": i, "bank": bank, "scope": scope, "label": label,
                        "ids": ids, "choices": {a: v.tolist() for a, v in ch.items()}})
                    saved.update({f"{i}_{scope}_{a}": v[ids] for a, v in risks.items()})
                    for future, b in (("same", bank), ("opposite", 1-bank)):
                        previous.append({"bank": bank, "scope": scope+"_"+future,
                            "outcomes": learning.policy.outcomes(values[ids, b*16:(b+1)*16], ch)})
        class Archive(dict):
            def __enter__(self): return self
            def __exit__(self, *args): return False
        proposals, accesses = [], []
        def read(path):
            return json.dumps(queries if path.name == "actions.json" else previous)
        def write(path, payload):
            if path.name.startswith("proposal_"):
                proposals.append(payload)
        def draws(rs, panels, ds):
            qi = len(accesses)//2
            q = queries[qi]
            expected, _ = spec.cache.sampling.learning.draw_assignment(q["bank"] if len(accesses)%2 == 0 else 1-q["bank"], "expanded")
            self.assertEqual(ds, expected)
            if len(accesses)%2:
                self.assertEqual(len(proposals), qi+1)
            accesses.append(True)
            return values[[r["scene_seed"]%10*6+r["probe_index"] for r in rs]][:, ds]
        with tempfile.TemporaryDirectory() as tmp, patch.object(spec.Path, "read_text", autospec=True, side_effect=read), \
                patch.object(spec.np, "load", return_value=Archive(saved)), patch.object(spec.source, "read_cell", return_value={"reference": {}}), \
                patch.object(spec, "write_json", side_effect=write), patch.object(spec.cache.cached.training.cohort, "split_rows", return_value=(rows, held)), \
                patch.object(spec.cache.learning.base, "causal_data", return_value=data), patch.object(spec.cache.learning.base, "training_draws", side_effect=draws), \
                patch.object(spec.source.source.source.learning, "fit", side_effect=AssertionError("Risk refit")), \
                patch.object(spec.cache.learning, "fit_inner", side_effect=AssertionError("Prior refit")):
            result = spec.analyze_cell(spec.Path(tmp), spec.KEYS[0], {spec.KEYS[0][0]: []}, {})
        self.assertEqual(len(accesses), 128)
        self.assertEqual(result["model_state_exposures"], 2640)
        self.assertLess(result["outcome_error"], 1e-12)
        before = {"groups": {}}
        for g in ("mixed_pooled", *spec.VIEWS):
            keys = [k for k in spec.KEYS if k[0].startswith("mixed")] if g == "mixed_pooled" else [k for k in spec.KEYS if k[0] == g]
            before["groups"][g] = {}
            for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
                before["groups"][g][label] = {}
                for scope in spec.SCOPES:
                    rs = [r for _ in keys for r in previous if r["scope"] == scope and (bank is None or r["bank"] == bank)]
                    before["groups"][g][label][scope] = {"outcomes": {h: {a: learning.policy.summarize([r["outcomes"][h][a] for r in rs])
                        for a in learning.CONTROLS} for h in learning.policy.DRAW_GROUPS}}
        report = spec.report([{**result, "key": list(k)} for k in spec.KEYS], before)
        self.assertLess(report["reproduction"]["group_outcome_error"], 1e-12)
        self.assertEqual(set(report["groups"]["mixed_pooled"]["pooled"]["held_opposite"]["outcomes"]), {"all16", "first8", "last8"})
        json.dumps(report, allow_nan=False)

    def test_parent_completion_required_and_scheduler_unpinned(self):
        task = submit.task_specification("unit_component")
        self.assertEqual(task["cpu"], 1)
        self.assertEqual(task["ram_mb"], 3072)
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        parent = {"protocol": spec.source.EXPERIMENT_PROTOCOL, "contract": spec.source.contract(), "coverage_qualified": True}
        receipt = {"scheduler": {"status_counts": {"done": 1}, "exit_code_counts": {"0": 1}}}
        with patch("sys.argv", ["submit", "--run-name", "unit_component"]), patch.object(submit, "inventory", return_value=[]), \
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
