import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_prefix_coverage_stage288 as spec
from scripts import submit_native_transit_prefix_coverage_stage288_scheduleurm as submit

learning = spec.learning


class PrefixCoverageTest(unittest.TestCase):
    def test_matched_budgets_and_rotations_use_identical_total_information(self):
        rows = [{"root": k[1], "scenario": k[2], "replica": k[3], "probe_index": p}
            for k in spec.keys("train") for p in spec.PROBES]
        memberships = {a: [] for a in ("dense", "wide")}
        for rotation in range(6):
            for arm in memberships:
                ix = learning.assignment(rows, arm, rotation)
                selected = [rows[i] for i in ix]
                self.assertEqual(len(ix), 60)
                self.assertEqual(len({(r["root"], r["scenario"], r["replica"]) for r in selected}), 10 if arm == "dense" else 60)
                self.assertEqual([sum(r["probe_index"] == p for r in selected) for p in spec.PROBES], [10]*6)
                self.assertEqual([sum(r["scenario"] == s for r in selected) for s in spec.SCENARIOS], [12]*5)
                memberships[arm].extend(ix)
        for indices in memberships.values():
            self.assertEqual(sorted(indices), list(range(360)))
        self.assertEqual(len(submit.tasks("unit", "collect")), 3120)
        self.assertEqual(sum(24 if k[0] == "train" else 16 for k in spec.keys())*6*7, 87360)

    def test_new_seed_namespaces_pilot_and_futures_disjoint(self):
        all_keys = spec.keys()+spec.keys("pilot")
        scenes = [spec.scene_seed(k) for k in all_keys]
        self.assertEqual(len(set(scenes)), 110)
        old = {spec.native.scene_seed(*k) for k in spec.native.prefixes()+spec.native.prefixes(pilot=True)}
        self.assertFalse(set(scenes)&old)
        futures = [spec.future_seed(k, p, d) for k in all_keys for p in spec.PROBES for d in range(24)]
        self.assertEqual(len(futures), len(set(futures)))
        self.assertEqual({spec.pilot_probe(k) for k in spec.keys("pilot")}, set(spec.PROBES))

    def test_frozen_zero_profiles_skip_fit_and_pair_noise_is_collapsed(self):
        old = {"scale": 1., "weights": {"zero": np.zeros((2, 198)).tolist()}}
        with patch.object(learning.base.optimizer, "quadratic_fit", side_effect=AssertionError("Zero must not fit")):
            model, ledger = learning.fit([], {}, None, {"model": old, "selection": None})
        self.assertEqual(model, old)
        self.assertFalse(ledger["active"])
        choices = np.zeros((2, 6, 4, 3), int)
        choices[:, :, 2] = 1
        choices[:, :, 3] = 2
        draws = np.zeros((3, 4, 6, 4))
        draws[:, :, 1] = np.arange(4)[None, :, None]+1
        draws[:, :, 2] = np.arange(4)[None, :, None]+2
        effects = learning.paired_controls(choices, draws)
        np.testing.assert_allclose(effects[:, :, 3]-effects[:, :, 2], 1.)

    def test_fit_uses_frozen_basis_and_preserves_head_difference(self):
        rng = np.random.default_rng(288)
        rows = [{"root": 397, "scene_seed": i//6} for i in range(60)]
        x = np.zeros((60, 6, 198))
        x[..., :3] = rng.normal(size=(60, 6, 3))
        x[:, 0] = 0.
        data = {"x": x, "originals": np.zeros(60, int), "raw_logits": -np.ones((60, 6)),
            "components": np.zeros((60, 6, 2))}
        basis = np.eye(198, 3)
        frozen = {"selection": "3@0.5", "level": "3", "basis": basis.tolist(),
            "difference": [.1, -.2, .1], "initial": [0., 0., 0.], "radius": .2,
            "model": {"scale": 1., "weights": {"zero": np.zeros((2, 198)).tolist(), "3": np.zeros((2, 198)).tolist()}}}
        draws = rng.normal(size=(60, 16, 6, 4))
        draws[..., 0] = draws[..., 1]+draws[..., 2]
        before = copy.deepcopy(frozen)
        with patch.object(np.linalg, "svd", side_effect=AssertionError("No refit of representation")):
            model, ledger = learning.fit(rows, data, draws, frozen)
        weights = np.asarray(model["weights"]["3"])
        np.testing.assert_allclose(weights[0]-weights[1], basis@np.asarray(frozen["difference"]))
        self.assertLessEqual(np.linalg.norm(weights.sum(axis=0)), frozen["radius"]+1e-10)
        self.assertLessEqual(ledger["optimality"], 1e-8)
        self.assertEqual(frozen, before)

    def test_error_pairing_and_prefix_not_model_unit(self):
        data = {"raw_logits": -np.ones((1, 6))}
        draws = np.arange(4.)[None, :, None, None]*np.arange(6.)[None, None, :, None]*np.ones((1, 4, 6, 4))
        prediction = {"cost": np.zeros((1, 6)), "components": np.zeros((1, 6, 2))}
        error, noise = learning.margin_errors(data, draws, prediction)
        self.assertTrue(np.all(error > noise))
        records = []
        for i in range(4):
            mean = np.zeros((4, 4))
            mean[3] = -i-1
            records.append({"root": 397, "scenario": "low_noise", "mean": mean})
        result = learning.paired_prefix_summary(records, bootstrap=100)
        self.assertEqual(result["wide_vs_dense"]["mean"]["cost"], -2.5)
        self.assertLess(result["wide_vs_dense"]["prefix_CI95"]["cost"][1], 0.)

    def test_fit_does_not_open_held_labels_and_scheduler_has_no_pin(self):
        def panels(run, role):
            if role != "train":
                raise AssertionError("Fit opened held labels")
            raise RuntimeError("training-only sentinel")
        with patch.object(spec, "load_rows", return_value=[]), patch.object(spec, "load_panels", side_effect=panels):
            with self.assertRaisesRegex(RuntimeError, "training-only sentinel"):
                spec.fit("unit")
        task = submit.task_specification("unit", "pilot-collect", spec.keys("pilot")[0])
        self.assertEqual(task["cpu"], 1)
        self.assertEqual(task["ram_mb"], 1536)
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])

    def test_toy_fit_report_roundtrip_and_held_blinding(self):
        rng = np.random.default_rng(288)
        rows = {role: [{"root": k[1], "scenario": k[2], "replica": k[3], "probe_index": p,
            "scene_seed": spec.scene_seed(k), "original_choice": 0} for k in spec.keys(role) for p in spec.PROBES]
            for role in ("train", "held")}
        features = {role: rng.normal(size=(len(rr), 6, 3)) for role, rr in rows.items()}
        panels = {role: rng.normal(size=(len(rr), 24 if role == "train" else 16, 6, 4)) for role, rr in rows.items()}
        for role in panels:
            panels[role][..., 0] = panels[role][..., 1]+panels[role][..., 2]
            features[role][:, 0] = 0.

        def data(rr, reference):
            role = "train" if len(rr) == 360 else "held"
            x = np.zeros((len(rr), 6, 198))
            x[..., :3] = features[role]
            return {"x": x, "components": np.zeros((len(rr), 6, 2)), "heads": np.zeros((len(rr), 6, 2)),
                "originals": np.zeros(len(rr), int), "raw_logits": -np.ones((len(rr), 6)), "raw_choices": np.zeros(len(rr), int)}

        frozen = {"reference": {}, "selection": "3@0.5", "level": "3", "basis": np.eye(198, 3).tolist(),
            "difference": [0., 0., 0.], "initial": [0., 0., 0.], "radius": .2,
            "model": {"scale": 1., "weights": {k: np.zeros((2, 198)).tolist() for k in ("zero", "1", "3", "8", "16")}}}
        with tempfile.TemporaryDirectory() as temp, patch.object(spec, "ROOT", Path(temp)):
            root = spec.directory("unit")
            spec.write_json(root/"freeze_raw/models.json", [frozen, {**frozen, "selection": None}])

            def load(run, role):
                if role == "held":
                    self.assertTrue(spec.read(root/"fit/result.json")["models_and_choices_frozen"])
                return panels[role], [{"native_episodes": 7, "peak_rss_MiB": 700.}]

            with patch.object(spec, "load_rows", side_effect=lambda run, role: rows[role]), \
                    patch.object(spec, "load_panels", side_effect=load), \
                    patch.object(learning.base.base, "causal_data", side_effect=data):
                spec.fit("unit")
                self.assertFalse(spec.read(root/"fit/result.json")["held_labels_read"])
                spec.report("unit")
            report = spec.read(root/"report/result.json")
            self.assertEqual(report["held_prefixes"], 40)
            self.assertIn("wide_vs_dense", report["controls"])
            self.assertIn("independent8_minus_MC_variance", report["training_errors"]["wide"])
            self.assertFalse(report["controller_promoted"])

    def test_scheduler_gate_excludes_earlier_phases_but_requires_all_captures(self):
        records = [{"id": f"t{i}", "status": "done", "exit_code": 0, **t}
            for i, t in enumerate(submit.tasks("unit", "freeze")+submit.tasks("unit", "capture-pilot"))]
        fake_scheduler = SimpleNamespace(load_state=lambda: {"tasks": records})
        with patch.dict(sys.modules, {"scheduler": fake_scheduler}), \
                patch.object(sys, "argv", ["submit", "--run-name", "unit", "--phase", "pilot-collect"]), \
                patch.object(submit, "inventory", return_value=records), \
                patch.object(spec, "read", return_value={"contract": spec.contract()}), \
                patch.object(submit, "execute_bulk") as execute:
            submit.main()
            self.assertEqual(len(execute.call_args.args[0]), 10)
            records[-1]["status"] = "running"
            with self.assertRaises(SystemExit):
                submit.main()


if __name__ == "__main__":
    unittest.main()
