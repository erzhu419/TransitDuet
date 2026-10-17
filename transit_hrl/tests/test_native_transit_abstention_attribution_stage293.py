from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_abstention_attribution_stage293 as spec
from scripts import submit_native_transit_abstention_attribution_stage293_scheduleurm as submit


def fixture(draws=64):
    cube = np.arange(3).reshape(1, 3, 1)
    raw, proposal = np.zeros((1, 3), int), np.ones((1, 3), int)
    keep = np.array([[True, False, False]])
    choices = np.stack((raw, raw, np.where(keep, proposal, raw), raw), axis=1)
    sealed = {"choices": choices.tolist(), "proposal": proposal.tolist(), "matched_thin_probabilities": [[1/3]*3],
        "states": [{"key": ["train", 397, "low_noise", i], "probe": 6} for i in range(3)]}
    panels = np.zeros((3, draws, 6, 4))
    panels[:, :, 1, 0] = np.array([-2., 2., 2.])[:, None]+(np.arange(draws)-(draws-1)/2)[None]/draws
    panels[:, :, 1, 1] = panels[:, :, 1, 0]+1
    panels[:, :, 1, 2] = -1
    panels[:, :, 1, 3] = -.5
    rows = [{"root": 397, "scenario": "low_noise", "replica": i, "probe_index": 6} for i in range(3)]
    return sealed, panels, cube, np.ones((1, 3), bool), keep, rows


class AbstentionAttributionTest(unittest.TestCase):
    def test_same_resamples_isolate_fixed_and_rematched_control(self):
        sealed, panels, cube, eligible, keep, _ = fixture()
        result, _ = spec.bridge(sealed, panels, cube, eligible, keep)
        ix = np.random.default_rng(292).choice(3, (5000, 3), replace=True)
        values = panels[:, :, 1, 0].mean(axis=1)
        witness = ((ix == 0)*values[ix]).mean(axis=1)
        fixed = values[ix].mean(axis=1)/3
        rematched = values[ix].mean(axis=1)*(ix == 0).mean(axis=1)
        contrast = result["zero_abstain_vs_matched_thin"]
        np.testing.assert_allclose(contrast["fixed_prefix_CI95"]["cost"], np.quantile(witness-fixed, [.025, .975]))
        np.testing.assert_allclose(contrast["rematched_prefix_CI95"]["cost"], np.quantile(witness-rematched, [.025, .975]))
        raw = result["zero_abstain_vs_raw"]
        self.assertEqual(raw["fixed_prefix_CI95"], raw["rematched_prefix_CI95"])

    def test_shared_profile_noise_is_not_replicated(self):
        sealed, panels, cube, eligible, keep, rows = fixture()
        result, actual = spec.bridge(sealed, panels, cube, eligible, keep)
        attributed, _ = spec.attribution(actual, cube, rows)
        for name in result:
            a, b = (spec.ARMS.index(s) for s in name.split("_vs_"))
            delta = actual[:, :, a, 0]-actual[:, :, b, 0]
            expected = np.sqrt(delta.var(axis=1, ddof=1).sum()/64)/3
            self.assertAlmostEqual(attributed[name]["variance"]["cost"]["conditional_MC_SE"], expected)
        repeated = {k: np.repeat(np.asarray(v), 16, axis=0).tolist() for k, v in sealed.items() if k != "states"}
        duplicate = spec.effects(repeated, panels)
        np.testing.assert_allclose(actual, duplicate, atol=1e-15)

    def test_variance_subtraction_and_leave_one_preserve_stratum_weight(self):
        actual = np.zeros((3, 64, 4, 4))
        noise = (np.arange(64)-31.5)/64
        actual[:, :, 2, 0] = np.array([-2., 0., 2.])[:, None]+noise
        _, _, cube, _, _, rows = fixture()
        result, records = spec.attribution(actual, cube, rows)
        row = result["zero_abstain_vs_raw"]
        variance = row["variance"]["cost"]
        self.assertAlmostEqual(variance["analytic_prefix_SE_ddof1"]**2, 4/3)
        mc = noise.var(ddof=1)/64/3
        self.assertAlmostEqual(variance["conditional_MC_SE"]**2, mc)
        self.assertAlmostEqual(variance["latent_signed_variance"], 4/3-mc)
        np.testing.assert_allclose(row["leave_one_prefix_cost_range"], [-1, 1])
        self.assertEqual(row["prefix_signs"]["cost"], {"better": 1, "worse": 1, "equal": 1})
        self.assertAlmostEqual(sum(r["contribution_to_pooled_mean"]["cost"] for r in records[0]["prefixes"]), 0.)
        actual[:, :, 2, 0] = noise
        row = spec.attribution(actual, cube, rows)[0]["zero_abstain_vs_raw"]["variance"]["cost"]
        self.assertLess(row["latent_signed_variance"], 0)
        self.assertEqual(row["latent_nonnegative_SE"], 0.)

    def test_old_and_fresh_roundtrip_does_not_reselect_or_read_models(self):
        sealed, new, cube, eligible, keep, rows = fixture()
        old = fixture(8)[1]
        fresh_bridge = spec.bridge(sealed, new, cube, eligible, keep)[0]
        old_bridge = spec.bridge(sealed, old, cube, eligible, keep)[0]
        parent = {"protocol": spec.SOURCE_PROTOCOL, "contract": {"fixture": True}, "coverage_qualified": True,
            "evaluation": {"comparisons": {k: {"mean": v["mean"], "prefix_CI95": v["fixed_prefix_CI95"]} for k, v in fresh_bridge.items()}},
            "joint_retention_supported": False, "cost_information_supported": False, "fleet_information_supported": False}
        old_published = {"comparisons": {k: {"mean": v["mean"]} for k, v in old_bridge.items()}}
        raw, proposal = np.zeros((1, 3), int), np.ones((1, 3), int)
        legacy = {"choices": np.stack((raw, raw, proposal, proposal, proposal, proposal), axis=1).tolist()}
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            stack.enter_context(patch.object(spec, "ROOT", Path(temp)))
            spec.write_json(spec.directory("unit293")/"inputs/preregistration.json", {"contract": spec.contract()})
            spec.write_json(spec.directory(spec.SOURCE_RUN)/"report/result.json", parent)
            spec.write_json(spec.directory(spec.SOURCE_RUN)/"preregistration.json", {"contract": parent["contract"]})
            spec.write_json(spec.directory(spec.OLD_RUN)/"report/result.json",
                {"queries": {"primary": {"evaluation8": old_published}}})
            spec.write_json(spec.directory(spec.GATE_RUN)/"report_raw/primary.json", legacy)
            stack.enter_context(patch.object(spec, "load_confirmation", return_value=(sealed, new)))
            old_loader = stack.enter_context(patch.object(spec, "load_old", return_value=old))
            stack.enter_context(patch.object(np, "load", side_effect=AssertionError("No model files")))
            spec.analyze("unit293")
            old_loader.assert_called_once_with(sealed)
            result = spec.read(spec.directory("unit293")/"report/result.json")
            self.assertTrue(result["frozen_statistics_replayed"])
            self.assertEqual(result["primary_conclusion_unchanged"]["joint_retention_supported"], False)
            self.assertEqual((result["new_fits"], result["new_native_episodes"], result["held_prefixes_read"]), (0, 0, 0))

    def test_metadata_staging_cannot_mirror_the_output_directory(self):
        task = submit.task_specification("unit293")
        self.assertEqual((task["cpu"], task["ram_mb"], task["require_node"]), (1, 3072, None))
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertNotIn(str(spec.directory("unit293")), task["stage_input_paths"])
        self.assertIn(str(spec.directory("unit293")/"inputs"), task["stage_input_paths"])
        self.assertEqual(task["result_dir"], str(spec.directory("unit293")/"report"))

    def test_parent_requires_completed_collection_not_positive_outcomes(self):
        parent = {"protocol": spec.SOURCE_PROTOCOL, "contract": {"fixture": True}, "coverage_qualified": True}
        receipt = {"scheduler": {"collection_status_counts": {"done": 2672}, "collection_exit_code_counts": {"0": 2672},
            "report": {"status": "done", "exit_code": 0}}}
        with patch.object(spec, "read", side_effect=lambda p: receipt if p.name == "completion_receipt.json" else parent):
            spec.qualify_parent()
            receipt["scheduler"]["collection_exit_code_counts"] = {"0": 2671, "1": 1}
            with self.assertRaisesRegex(ValueError, "Complete qualified Stage292"):
                spec.qualify_parent()

    def test_old_reader_uses_only_registered_train_chunks(self):
        sealed, _, _, _, _, _ = fixture()
        with tempfile.TemporaryDirectory() as temp, patch.object(spec, "ROOT", Path(temp)):
            for state in sealed["states"]:
                path = spec.directory(spec.CACHE_RUN)/"prefixes"/Path(*map(str, state["key"]))/"continuations/probe6"
                for chunk in (4, 5):
                    spec.write_json(path/f"chunk{chunk}/labels.json", [{"draw": d,
                        "values": {str(a): [float(d+a)]*4 for a in range(6)}} for d in range(4*chunk, 4*chunk+4)])
            data = spec.load_old(sealed)
            self.assertEqual(data.shape, (3, 8, 6, 4))
            np.testing.assert_array_equal(data[0, :, 0, 0], np.arange(16, 24))
            rows, cube = spec.rows_and_cube(sealed)
            self.assertEqual(cube.shape, (1, 3, 1))
            self.assertEqual(rows[2]["replica"], 2)

    def test_cold_statistics_import_does_not_load_historical_native_chain(self):
        code = "import sys; from scripts import run_native_transit_abstention_attribution_stage293; assert 'scripts.run_native_transit_abstention_confirmation_stage292' not in sys.modules"
        subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
