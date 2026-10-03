import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_fresh_joint as experiment
from scripts import pointmaze_fresh_joint_stage98_spec as spec
from scripts.submit_pointmaze_fresh_joint_stage98_scheduleurm import task_specification, qualification_task
import test_pointmaze_call_weighted as call_fixture
import test_pointmaze_fresh_decoder as decoder_fixture
from test_pointmaze_update_isolation import ImmediatePool


class FreshJointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_only_sources_roots_and_sample_roles_change_the_registered_method(self):
        seen = set()
        for preflight in (True, False):
            self.assertEqual(spec.options(preflight=preflight), spec.reference.options(preflight=preflight))
            self.assertEqual(spec.budget(preflight=preflight), spec.reference.budget(preflight=preflight))
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                seeds = call_fixture.seeds(roles)
                self.assertEqual(len(seeds), len(set(seeds)))
                self.assertFalse(set(seeds) & seen)
                seen.update(seeds)
                old = call_fixture.seeds(spec.reference.seed_roles(310011, preflight=preflight))
                upstream = sum(spec.decoder.source.seed_roles(root, preflight=False).values(), [])
                upstream += spec.decoder.seed_roles(root, preflight=False)["native_probe"]
                self.assertFalse(set(seeds) & set(old + upstream))
                before, after = vars(spec.reference.arguments(310011, preflight=preflight)), vars(spec.arguments(root, preflight=preflight))
                for key in after.keys() - {"optimizer_seed", "preflight"}:
                    self.assertEqual(before[key], after[key], key)
                t = task_specification("unit_stage98", root, preflight=preflight)
                self.assertEqual(t["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertFalse(t.get("require_node"))
                self.assertEqual((t["cpu"], t["ram_mb"]), (3, 3072) if preflight else (9, 8192))
                self.assertTrue(t["result_dir"].endswith("/completion"))
            q = qualification_task("unit_stage98", preflight=preflight)
            self.assertIsNone(q["result_dir"])
            self.assertEqual(len(q["wait_for_files"]), len(spec.roots(preflight=preflight)))
        for p in spec.PERIODS:
            for method in spec.METHODS:
                self.assertEqual(spec.allocation(method, p), spec.reference.allocation(method, p))
        self.assertEqual((spec.ENDPOINTS, spec.BOOTSTRAP_DRAWS, spec.BOOTSTRAP_SEED),
            (spec.reference.ENDPOINTS, spec.reference.BOOTSTRAP_DRAWS, spec.reference.BOOTSTRAP_SEED))
        self.assertEqual((spec.budget(preflight=True)["native_episodes"], spec.budget(preflight=True)["native_steps"]), (136, 40800))

    def test_actual_new_source_load_shared_training_freeze_budget_and_no_old_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            args, teacher_file = decoder_fixture.FreshDecoderTest().source(directory / "teacher")
            decoder_file = directory / "decoder/replicate_410011/result.json"
            bounds = (-2 * np.ones(2), 2 * np.ones(2))

            def decoder_args(root, *, preflight):
                result = copy.copy(args)
                result.optimizer_seed, result.preflight = root, preflight
                return result

            core = experiment.learning.learning
            with patch.object(spec.decoder.source, "arguments", return_value=args), \
                    patch.object(spec.decoder, "arguments", side_effect=decoder_args), \
                    patch.object(spec.decoder, "source_result", return_value=teacher_file), \
                    patch.object(spec, "teacher_result", return_value=teacher_file), \
                    patch.object(spec, "source_result", return_value=decoder_file), \
                    patch.object(experiment.decoder.teachers, "qualify"), \
                    patch.object(experiment.decoder, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(core, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(core.native.joint, "_make_task", side_effect=lambda **kw: decoder_fixture.DenseTask()), \
                    patch.object(core.native.joint, "pointmaze_goal_bounds", return_value=bounds), \
                    patch.object(core.native.curves.support.native, "load_source", side_effect=AssertionError("old teacher loaded")) as old:
                upstream = experiment.decoder.run(410011, preflight=False, output=decoder_file)
                cell = experiment.run(410011, preflight=True, output=directory / "training/result.json")
                result = experiment.aggregate([cell], preflight=True)
                self.assertEqual(result["independent_confirmation"]["status"], "mechanical_only")
                self.assertEqual(cell["cost"], spec.budget(preflight=True))
                self.assertEqual(cell["source_initialization"], spec.source_record(410011))
                old.assert_not_called()
                for p, g in cell["groups"].items():
                    self.assertEqual(g["alpha"], upstream["groups"][p]["calibration"]["alpha"])
                    self.assertEqual(set(g["trained"]), set(spec.METHODS))
                    self.assertEqual(g["source_and_Adam_unchanged"], "passed")
                    for method, trained in g["trained"].items():
                        self.assertIsNone(trained["checkpoint"])
                        for row in trained["history"]:
                            experiment.learning.check_update(row, method=method, period=int(p), horizon=300, preflight=True, protocol=spec)
                self.assertTrue((directory / "training/completion/ready.json").is_file())
                bad = copy.deepcopy(cell)
                bad["source_initialization"]["teacher_result"] = "old_teacher/result.json"
                with self.assertRaisesRegex(ValueError, "source record"): experiment.qualify(bad, preflight=True)
                bad = copy.deepcopy(cell)
                bad["cost"]["native_steps"] -= 1
                with self.assertRaises(ValueError): experiment.qualify(bad, preflight=True)
                bad = copy.deepcopy(cell)
                for p, g in bad["groups"].items():
                    for rows in g["evaluation"].values():
                        for row in rows: row["episode_return"] = -1000000.
                    g["effects"] = core.native.paired_effects(p, g["evaluation"], bad["seed_roles"]["native_evaluation"], protocol=spec)
                self.assertEqual(experiment.aggregate([bad], preflight=True)["status"], "preflight_passed")
                payload_path = Path(json.loads(teacher_file.read_text())["checkpoints"]["clone_50"])
                payload = torch.load(payload_path, weights_only=False)
                payload["root"] = 310011
                torch.save(payload, payload_path)
                with self.assertRaisesRegex(ValueError, "fixed-final new cohort"): experiment.load_source(410011)

    def test_same_corrected_family_no_missing_root_or_zero_bound_confirmation(self):
        cells = [{"root": r, "groups": {"both": {"effects": dict.fromkeys(spec.ENDPOINTS, 2.)}},
            "cost": spec.budget(preflight=False)} for r in spec.roots(preflight=False)]
        with patch.object(experiment, "qualify", side_effect=lambda c, **kw: c), \
                patch.object(spec, "BOOTSTRAP_DRAWS", 128), \
                patch.object(experiment.learning.learning.native.np, "quantile", wraps=np.quantile) as quantile:
            result = experiment.aggregate(cells, preflight=False)
            self.assertEqual(quantile.call_args.args[1], [.05 / 40, 1 - .05 / 40])
            self.assertEqual(result["independent_confirmation"]["status"], "confirmed")
            for key in spec.PRIMARY_ENDPOINTS:
                bad = copy.deepcopy(result)
                bad["endpoints"][key]["ci"] = [0., 3.]
                self.assertEqual(spec.confirmation(bad)["status"], "not_confirmed")
            with self.assertRaises(ValueError): experiment.aggregate(cells[:-1], preflight=False)


if __name__ == "__main__":
    unittest.main()
