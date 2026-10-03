import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_fresh_lower as experiment
from scripts import pointmaze_fresh_lower_stage99_spec as spec
from scripts.submit_pointmaze_fresh_lower_stage99_scheduleurm import task_specification, qualification_task
import test_pointmaze_call_weighted as call_fixture
import test_pointmaze_feasible_credit as fixture
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool
from test_pointmaze_upper_paths import predictor


class FreshLowerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_same_methods_budget_statistics_but_fresh_disjoint_rosters_and_sources(self):
        seen = set()
        for preflight in (True, False):
            self.assertEqual(spec.options(preflight=preflight), spec.reference.options(preflight=preflight))
            self.assertEqual(spec.budget(preflight=preflight), spec.reference.budget(preflight=preflight))
            for root in spec.roots(preflight=preflight):
                seeds = call_fixture.seeds(spec.seed_roles(root, preflight=preflight))
                self.assertEqual(len(seeds), len(set(seeds)))
                self.assertFalse(set(seeds) & seen)
                seen.update(seeds)
                prior = call_fixture.seeds(spec.source.seed_roles(root, preflight=preflight))
                prior += sum(spec.source.decoder.source.seed_roles(root, preflight=False).values(), [])
                prior += spec.source.decoder.seed_roles(root, preflight=False)["native_probe"]
                self.assertFalse(set(seeds) & set(prior))
                task = task_specification("unit_stage99", root, preflight=preflight)
                self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertFalse(task.get("require_node"))
                self.assertEqual((task["cpu"], task["ram_mb"]), (3, 3072) if preflight else (9, 8192))
                self.assertTrue(task["result_dir"].endswith("/completion"))
            self.assertIsNone(qualification_task("unit_stage99", preflight=preflight)["result_dir"])
        self.assertEqual((spec.ENDPOINTS, spec.BOOTSTRAP_SEED), (spec.reference.ENDPOINTS, spec.reference.BOOTSTRAP_SEED))
        self.assertEqual((spec.budget(preflight=True)["native_episodes"], spec.budget(preflight=True)["native_steps"]), (224, 67200))
        self.assertEqual((8*spec.budget(preflight=False)["native_episodes"], 8*spec.budget(preflight=False)["native_steps"]), (38912, 46694400))

    def test_real_new_joint_checkpoints_feed_four_lowers_and_exact_training_cost(self):
        root = 410011
        original = fixture.FeasibleCreditTest().source()
        clones = {str(p): copy.deepcopy(original) for p in spec.PERIODS}
        snapshots = {p: copy.deepcopy(m.state_dict()) for p, m in clones.items()}
        envelope = {"velocity_speed_q99": 1., "axis_min": [-1., -1.], "axis_max": [1., 1.]}
        calibrations = {str(p): {"alpha": .02, "envelope": envelope} for p in spec.PERIODS}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            donor_file = directory / "joint/result.json"
            donor_file.parent.mkdir()
            training = {"status": "complete", "protocol": spec.source.EXPERIMENT_PROTOCOL,
                "root": root, "preflight": False, "groups": {}}
            for p in spec.PERIODS:
                weights = experiment.learning.native.joint.inference_weights(original)
                for a in ("upper", "lower"):
                    weights[a + "_actor"]["net.0.weight"] += .001
                path = donor_file.parent / "final_weights" / f"period_{p}_joint_call.pt"
                path.parent.mkdir(exist_ok=True)
                torch.save({"protocol": spec.source.EXPERIMENT_PROTOCOL, "root": root, "period": p,
                    "method": "joint_call", "updates": 8, "weights": weights}, path)
                training["groups"][str(p)] = {"trained": {"joint_call": {"evaluation_update": 8,
                    "final_freeze_check": "passed", "checkpoint": str(path)}}}
            donor_file.write_text(json.dumps(training))
            source_record = spec.source.source_record(root)
            # Upstream full traces are omitted; actual checkpoint loading and this training run remain real.
            with patch.object(spec, "donor_result", return_value=donor_file), \
                    patch.object(experiment.fresh_joint, "qualify"), \
                    patch.object(experiment.fresh_joint, "load_source", return_value=(clones, predictor(), source_record, calibrations)), \
                    patch.object(experiment.learning, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment.learning.native.joint, "_make_task", side_effect=lambda **kw: DenseTask()), \
                    patch.object(experiment.learning.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
                cell = experiment.run(root, preflight=True, output=directory / "lower/result.json")
                self.assertEqual(cell["cost"], spec.budget(preflight=True))
                self.assertEqual(experiment.aggregate([cell], preflight=True)["conditioning_confirmation"], "mechanical_only")
                for p, g in cell["groups"].items():
                    self.assertEqual(g["training_noise_pairing"], spec.NOISE_MODES)
                    self.assertEqual(set(g["trained"]), set(spec.METHODS))
                    self.assertEqual(g["final_upper_freeze"], "passed")
                    self.assertEqual(set(g["evaluation"]), set(spec.VARIANTS))
                    for t in g["trained"].values(): self.assertIsNone(t["checkpoint"])
                    for rows in g["evaluation"].values():
                        for r in rows:
                            self.assertEqual(r["upper_replay_forward_calls"], 0)
                            self.assertNotIn("upper_noise_seed", r)
                    experiment.learning.native.curves.support.assert_frozen(clones[p], snapshots[p])
                bad = copy.deepcopy(cell)
                bad["source_initialization"]["decoder_result"] = "old_decoder/result.json"
                with self.assertRaisesRegex(ValueError, "source record"): experiment.qualify(bad, preflight=True)
                bad = copy.deepcopy(cell)
                bad["groups"]["50"]["evaluation"]["base"][0]["upper_noise_seed"] = 1
                with self.assertRaises(ValueError): experiment.qualify(bad, preflight=True)
                payload_path = Path(training["groups"]["50"]["trained"]["joint_call"]["checkpoint"])
                payload = torch.load(payload_path, weights_only=False)
                payload["protocol"] = spec.reference.source.source.teacher_source.EXPERIMENT_PROTOCOL
                torch.save(payload, payload_path)
                models = {m: copy.deepcopy(original) for m in spec.METHODS}
                with self.assertRaisesRegex(ValueError, "registered final checkpoint"):
                    experiment.previous.prepare_training(training, root, 50, models,
                        dict.fromkeys(spec.budget(preflight=True), 0), protocol=spec, checkpoint_protocol=spec)

    def test_shared_26_contrast_family_missing_roots_and_negative_conditioning_kept(self):
        cells = [{"root": r, "groups": {"both": {"effects": dict.fromkeys(spec.ENDPOINTS, 2.)}},
            "cost": spec.budget(preflight=False)} for r in spec.roots(preflight=False)]
        with patch.object(experiment, "qualify", side_effect=lambda c, **kw: c), \
                patch.object(spec, "BOOTSTRAP_DRAWS", 128), \
                patch.object(experiment.learning.native.np, "quantile", wraps=np.quantile) as quantile:
            result = experiment.aggregate(cells, preflight=False)
            self.assertEqual(quantile.call_args.args[1], [.05/52, 1-.05/52])
            self.assertEqual(result["conditioning_confirmation"], "supported")
            for c in cells: c["groups"]["both"]["effects"][spec.PRIMARY_ENDPOINTS[-1]] = -1.
            self.assertEqual(experiment.aggregate(cells, preflight=False)["conditioning_confirmation"], "not_supported")
            with self.assertRaises(ValueError): experiment.aggregate(cells[:-1], preflight=False)


if __name__ == "__main__":
    unittest.main()
