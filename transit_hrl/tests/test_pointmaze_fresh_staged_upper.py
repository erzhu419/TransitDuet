import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_fresh_staged_upper as experiment
from scripts import pointmaze_fresh_staged_upper_stage100_spec as spec
from scripts.submit_pointmaze_fresh_staged_upper_stage100_scheduleurm import task_specification, qualification_task
import test_pointmaze_call_weighted as call_fixture
import test_pointmaze_feasible_credit as fixture
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool
from test_pointmaze_upper_paths import predictor


class FreshStagedUpperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_unchanged_rule_budget_family_and_fresh_roles_dynamic_nodes(self):
        seen = set()
        for preflight in (True, False):
            self.assertEqual(spec.options(preflight=preflight), spec.reference.options(preflight=preflight))
            self.assertEqual(spec.budget(preflight=preflight), spec.reference.budget(preflight=preflight))
            for root in spec.roots(preflight=preflight):
                seeds = call_fixture.seeds(spec.seed_roles(root, preflight=preflight))
                self.assertEqual(len(seeds), len(set(seeds)))
                self.assertFalse(set(seeds) & seen)
                seen.update(seeds)
                joint, decoder = spec.source.source, spec.source.source.decoder
                prior = call_fixture.seeds(spec.source.seed_roles(root, preflight=preflight))
                prior += call_fixture.seeds(joint.seed_roles(root, preflight=preflight))
                prior += sum(decoder.source.seed_roles(root, preflight=False).values(), [])
                prior += decoder.seed_roles(root, preflight=False)["native_probe"]
                self.assertFalse(set(seeds) & set(prior))
                task = task_specification("unit_stage100", root, preflight=preflight)
                self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertFalse(task.get("require_node"))
                self.assertEqual((task["cpu"], task["ram_mb"]), (3, 3072) if preflight else (9, 8192))
                self.assertTrue(task["result_dir"].endswith("/completion"))
            self.assertIsNone(qualification_task("unit_stage100", preflight=preflight)["result_dir"])
        self.assertEqual((spec.ENDPOINTS, spec.PRIMARY_ENDPOINTS, spec.BOOTSTRAP_SEED),
            (spec.reference.ENDPOINTS, spec.reference.PRIMARY_ENDPOINTS, spec.reference.BOOTSTRAP_SEED))
        self.assertEqual((spec.budget(preflight=True)["native_episodes"], spec.budget(preflight=True)["native_steps"]), (152, 45600))
        self.assertEqual((8*spec.budget(preflight=False)["native_episodes"], 8*spec.budget(preflight=False)["native_steps"]), (22016, 26419200))

    def test_real_new_donors_train_only_upper_and_preserve_all_compositions(self):
        root = 410011
        original = fixture.FeasibleCreditTest().source()
        clones = {str(p): copy.deepcopy(original) for p in spec.PERIODS}
        snapshots = {p: copy.deepcopy(m.state_dict()) for p, m in clones.items()}
        envelope = {"velocity_speed_q99": 1., "axis_min": [-1., -1.], "axis_max": [1., 1.]}
        calibrations = {str(p): {"alpha": .02, "envelope": envelope} for p in spec.PERIODS}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            files = {"lower": directory / "lower/result.json", "joint": directory / "joint/result.json"}
            records = {name: {"root": root, "groups": {}} for name in files}
            for name, path in files.items(): path.parent.mkdir()
            for p in spec.PERIODS:
                for name in records: records[name]["groups"][str(p)] = {"trained": {}}
                for i, method in enumerate(spec.CHECKPOINT_METHODS, 1):
                    name = "joint" if method == "joint_call" else "lower"
                    weights = experiment.learning.native.joint.inference_weights(original)
                    weights["lower_actor"]["net.0.weight"] += .001*i
                    if method == "joint_call": weights["upper_actor"]["net.0.weight"] += .001
                    path = files[name].parent / "final_weights" / f"period_{p}_{method}.pt"
                    path.parent.mkdir(exist_ok=True)
                    protocol = spec.source.source if name == "joint" else spec.source
                    torch.save({"protocol": protocol.EXPERIMENT_PROTOCOL, "root": root, "period": p,
                        "method": method, "updates": 8, "weights": weights}, path)
                    records[name]["groups"][str(p)]["trained"][method] = {
                        "evaluation_update": 8, "final_freeze_check": "passed", "checkpoint": str(path)}
            for name, record in records.items(): files[name].write_text(json.dumps(record))
            # Only upstream full traces are omitted; checkpoint loads, updates and evaluation remain real.
            with patch.object(spec, "donor_result", side_effect=lambda r, m: files["joint" if m == "joint_call" else "lower"]), \
                    patch.object(experiment.fresh_joint, "qualify"), patch.object(experiment.fresh_lower, "qualify"), \
                    patch.object(experiment.fresh_joint, "load_source", return_value=(clones, predictor(), spec.source_record(root), calibrations)), \
                    patch.object(experiment.learning, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment.learning.native.joint, "_make_task", side_effect=lambda **kw: DenseTask()), \
                    patch.object(experiment.learning.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
                cell = experiment.run(root, preflight=True, output=directory / "upper/result.json")
                self.assertEqual(cell["cost"], spec.budget(preflight=True))
                self.assertEqual(experiment.aggregate([cell], preflight=True)["staged_confirmation"], "mechanical_only")
                for p, g in cell["groups"].items():
                    self.assertEqual(g["lower_for_method"], spec.LOWER_FOR_METHOD)
                    self.assertEqual(g["final_lower_freeze"], "passed")
                    self.assertEqual(set(g["evaluation"]), set(spec.VARIANTS))
                    for t in g["trained"].values():
                        self.assertIsNone(t["checkpoint"])
                        self.assertTrue(all(set(r["actors"]) == {"upper"} for r in t["history"]))
                    for rows in g["evaluation"].values():
                        for r in rows:
                            self.assertEqual(r["upper_replay_forward_calls"], 0)
                            self.assertNotIn("upper_noise_seed", r)
                    experiment.learning.native.curves.support.assert_frozen(clones[p], snapshots[p])
                bad = copy.deepcopy(cell)
                bad["source_initialization"]["teacher_result"] = "old_teacher/result.json"
                with self.assertRaisesRegex(ValueError, "source record"): experiment.qualify(bad, preflight=True)
                bad = copy.deepcopy(cell)
                bad["groups"]["50"]["evaluation"]["base"][0]["upper_noise_seed"] = 1
                with self.assertRaises(ValueError): experiment.qualify(bad, preflight=True)
                training = {m: records["joint" if m == "joint_call" else "lower"] for m in spec.CHECKPOINT_METHODS}
                cost = dict.fromkeys(spec.budget(preflight=True), 0)
                models = {m: copy.deepcopy(original) for m in spec.METHODS}
                path = Path(training["joint_call"]["groups"]["50"]["trained"]["joint_call"]["checkpoint"])
                payload = torch.load(path, weights_only=False)
                payload["protocol"] = spec.source.source.reference.EXPERIMENT_PROTOCOL
                torch.save(payload, path)
                with self.assertRaisesRegex(ValueError, "registered final checkpoint"):
                    experiment.previous.prepare_training(training, root, 50, models, cost,
                        protocol=spec, joint_checkpoint_protocol=spec.source)
                records["lower"]["root"] = 410023
                files["lower"].write_text(json.dumps(records["lower"]))
                with self.assertRaisesRegex(ValueError, "donor root"):
                    experiment.run(root, preflight=True, output=directory / "bad/result.json")

    def test_same_28_contrast_family_and_all_four_gate_retain_failure(self):
        cells = [{"root": r, "groups": {"both": {"effects": dict.fromkeys(spec.ENDPOINTS, 2.)}},
            "cost": spec.budget(preflight=False)} for r in spec.roots(preflight=False)]
        with patch.object(experiment, "qualify", side_effect=lambda c, **kw: c), \
                patch.object(spec, "BOOTSTRAP_DRAWS", 128), \
                patch.object(experiment.learning.native.np, "quantile", wraps=np.quantile) as quantile:
            result = experiment.aggregate(cells, preflight=False)
            self.assertEqual(quantile.call_args.args[1], [.05/56, 1-.05/56])
            self.assertEqual(result["staged_confirmation"], "supported")
            for c in cells: c["groups"]["both"]["effects"][spec.PRIMARY_ENDPOINTS[-1]] = 0.
            self.assertEqual(experiment.aggregate(cells, preflight=False)["staged_confirmation"], "not_supported")
            with self.assertRaises(ValueError): experiment.aggregate(cells[:-1], preflight=False)


if __name__ == "__main__":
    unittest.main()
