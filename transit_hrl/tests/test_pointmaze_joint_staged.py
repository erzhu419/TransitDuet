import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_joint_staged as experiment
from scripts import pointmaze_joint_staged_stage103_spec as spec
from scripts.submit_pointmaze_joint_staged_stage103_scheduleurm import task_specification, qualification_task
import test_pointmaze_feasible_credit as fixture
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool
from test_pointmaze_upper_paths import predictor


class JointStagedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_matched_paths_fixed_donors_fresh_evaluation_and_resources(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                self.assertFalse(roles["training_rounds"])
                self.assertFalse(set(roles["native_evaluation"]) & seen)
                seen.update(roles["native_evaluation"])
                for protocol in spec.SOURCE_PROTOCOLS.values():
                    for pref in (True, False):
                        if root not in protocol.roots(preflight=pref): continue
                        old = protocol.seed_roles(root, preflight=pref)
                        seeds = {v for r in old["training_rounds"] for rows in r.values() for s in rows
                            for v in (s["scenario_seed"], *s["noise_seeds"])} | set(old["native_evaluation"])
                        self.assertFalse(set(roles["native_evaluation"]) & seeds)
                task = task_specification("unit_stage103", root, preflight=preflight)
                self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertIsNone(task.get("require_node"))
                self.assertEqual((task["cpu"], task["ram_mb"]), (3, 3072) if preflight else (9, 8192))
                self.assertTrue(task["result_dir"].endswith("/completion"))
            self.assertIsNone(qualification_task("unit_stage103", preflight=preflight)["result_dir"])
        for p in spec.PERIODS:
            record = spec.method_path_budgets(410011, p)
            self.assertEqual(record["joint"], record["staged"])
            self.assertEqual(record["joint"]["native_episodes"], 1024)
            self.assertEqual(record["joint"]["upper_credit_paths"], 512)
            self.assertEqual(record["joint"]["common_lower_extra_upper_replay_forwards"], 6144 if p == 50 else 3072)
        real_options = spec.joint.options
        with patch.object(spec.joint, "options", side_effect=lambda **kw: {**real_options(**kw), "updates": 7}):
            with self.assertRaisesRegex(ValueError, "do not match"): spec.method_path_budgets(410011, 50)
        b = spec.budget(preflight=False)
        self.assertEqual((8*b["native_episodes"], 8*b["native_steps"]), (3072, 3686400))
        self.assertEqual((b["credit_episodes"], b["actor_mean_parameter_updates"], b["checkpoint_writes"]), (0, 0, 0))
        self.assertEqual(len(spec.ENDPOINTS), 18)
        self.assertFalse(any("101" in run for run in spec.SOURCE_RUNS.values()))

    def test_actual_multi_protocol_checkpoints_frozen_native_evaluation_and_installed_lower(self):
        root = 410011
        original = fixture.FeasibleCreditTest().source()
        clones = {str(p): copy.deepcopy(original) for p in spec.PERIODS}
        snapshots = {p: copy.deepcopy(m.state_dict()) for p, m in clones.items()}
        envelope = {"velocity_speed_q99": 1., "axis_min": [-1., -1.], "axis_max": [1., 1.]}
        calibrations = {str(p): {"alpha": .02, "envelope": envelope} for p in spec.PERIODS}
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            files = {owner: folder/owner/"result.json" for owner in spec.SOURCE_PROTOCOLS}
            records = {owner: {"root": root, "groups": {str(p): {"trained": {}} for p in spec.PERIODS}} for owner in files}
            for path in files.values(): path.parent.mkdir()
            for p in spec.PERIODS:
                donors = {}
                for index, (method, owner) in enumerate(spec.DONORS.items(), 1):
                    weights = experiment.learning.native.joint.inference_weights(original)
                    for actor in spec.SOURCE_PROTOCOLS[owner].METHODS[method]:
                        weights[actor+"_actor"]["net.0.weight"] += .001*index
                    if owner == "upper":
                        weights["lower_actor"] = copy.deepcopy(donors[spec.upper.LOWER_FOR_METHOD[method]]["lower_actor"])
                    donors[method] = weights
                    path = files[owner].parent/"final_weights"/f"period_{p}_{method}.pt"
                    path.parent.mkdir(exist_ok=True)
                    torch.save({"protocol": spec.SOURCE_PROTOCOLS[owner].EXPERIMENT_PROTOCOL, "root": root,
                        "period": p, "method": method, "updates": 8, "weights": weights}, path)
                    records[owner]["groups"][str(p)]["trained"][method] = {
                        "evaluation_update": 8, "final_freeze_check": "passed", "checkpoint": str(path)}
            for owner, record in records.items(): files[owner].write_text(json.dumps(record))
            # Upstream trajectories are omitted; actual checkpoint loads, actor checks and evaluation remain real.
            with patch.object(spec, "donor_result", side_effect=lambda r, owner: files[owner]), \
                    patch.object(experiment.joint, "qualify"), patch.object(experiment.upper, "qualify"), \
                    patch.object(experiment.lower, "qualify"), \
                    patch.object(experiment.fresh, "load_source", return_value=(clones, predictor(), spec.source_record(root), calibrations)), \
                    patch.object(experiment.learning, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment.learning, "update_mean", side_effect=AssertionError("Evaluation attempted learning")), \
                    patch.object(experiment.learning.native.joint, "_make_task", side_effect=lambda **kw: DenseTask()), \
                    patch.object(experiment.learning.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
                cell = experiment.run(root, preflight=True, output=folder/"evaluation/result.json")
                self.assertEqual(cell["cost"], spec.budget(preflight=True))
                self.assertEqual(experiment.aggregate([cell], preflight=True)["joint_recipe_confirmation"], "mechanical_only")
                for p, g in cell["groups"].items():
                    self.assertFalse(g["trained"])
                    self.assertEqual(set(g["evaluation"]), set(spec.VARIANTS))
                    self.assertTrue(all("upper_noise_seed" not in r and r["upper_replay_forward_calls"] == 0
                        for rows in g["evaluation"].values() for r in rows))
                bad = copy.deepcopy(cell)
                bad["groups"]["50"]["matched_training_budget"]["joint"]["native_episodes"] += 1
                with self.assertRaisesRegex(ValueError, "training budget"): experiment.qualify(bad, preflight=True)
                bad = copy.deepcopy(cell)
                bad["groups"]["50"]["checkpoints"]["staged_common"] = "Stage101_UJ.pt"
                with self.assertRaisesRegex(ValueError, "donor composition"): experiment.qualify(bad, preflight=True)
                path = spec.checkpoint_path(root, 50, "staged_common")
                payload = torch.load(path, map_location="cpu", weights_only=False)
                payload["weights"]["lower_actor"]["net.0.weight"] += .002
                torch.save(payload, path)
                weights = experiment.learning.native.joint.inference_weights(original)
                with self.assertRaises(AssertionError):
                    experiment.load_donors(records, weights, root, 50, dict.fromkeys(spec.budget(preflight=True), 0))
        for p, model in clones.items(): experiment.learning.native.curves.support.assert_frozen(model, snapshots[p])

    def test_joint_recipe_gate_cannot_be_rescued_by_teacher_or_independent_controls(self):
        cells = [{"root": r, "groups": {"both": {"effects": dict.fromkeys(spec.ENDPOINTS, 2.)}},
            "cost": spec.budget(preflight=False)} for r in spec.roots(preflight=False)]
        with patch.object(experiment, "qualify", side_effect=lambda c, **kw: c), \
                patch.object(spec, "BOOTSTRAP_DRAWS", 128), \
                patch.object(experiment.learning.native.np, "quantile", wraps=np.quantile) as quantile:
            result = experiment.aggregate(cells, preflight=False)
            self.assertEqual(quantile.call_args.args[1], [.05/36, 1-.05/36])
            self.assertEqual(result["joint_recipe_confirmation"], "supported")
            for c in cells: c["groups"]["both"]["effects"][spec.PRIMARY_ENDPOINTS[-1]] = 0.
            self.assertEqual(experiment.aggregate(cells, preflight=False)["joint_recipe_confirmation"], "not_supported")
            with self.assertRaises(ValueError): experiment.aggregate(cells[:-1], preflight=False)


if __name__ == "__main__":
    unittest.main()
