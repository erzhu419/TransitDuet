import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_upper_refinement as experiment
from scripts import pointmaze_upper_refinement_stage101_spec as spec
from scripts.submit_pointmaze_upper_refinement_stage101_scheduleurm import task_specification, qualification_task
import test_pointmaze_call_weighted as call_fixture
import test_pointmaze_feasible_credit as fixture
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool
from test_pointmaze_upper_paths import predictor


class UpperRefinementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fixed_baseline_compositions_fresh_roles_and_incremental_budget(self):
        seen = set()
        for preflight in (True, False):
            b, old = spec.budget(preflight=preflight), spec.reference.budget(preflight=preflight)
            self.assertEqual({k: v for k, v in b.items() if k != "upper_initialization_checks"}, old)
            self.assertEqual(b["upper_initialization_checks"], 4)
            for root in spec.roots(preflight=preflight):
                seeds = call_fixture.seeds(spec.seed_roles(root, preflight=preflight))
                self.assertEqual(len(seeds), len(set(seeds)))
                self.assertFalse(set(seeds) & seen)
                seen.update(seeds)
                prior = []
                for protocol in (spec.reference, spec.source, spec.source.source):
                    for pref in (True, False):
                        if root in protocol.roots(preflight=pref):
                            prior += call_fixture.seeds(protocol.seed_roles(root, preflight=pref))
                decoder = spec.source.source.decoder
                prior += decoder.seed_roles(root, preflight=False)["native_probe"]
                prior += sum(decoder.source.seed_roles(root, preflight=False).values(), [])
                self.assertFalse(set(seeds) & set(prior))
                task = task_specification("unit_stage101", root, preflight=preflight)
                self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertFalse(task.get("require_node"))
                self.assertEqual((task["cpu"], task["ram_mb"]), (3, 3072) if preflight else (9, 8192))
                self.assertTrue(task["result_dir"].endswith("/completion"))
            self.assertIsNone(qualification_task("unit_stage101", preflight=preflight)["result_dir"])
        self.assertEqual((spec.BOOTSTRAP_DRAWS, spec.BOOTSTRAP_SEED), (spec.reference.BOOTSTRAP_DRAWS, spec.reference.BOOTSTRAP_SEED))
        self.assertEqual(len(spec.ENDPOINTS), 28)
        for p in spec.PERIODS:
            for branch in ("common", "independent"):
                self.assertIn(f"{p}/refined_{branch}_minus_fixed_joint_{branch}", spec.PRIMARY_ENDPOINTS)
                self.assertEqual(spec.COMPOSITIONS[f"refined_{branch}"][1], spec.COMPOSITIONS[f"fixed_joint_{branch}"][1])
                self.assertEqual(spec.allocation(f"refined_{branch}", p), {"upper": .5})
        self.assertEqual((8*b["native_episodes"], 8*b["native_steps"]), (22016, 26419200))

    def test_actual_checkpoints_initialize_UJ_and_updates_keep_installed_lowers(self):
        root = 410011
        original = fixture.FeasibleCreditTest().source()
        clones = {str(p): copy.deepcopy(original) for p in spec.PERIODS}
        snapshots = {p: copy.deepcopy(m.state_dict()) for p, m in clones.items()}
        envelope = {"velocity_speed_q99": 1., "axis_min": [-1., -1.], "axis_max": [1., 1.]}
        calibrations = {str(p): {"alpha": .02, "envelope": envelope} for p in spec.PERIODS}
        initialized = []
        real_initialize = experiment.initialize_upper

        def inspect_initialization(models, donors, cost):
            real_initialize(models, donors, cost)
            for method, model in models.items():
                weights = experiment.learning.native.joint.inference_weights(model)
                torch.testing.assert_close(weights["upper_actor"], donors["joint_call"]["upper_actor"], atol=0, rtol=0)
                torch.testing.assert_close(weights["lower_actor"], donors[spec.LOWER_FOR_METHOD[method]]["lower_actor"], atol=0, rtol=0)
            initialized.append((models, donors))

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            files = {"lower": directory / "lower/result.json", "joint": directory / "joint/result.json"}
            records = {name: {"root": root, "groups": {}} for name in files}
            for path in files.values(): path.parent.mkdir()
            for p in spec.PERIODS:
                for record in records.values(): record["groups"][str(p)] = {"trained": {}}
                for i, method in enumerate(spec.CHECKPOINT_METHODS, 1):
                    name = "joint" if method == "joint_call" else "lower"
                    weights = experiment.learning.native.joint.inference_weights(original)
                    weights["lower_actor"]["net.0.weight"] += .001*i
                    if method == "joint_call": weights["upper_actor"]["net.0.weight"] += .005
                    path = files[name].parent / "final_weights" / f"period_{p}_{method}.pt"
                    path.parent.mkdir(exist_ok=True)
                    protocol = spec.source.source if name == "joint" else spec.source
                    torch.save({"protocol": protocol.EXPERIMENT_PROTOCOL, "root": root, "period": p,
                        "method": method, "updates": 8, "weights": weights}, path)
                    records[name]["groups"][str(p)]["trained"][method] = {
                        "evaluation_update": 8, "final_freeze_check": "passed", "checkpoint": str(path)}
            for name, record in records.items(): files[name].write_text(json.dumps(record))
            # Upstream full traces are omitted; checkpoint loading, initialization and mean updates remain real.
            with patch.object(spec, "donor_result", side_effect=lambda r, m: files["joint" if m == "joint_call" else "lower"]), \
                    patch.object(experiment.fresh.fresh_joint, "qualify"), patch.object(experiment.fresh.fresh_lower, "qualify"), \
                    patch.object(experiment.fresh.fresh_joint, "load_source", return_value=(clones, predictor(), spec.source_record(root), calibrations)), \
                    patch.object(experiment, "initialize_upper", side_effect=inspect_initialization), \
                    patch.object(experiment.learning, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment.learning.native.joint, "_make_task", side_effect=lambda **kw: DenseTask()), \
                    patch.object(experiment.learning.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
                cell = experiment.run(root, preflight=True, output=directory / "refined/result.json")
                self.assertEqual(cell["cost"], spec.budget(preflight=True))
                self.assertEqual(experiment.aggregate([cell], preflight=True)["refinement_confirmation"], "mechanical_only")
                self.assertEqual(len(initialized), 2)
                for p, (models, donors) in zip(spec.PERIODS, initialized):
                    for method, model in models.items():
                        weights = experiment.learning.native.joint.inference_weights(model)
                        torch.testing.assert_close(weights["lower_actor"], donors[spec.LOWER_FOR_METHOD[method]]["lower_actor"], atol=0, rtol=0)
                        self.assertFalse(torch.equal(weights["upper_actor"]["net.0.weight"], donors["joint_call"]["upper_actor"]["net.0.weight"]))
                    g = cell["groups"][str(p)]
                    self.assertEqual(g["upper_initialization"], spec.UPPER_INITIALIZATION)
                    self.assertEqual(set(g["evaluation"]), set(spec.VARIANTS))
                    for t in g["trained"].values():
                        self.assertIsNone(t["checkpoint"])
                        self.assertTrue(all(set(r["actors"]) == {"upper"} for r in t["history"]))
                    for rows in g["evaluation"].values():
                        self.assertTrue(all(r["upper_replay_forward_calls"] == 0 and "upper_noise_seed" not in r for r in rows))
                    experiment.learning.native.curves.support.assert_frozen(clones[str(p)], snapshots[str(p)])
                bad = copy.deepcopy(cell)
                bad["groups"]["50"]["upper_initialization"] = "U0"
                with self.assertRaisesRegex(ValueError, "registered final UJ"): experiment.qualify(bad, preflight=True)
                bad = copy.deepcopy(cell)
                bad["source_initialization"]["teacher_result"] = "old_teacher/result.json"
                with self.assertRaisesRegex(ValueError, "source record"): experiment.qualify(bad, preflight=True)

    def test_primary_refinement_gate_cannot_be_replaced_by_matching_or_route_gain(self):
        cells = [{"root": r, "groups": {"both": {"effects": dict.fromkeys(spec.ENDPOINTS, 2.)}},
            "cost": spec.budget(preflight=False)} for r in spec.roots(preflight=False)]
        for c in cells:
            for p in spec.PERIODS:
                c["groups"]["both"]["effects"][f"{p}/common_upper_independent_lower_minus_refined_independent"] = -2.
        with patch.object(experiment, "qualify", side_effect=lambda c, **kw: c), \
                patch.object(spec, "BOOTSTRAP_DRAWS", 128), \
                patch.object(experiment.learning.native.np, "quantile", wraps=np.quantile) as quantile:
            result = experiment.aggregate(cells, preflight=False)
            self.assertEqual(quantile.call_args.args[1], [.05/56, 1-.05/56])
            self.assertEqual((result["refinement_confirmation"], result["matched_upper_confirmation"]), ("supported", "supported"))
            for c in cells: c["groups"]["both"]["effects"][spec.PRIMARY_ENDPOINTS[-1]] = 0.
            result = experiment.aggregate(cells, preflight=False)
            self.assertEqual((result["refinement_confirmation"], result["matched_upper_confirmation"]), ("not_supported", "supported"))
            for c in cells:
                c["groups"]["both"]["effects"][spec.PRIMARY_ENDPOINTS[-1]] = 2.
                c["groups"]["both"]["effects"]["50/refined_common_minus_independent_upper_common_lower"] = -1.
            result = experiment.aggregate(cells, preflight=False)
            self.assertEqual((result["refinement_confirmation"], result["matched_upper_confirmation"]), ("supported", "not_supported"))
            with self.assertRaises(ValueError): experiment.aggregate(cells[:-1], preflight=False)


if __name__ == "__main__":
    unittest.main()
