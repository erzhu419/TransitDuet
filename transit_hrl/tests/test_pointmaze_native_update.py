import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_native_update as experiment
from freq_hrl.experiments import pointmaze_first_update as archive
from freq_hrl.experiments import pointmaze_update_diagnostics as diagnostics
from freq_hrl.experiments import pointmaze_matched_upper as native
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from scripts import pointmaze_native_update_stage61_spec as spec
from scripts.submit_pointmaze_native_update_stage61_scheduleurm import task_specification
import test_pointmaze_update_diagnostics as archive_data
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class NativeUpdateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fresh_paired_paths_frozen_budget_and_dynamic_pool(self):
        b = spec.budget(preflight=False)
        self.assertEqual(8 * b["native_trace_audits"], 1792)
        self.assertEqual(8 * b["native_evaluation"]["primitive_steps"], 2150400)
        self.assertEqual(8 * b["native_evaluation"]["upper_inference_calls"], 32256)
        self.assertEqual(8 * b["candidate_checkpoint_writes"], 96)
        self.assertEqual(b["archive"], spec.source.budget(preflight=False))
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                seeds = set(spec.seed_roles(root, preflight=preflight)["evaluation"])
                self.assertFalse(seeds.intersection(seen))
                seen.update(seeds)
                for origin in (spec.deployment, spec.deployment.SOURCE_SPEC):
                    used = {s for values in origin.seed_roles(root, preflight=preflight).values() for s in values}
                    self.assertFalse(seeds.intersection(used))
        task = task_specification("unit_stage61", 310011, preflight=False)
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03d}" for i in range(1, 7)])
        self.assertEqual((task["cpu"], task["ram_mb"]), (9, 12288))

    def test_bootstrap_equal_root_means_reproducible_and_signed(self):
        rows = [{"endpoints": {k: float(i + 1) for k in spec.ENDPOINTS}} for i in range(8)]
        positive, negative, null = spec.ENDPOINTS[:3]
        for row in rows:
            row["endpoints"][negative], row["endpoints"][null] = -1., 0.
        result = experiment.bootstrap(rows)
        self.assertEqual(result, experiment.bootstrap(rows))
        self.assertEqual(result[positive]["mean"], 4.5)
        self.assertEqual(result[positive]["effect"], "positive")
        self.assertEqual(result[negative]["effect"], "negative")
        self.assertEqual(result[null]["effect"], "inconclusive")
        self.assertEqual(len(result), 12)

    def test_gate_requires_all_arms_periods_and_keeps_failed_endpoint(self):
        cells = [{"root": root, "archive_replay": {},
            "native_evaluation_counts": dict.fromkeys(experiment.learned.COUNT_KEYS, 0),
            **dict.fromkeys(("native_trace_audits", "candidate_checkpoint_writes", "evaluation_forecaster_loads",
                "new_forecaster_fits", "supervised_steps"), 0)} for root in spec.roots(preflight=False)]
        endpoint = "period100:joint_ppo:backtracking_kl_minus_clone"
        for effect, expected in ((1., "passed"), (-1., "failed"), (0., "failed")):
            def row(c, **kwargs):
                return {"root": c["root"], "endpoints": {k: effect if k == endpoint else 1. for k in spec.ENDPOINTS}}
            with patch.object(experiment, "qualify", side_effect=row), \
                    patch.object(archive, "aggregate", return_value={"root_rows": []}):
                result = experiment.aggregate(cells, preflight=False)
            self.assertEqual(result["repair_gain_gate"], "passed")
            self.assertEqual(result["training_gain_gate"], expected)
            self.assertEqual(result["primary_endpoints"][endpoint]["mean"], effect)
        with self.assertRaises(ValueError):
            experiment.aggregate(cells[:-1], preflight=False)

    def test_pipeline_exact_replay_shared_clone_saved_weights_and_native_counts(self):
        archive_data.UpdateDiagnosticsTest.setUpClass()
        helper = archive_data.UpdateDiagnosticsTest()
        model, predictor = helper.model(), helper.predictor
        source = {"config": json.loads(json.dumps(model.config.__dict__)),
            "checkpoints": {"50": "clone50.pt", "100": "clone100.pt"}}
        snapshots = {}
        evaluate = experiment.evaluate
        def observed(pool, candidate, predictor, args, **kwargs):
            before = copy.deepcopy(candidate.state_dict())
            rows = evaluate(pool, candidate, predictor, args, **kwargs)
            torch.testing.assert_close(candidate.state_dict(), before, atol=0, rtol=0)
            snapshots[str(kwargs["directory"])] = before
            return rows
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            file, diagnostic_file, rejection_file, backtracking_file = [directory / stage / "result.json"
                for stage in ("stage57", "stage58", "stage59", "stage60")]
            with patch.object(native, "load_source", side_effect=lambda *a, **kw:
                    ({p: copy.deepcopy(model) for p in ("50", "100")}, predictor, source)), \
                    patch.object(native, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(diagnostics, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(archive, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(diagnostics.spec, "source_result", return_value=file), \
                    patch.object(archive.spec, "diagnostic_result", return_value=diagnostic_file), \
                    patch.object(spec.source, "diagnostic_result", return_value=diagnostic_file), \
                    patch.object(spec.source, "rejection_result", return_value=rejection_file), \
                    patch.object(spec, "source_result", return_value=backtracking_file), \
                    patch.object(joint, "_make_task", side_effect=lambda **kw: DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(-2 * np.ones(2), 2 * np.ones(2))):
                native.train(310001, preflight=True, output=file)
                diagnostics.replay(310001, preflight=True, output=diagnostic_file)
                archive.replay(310001, preflight=True, output=rejection_file)
                reference = archive.replay(310001, preflight=True, output=backtracking_file, specification=spec.source)
                with patch.object(experiment, "evaluate", side_effect=observed):
                    result = experiment.train(310001, preflight=True, output=directory / "stage61/result.json")
                summary = experiment.aggregate([result], preflight=True)
                self.assertEqual(result["archive_replay"]["comparisons"], reference["comparisons"])
                self.assertEqual(len(snapshots), 14)
                self.assertEqual(sum(path.endswith("/clone") for path in snapshots), 2)
                for p, arms in result["checkpoints"].items():
                    for arm, treatments in arms.items():
                        for treatment, file in treatments.items():
                            saved = torch.load(file, map_location="cpu", weights_only=False)
                            self.assertEqual((saved["protocol"], saved["root"], saved["period"], saved["arm"], saved["treatment"]),
                                (spec.EXPERIMENT_PROTOCOL, 310001, int(p), arm, treatment))
                            torch.testing.assert_close(saved["state_dict"], snapshots[str(Path(file).parent)], atol=0, rtol=0)
                self.assertEqual(summary["status"], "preflight_passed")
                self.assertEqual(summary["native_evaluation_counts"], spec.budget(preflight=True)["native_evaluation"])
                self.assertEqual(summary["native_trace_audits"], 28)
                self.assertEqual(summary["candidate_checkpoint_writes"], 12)
                self.assertEqual(summary["evaluation_forecaster_loads"], 0)
                for mutation in ("pairing", "count", "execution", "identity", "duplicate_clone"):
                    bad = copy.deepcopy(result)
                    stage = bad["evaluation_rows"]["50"]
                    row = stage["zero_train"]["backtracking_kl"][0]
                    if mutation == "pairing":
                        row["seed"] += 1
                    elif mutation == "count":
                        bad["native_evaluation_counts"]["primitive_steps"] += 1
                    elif mutation == "execution":
                        row["executed_action_rms"] = .1
                    elif mutation == "identity":
                        bad["archive_reproduction_check"] = "failed"
                    else:
                        stage["clone"].append(copy.deepcopy(stage["clone"][0]))
                    with self.assertRaises(ValueError):
                        experiment.qualify(bad, preflight=True)


if __name__ == "__main__":
    unittest.main()
