from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments import plan_validity_development as dev
from freq_hrl.experiments import plan_validity_transfer as shared
from freq_hrl.experiments.compute_budget import count_work
from freq_hrl.experiments.plan_validity_preregistration import draft_plan, seed_cells
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from scripts import run_plan_validity_development_cell as worker
from scripts import submit_plan_validity_development_scheduleurm as submitter
from test_plan_validity_second_task import Controller, DenseTask


def reports_for(plan, phase="preflight", run_name="unit_dev"):
    output = []
    for cell, registered in zip(dev.development_cells(plan, phase), dev.expected_runs(plan, run_name, phase)):
        args = dev.cell_arguments(plan, run_name, cell, root=Path("."))
        output.append({"stage": phase, "run_name": run_name, "native_preflight": {"passed": True},
            "records": [{**registered, "status": "validation_complete", "accounting_complete": True,
                "hardware_profile": {"cpu": "fixture", "threads": 1},
                "configuration": _json_ready({k: v for k, v in vars(args).items() if k not in ("output", "dry_run")}),
                "offline_cost": {"cpu_seconds": 2, "counts": {}},
                "termination_readiness": {"eligible_for_fresh_confirmation": False}}]})
    return output


class DevelopmentSchedulerTest(unittest.TestCase):
    def test_phase_arguments_follow_registered_roles_and_only_declared_overrides(self):
        plan = draft_plan()
        confirmation = {cell["optimizer_seed"] for cell in seed_cells(plan) if cell["phase"] == "confirmation"}
        for phase, count, horizon, iterations in (("preflight", 6, 300, 2), ("cost_pilot", 12, 1200, 384)):
            cells = dev.development_cells(plan, phase)
            self.assertEqual(len(cells), count)
            for cell in cells:
                args = dev.cell_arguments(plan, "unit_dev", cell, root=Path("."))
                self.assertEqual((args.horizon, args.iterations), (horizon, iterations))
                self.assertEqual(args.tasks, [cell["task"]])
                self.assertEqual(args.threshold_quantile, 0.75)
                self.assertEqual(args.termination_epochs, 200)
                self.assertEqual(args.checkpoint_evaluation_interval, 8)
                self.assertNotIn(args.optimizer_seed, confirmation)
                for name, values in cell["seed_roles"].items():
                    self.assertEqual(getattr(args, name), values)
        with self.assertRaises(ValueError):
            dev.development_cells(plan, "confirmation")

    def test_scheduler_specs_are_unpinned_one_core_and_only_sync_public_results(self):
        args = SimpleNamespace(preregistration=submitter.ROOT / "protocols/plan_validity_preregistration_2026-10-22.json",
            stage="cost_pilot", run_name="unit_dev", preflight_gate=submitter.ROOT / "results/unit_dev/gates/preflight/gate.json",
            python_executable="/worker/python", ram_mb=2048)
        bundle = submitter.build_bundle(args, draft_plan())
        self.assertEqual(len(bundle["jobs"]), 12)
        self.assertTrue(all(row["configuration"]["iterations"] == 384 for row in bundle["cells"]))
        for job in bundle["jobs"]:
            self.assertEqual(job["allowed_nodes"], [f"node{i:03d}" for i in range(1, 7)])
            self.assertIsNone(job["require_node"])
            self.assertIsNone(job["preferred_node"])
            self.assertEqual((job["cpu"], job["vram"], job["ram_mb"]), (1, 0, 2048))
            self.assertIn("OMP_NUM_THREADS=1", job["cmd"])
            self.assertIn("--preflight-gate results/unit_dev/gates/preflight/gate.json", job["cmd"])
            self.assertEqual(job["stage_input_paths"], [str(args.preflight_gate.parent)])
            self.assertTrue(job["result_dir"].endswith("/public"))
            self.assertTrue(job["no_auto_requeue"])
            self.assertNotIn("pip install", job["cmd"])
            self.assertNotIn("sbatch", job["cmd"])

    def test_missing_failed_or_other_seed_preflight_never_releases_pilot(self):
        plan = draft_plan()
        reports = reports_for(plan)
        gate = dev.summarize_phase(plan, "unit_dev", "preflight", reports)
        self.assertTrue(gate["preflight_passed"])
        dev.require_preflight_gate(plan, "unit_dev", gate)
        missing = dev.summarize_phase(plan, "unit_dev", "preflight", reports[:-1])
        self.assertEqual(len(missing["missing"]), 1)
        with self.assertRaises(ValueError):
            dev.require_preflight_gate(plan, "unit_dev", missing)
        reports[-1]["records"][0]["status"] = "failed"
        reports[-1]["records"][0].pop("termination_readiness")
        failed = dev.summarize_phase(plan, "unit_dev", "preflight", reports)
        self.assertFalse(failed["preflight_passed"])
        self.assertEqual(failed["known_measured_cpu_seconds"], 12)
        changed = deepcopy(gate)
        changed["seed_recipe"]["base"] += 1
        with self.assertRaises(ValueError):
            dev.require_preflight_gate(plan, "unit_dev", changed)

    def test_measurement_seed_or_runtime_drift_is_not_silently_admitted(self):
        plan, reports = draft_plan(), reports_for(draft_plan())
        reports[0]["records"][0]["configuration"]["train_seeds"][0] += 1
        with self.assertRaisesRegex(ValueError, "arguments/seed roles"):
            dev.summarize_phase(plan, "unit_dev", "preflight", reports)
        reports = reports_for(plan)
        reports[0]["records"][0]["hardware_profile"]["threads"] = 2
        self.assertFalse(dev.summarize_phase(plan, "unit_dev", "preflight", reports)["preflight_passed"])

    def test_default_prepares_both_phases_without_calling_scheduler(self):
        with tempfile.TemporaryDirectory() as folder:
            for phase in dev.PHASES:
                output = Path(folder) / f"{phase}.json"
                command = ["--run-name", "unit_dev", "--stage", phase, "--output", str(output)]
                if phase == "cost_pilot":
                    command += ["--preflight-gate", str(submitter.ROOT / "results/unit_dev/gates/preflight/gate.json")]
                with patch.object(submitter, "execute_bulk", side_effect=AssertionError("no submission")):
                    submitter.main(command)
                data = json.loads(output.read_text())
                self.assertEqual(data["submission_status"], "prepared_not_submitted")
                if phase == "preflight":
                    self.assertTrue(all(not job["stage_input_paths"] for job in data["jobs"]))

    def test_explicit_pilot_submit_requires_content_gate_and_records_attempt_once(self):
        plan = draft_plan()
        with tempfile.TemporaryDirectory(dir=submitter.ROOT / "protocols") as folder:
            gate, output = Path(folder) / "gates/preflight/gate.json", Path(folder) / "pilot.json"
            gate.parent.mkdir(parents=True)
            gate.write_text(json.dumps(dev.summarize_phase(plan, "unit_dev", "preflight", [])))
            command = ["--run-name", "unit_dev", "--stage", "cost_pilot", "--preflight-gate", str(gate),
                       "--output", str(output), "--submit"]
            with patch.object(submitter, "execute_bulk") as schedule:
                with self.assertRaises(ValueError):
                    submitter.main(command)
                schedule.assert_not_called()
            gate.write_text(json.dumps(dev.summarize_phase(plan, "unit_dev", "preflight", reports_for(plan))))
            with patch.object(submitter, "execute_bulk") as schedule:
                submitter.main(command)
                self.assertEqual(len(schedule.call_args.args[0]), 12)
                with self.assertRaises(FileExistsError):
                    submitter.main(command)
                self.assertEqual(schedule.call_count, 1)

    def test_pilot_completion_does_not_relabel_collapsed_heads_as_eligible(self):
        plan = draft_plan()
        report = dev.summarize_phase(plan, "unit_dev", "cost_pilot", reports_for(plan, "cost_pilot"))
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["termination_development_eligible"], {"pointmaze": False, "reacher_tracking": False})
        self.assertFalse(report["performance_claim_supported"])

    def test_native_audit_uses_real_kernel_with_mock_plant_not_a_training_run(self):
        plan = draft_plan()
        cell = dev.development_cells(plan, "preflight")[0]
        args = dev.cell_arguments(plan, "unit_dev", cell, root=Path("."))
        rule = TimingRule("audit", "threshold", feature="plan_age_fraction", threshold=0.8)
        spec = shared.PlanValidityTaskSpec("pointmaze", dev._task_options(args), args.env_id)
        with (
            patch.object(shared, "_make_task", side_effect=lambda **kw: DenseTask(**{k: v for k, v in kw.items() if k != "env_id"})),
            patch.object(shared, "pointmaze_goal_bounds", return_value=(np.full(2, -2), np.full(2, 2))),
            patch("scripts.run_plan_validity_reliability.load_controller", return_value=(Controller(), args)),
        ):
            rows = [shared.run_episode(Controller(), task_spec=spec, seed=seed, args=args,
                    time_scale=shared.time_contract(args), rule=rule)[1] for seed in args.timing_selection_seeds]
            record = {"task": "pointmaze", "selected_rule": rule.to_dict(),
                "validation_ise": float(np.mean([r["tracking_squared_error_integral"] for r in rows])),
                "validation_return": float(np.mean([r["episode_return"] for r in rows]))}
            result = dev.native_preflight(record)
        self.assertTrue(result["passed"])
        self.assertEqual(result["upper_calls"], 6)
        self.assertTrue(result["conditional_future_prefix"])

    def test_worker_counts_nested_pipeline_once_and_preserves_failed_audit(self):
        plan = draft_plan()
        cell = dev.development_cells(plan, "preflight")[0]
        profile = {"cpu": "fixture", "threads": 1}

        @contextmanager
        def runtime():
            yield profile

        def pipeline(args, task):
            cache = args.output.with_suffix(".cache.json")
            cache.write_text(json.dumps({"branch_labels": [{"category": "fixture", "renew_waypoint_delta": 1,
                                                           "renew_ise_advantage": 0.1}]}))
            return {**dev.expected_runs(plan, "unit_dev", "preflight")[0], "status": "validation_complete",
                "accounting_complete": True, "hardware_profile": profile, "fit_cache_file": str(cache),
                "selected_rule": TimingRule("fixture", "legacy", model={"weights": [1, 2]}).to_dict(),
                "configuration": _json_ready({k: v for k, v in vars(args).items() if k not in ("output", "dry_run")}),
                "offline_cost": {"cpu_seconds": 1000, "counts": {"environment_steps": 7}}}

        def audit(record):
            count_work("environment_steps", 3)
            raise RuntimeError("fixture native replay failed")

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "plan.json"
            source.write_text(json.dumps({"plan": plan}))
            command = ["--preregistration", str(source), "--run-name", "unit_dev", "--stage", "preflight",
                "--task", cell["task"], "--optimizer-seed", str(cell["optimizer_seed"]), "--method", "history"]
            with patch.object(worker, "ROOT", root), patch.object(worker, "single_thread_runtime", runtime), \
                 patch.object(worker, "run_trial", side_effect=pipeline), patch.object(worker, "native_preflight", side_effect=audit):
                self.assertEqual(worker.main(command), 1)
                with self.assertRaises(FileExistsError):
                    worker.main(command)
            result = json.loads((root / dev.cell_directory("unit_dev", cell) / "public/result.json").read_text())
            row = result["records"][0]
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["offline_cost"]["counts"]["environment_steps"], 10)
            self.assertLess(row["offline_cost"]["cpu_seconds"], 1000)
            self.assertEqual(row["pipeline_offline_cost"]["cpu_seconds"], 1000)
            self.assertNotIn("selected_rule", row)
            self.assertNotIn("model", row["selected_rule_summary"])
            private = json.loads(Path(result["private_source_report"]).read_text())
            self.assertEqual(private["records"][0]["selected_rule"]["model"]["weights"], [1, 2])


if __name__ == "__main__":
    unittest.main()
