from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from freq_hrl.experiments.plan_validity_development import cell_arguments, development_cells
from freq_hrl.experiments.plan_validity_preregistration import draft_plan, seed_cells
from freq_hrl.experiments.plan_validity_reliability import cross_future_attribution
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from scripts import run_plan_validity_development_diagnostic as worker
from scripts import submit_plan_validity_diagnostic_scheduleurm as submitter
from scripts import analyze_plan_validity_development_diagnostic as analyzer


def probe(seed, local, full, *, score=0, stop=False):
    return {"seed": seed, "deployed_score": score, "reference_score": score, "factual_stop": stop,
        "conditional_replicas": [{"local_advantage": l, "reference_local_advantage": l,
            "opportunity_gap": 0, "tail_advantage": g - l, "full_advantage": g}
            for l, g in zip(local, full)]}


def source_fixture(root, plan):
    files = []
    for index, cell in enumerate(development_cells(plan, "cost_pilot")):
        args = cell_arguments(plan, "fixture", cell, root=root)
        rule = TimingRule(cell["method"], "clock", threshold=0.5).to_dict()
        record = {"task": cell["task"], "optimizer_seed": cell["optimizer_seed"], "method": cell["method"],
            "status": "validation_complete", "run_id": str(index), "validation_ise": 1,
            "configuration": _json_ready({k: v for k, v in vars(args).items() if k not in ("output", "dry_run")}),
            "selected_rule": rule, "selected_rule_summary": {k: v for k, v in rule.items() if k != "model"},
            "termination_readiness": {"eligible_for_fresh_confirmation": False}}
        cache = root / f"cache{index}.json"
        cache.write_text('{"branch_labels": []}')
        record["fit_cache_file"] = str(cache)
        path = root / f"source{index}.json"
        path.write_text(json.dumps({"records": [record], "private_source_report": str(path)}))
        files.append(str(path))
    return {"status": "complete", "source_files": files, "hardware_profile": {"cpu": "fixture"}}


class DevelopmentDiagnosticTest(unittest.TestCase):
    def test_cross_future_gain_does_not_score_the_future_used_to_choose(self):
        rows = [probe(1, [1, 1, -1, -1], [1, 1, -1, -1]),
                probe(2, [-1, -1, 1, 1], [-1, -1, 1, 1])]
        result = cross_future_attribution(rows, threshold=0)
        self.assertLess(result["mean_gain_over_frozen"]["full_zero"], 0)
        self.assertLess(result["full_signal_gain_over_block_selected_constant"], 0)
        self.assertEqual(result["full_half_sign_agreement"], 0)
        self.assertLess(result["full_conditional_mean_second_moment"], 0)

    def test_state_conditional_headroom_is_compared_to_a_selected_constant(self):
        rows = [probe(1, [2] * 4, [2] * 4), probe(2, [-2] * 4, [-2] * 4)]
        result = cross_future_attribution(rows, threshold=0)
        self.assertEqual(result["mean_gain_over_frozen"]["full_zero"], 1)
        self.assertEqual(result["full_signal_gain_over_block_selected_constant"], 1)
        self.assertEqual(result["probe_paths"], 2)
        self.assertEqual(result["full_conditional_mean_variance_across_probes"], 4)

    def test_threshold_conservatism_is_not_mislabelled_as_prediction_failure(self):
        result = cross_future_attribution([probe(1, [1] * 4, [1] * 4, score=1)], threshold=2)
        self.assertEqual(result["mean_gain_over_frozen"]["local_same_threshold"], 0)
        self.assertEqual(result["mean_gain_over_frozen"]["local_zero"], 1)
        self.assertEqual(result["mean_gain_over_frozen"]["score_zero"], 1)
        self.assertEqual(result["conditional_local_errors"]["deployed"]["cross_block_conditional_mean_mse"], 0)
        self.assertEqual(result["full_conditional_mean_variance_across_probes"], 0)

    def test_local_credit_can_be_predictable_but_have_the_wrong_episode_sign(self):
        result = cross_future_attribution([probe(1, [1] * 4, [-1] * 4, score=1)], threshold=0)
        self.assertEqual(result["mean_gain_over_frozen"]["local_zero"], -1)
        self.assertEqual(result["directions"]["A_to_B"]["local_to_full_sign_flip_fraction"], 1)
        with self.assertRaises(ValueError):
            cross_future_attribution([probe(1, [1] * 3, [1] * 3)], threshold=0)

    def test_manifest_uses_four_pilot_roots_and_no_confirmation_roles(self):
        plan = draft_plan()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            summary = source_fixture(root, plan)
            cells = submitter.diagnostic_cells(summary, plan)
            confirmation = {seed for cell in seed_cells(plan) if cell["phase"] == "confirmation"
                            for seeds in cell["seed_roles"].values() for seed in seeds}
            self.assertEqual(len(cells), 4)
            for cell in cells:
                self.assertEqual(len(cell["evaluation_seeds"]), 16)
                self.assertEqual(cell["probe_seeds"], cell["evaluation_seeds"][:2] + cell["evaluation_seeds"][8:10])
                self.assertFalse(confirmation.intersection(cell["evaluation_seeds"]))
            changed = json.loads(Path(summary["source_files"][0]).read_text())
            changed["records"][0]["configuration"]["train_seeds"][0] += 1
            Path(summary["source_files"][0]).write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "seed roles"):
                submitter.diagnostic_cells(summary, plan)

    def test_preparing_manifest_does_not_submit_and_only_syncs_public_outputs(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory(dir=submitter.ROOT / "protocols") as proto:
            root, plan = Path(folder), draft_plan()
            source = root / "summary.json"
            source.write_text(json.dumps(source_fixture(root, plan)))
            manifest = Path(proto) / "diagnostic.json"
            with patch.object(submitter, "execute_bulk") as schedule:
                submitter.main(["--pilot-summary", str(source), "--run-name", "fixture", "--manifest", str(manifest)])
                schedule.assert_not_called()
            data = json.loads(manifest.read_text())
            self.assertEqual(len(data["jobs"]), 4)
            self.assertEqual(data["conditional_replicates"], 8)
            for job in data["jobs"]:
                self.assertEqual((job["cpu"], job["vram"], job["ram_mb"]), (1, 0, 2048))
                self.assertIsNone(job["require_node"])
                self.assertIsNone(job["preferred_node"])
                self.assertTrue(job["result_dir"].endswith("/public"))
                self.assertEqual(job["stage_input_paths"], [])

    def test_failed_native_reproduction_blocks_any_new_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root, plan = Path(folder), draft_plan()
            summary = source_fixture(root, plan)
            cell = submitter.diagnostic_cells(summary, plan)[0]
            cell["output_directory"] = "results/fixture"
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"run_name": "fixture", "scope": "development_only", "cells": [cell],
                "source_hardware_profile": {"cpu": "fixture"}, "probes_per_episode": 3, "conditional_replicates": 8}))

            @contextmanager
            def runtime():
                yield {"cpu": "fixture"}

            with patch.object(worker, "ROOT", root), patch.object(worker, "single_thread_runtime", runtime), \
                 patch.object(worker, "load_controller", side_effect=lambda r: (object(), SimpleNamespace(**r["configuration"]))), \
                 patch.object(worker, "run_episode", return_value=(None, {"tracking_squared_error_integral": 99})), \
                 patch.object(worker, "evaluate_profile") as evaluation:
                self.assertEqual(worker.main(["--manifest", str(manifest), "--cell-index", "0"]), 1)
                evaluation.assert_not_called()
            result = json.loads((root / "results/fixture/public/result.json").read_text())
            self.assertIn("native reproduction", result["error"])
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["confirmation_released"])
            self.assertGreater(result["evaluation_cost"]["cpu_seconds"], 0)

    def test_successful_worker_preserves_hold_and_performs_only_frozen_rollouts(self):
        with tempfile.TemporaryDirectory() as folder:
            root, plan = Path(folder), draft_plan()
            source = source_fixture(root, plan)
            cell = submitter.diagnostic_cells(source, plan)[-1]
            cell["output_directory"] = "results/fixture"
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"run_name": "fixture", "scope": "development_only", "cells": [cell],
                "source_hardware_profile": {"cpu": "fixture"}, "probes_per_episode": 3,
                "conditional_replicates": 8, "risk_spec": {"distance": 1, "duration_steps": 25}}))

            @contextmanager
            def runtime():
                yield {"cpu": "fixture"}

            profile = {"episodes": [{"seed": seed, "candidate": {"tracking_squared_error_integral": 1},
                "comparator": {"tracking_squared_error_integral": 2}} for seed in cell["evaluation_seeds"]],
                "probes": [probe(seed, [1] * 8, [1] * 8) for seed in cell["probe_seeds"] for _ in range(3)],
                "local_errors": {}, "occupancy": {}, "probe_sampling": "uniform_bin_starts"}
            with patch.object(worker, "ROOT", root), patch.object(worker, "single_thread_runtime", runtime), \
                 patch.object(worker, "load_controller", side_effect=lambda r: (object(), SimpleNamespace(**r["configuration"]))), \
                 patch.object(worker, "run_episode", return_value=(None, {"tracking_squared_error_integral": 1})), \
                 patch.object(worker, "evaluate_profile", return_value=profile) as evaluation:
                self.assertEqual(worker.main(["--manifest", str(manifest), "--cell-index", "0"]), 0)
            self.assertEqual(evaluation.call_args.kwargs["seeds"], cell["evaluation_seeds"])
            result = json.loads((root / "results/fixture/public/result.json").read_text())
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["candidate_recipe_decision"], "PILOT_RECIPE_HOLD")
            self.assertFalse(result["termination_development_eligible"])
            self.assertFalse(result["confirmation_released"])
            self.assertFalse(result["lifecycle_costs_complete"])
            self.assertEqual(result["cross_future_attribution"]["probe_states"], 12)
            self.assertEqual(len(result["episodes"]), 16)

    def test_compact_analysis_does_not_inflate_roots_or_hide_a_missing_cell(self):
        with tempfile.TemporaryDirectory() as folder:
            root, plan = Path(folder), draft_plan()
            cells = submitter.diagnostic_cells(source_fixture(root, plan), plan)
            manifest = {"run_name": "fixture", "scope": "development_only", "cells": cells}
            for i, cell in enumerate(cells):
                cell["output_directory"] = f"results/{i}"
                row = {"run_name": "fixture", "status": "complete", **{k: cell[k] for k in
                    ("task", "optimizer_seed", "evaluation_seeds", "probe_seeds")},
                    "source_run_ids": {m: s["run_id"] for m, s in cell["sources"].items()},
                    "episode_panels": {panel: {"mean_ise": {"history": 1, "threshold": 2, "termination_history": 3}}
                                       for panel in ("all", "first_half", "second_half")},
                    "local_errors": {}, "cross_future_attribution": {}, "termination_development_eligible": False,
                    "occupancy": {"panels": {}}, "evaluation_cost": {"cpu_seconds": 10}}
                p = root / cell["output_directory"] / "public/result.json"
                p.parent.mkdir(parents=True)
                p.write_text(json.dumps(row))
            result = analyzer.summarize(manifest, root=root)
            self.assertEqual(result["completed_root_cells"], 4)
            self.assertEqual(result["task_means"]["reacher_tracking"]["optimizer_roots"], 2)
            self.assertFalse(result["performance_claim_supported"])
            self.assertEqual(result["candidate_recipe_decision"], "PILOT_RECIPE_HOLD")
            p.unlink()
            result = analyzer.summarize(manifest, root=root)
            self.assertEqual(result["status"], "incomplete_or_failed")
            self.assertIsNone(result["task_means"]["reacher_tracking"])
            self.assertEqual(result["completed_evaluation_cpu_seconds"], 30)


if __name__ == "__main__":
    unittest.main()
