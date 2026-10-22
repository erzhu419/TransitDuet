import json
from pathlib import Path
import tempfile
import unittest

from freq_hrl.experiments import plan_validity_preregistration as registry
from freq_hrl.experiments.plan_validity_reliability import event_upper_bound
from scripts import prepare_plan_validity_preregistration as cli
from scripts.run_pointmaze_timing_baselines import build_timing_parser


class PreregistrationTest(unittest.TestCase):
    def test_reserved_roles_roots_and_phases_are_disjoint_and_reproducible(self):
        plan = registry.draft_plan()
        cells = list(registry.seed_cells(plan))
        self.assertEqual(len(cells), 22)
        self.assertEqual(cells, list(registry.seed_cells(plan)))
        for cell in cells:
            if cell["phase"] == "confirmation":
                self.assertEqual(len(cell["seed_roles"]["trigger_eval_seeds"]), 512)
                self.assertEqual(len(cell["probe_seeds"]), 8)
        result = registry.validate_plan(plan)
        self.assertFalse(result["scheduler_submission"])
        self.assertEqual(len(result["confirmation_optimizer_roots"]["pointmaze"]), 8)

    def test_frozen_recipe_matches_actual_parser_defaults(self):
        parser = build_timing_parser()
        for key, field in (("horizon", "horizon"), ("iterations", "primary_iterations"),
            ("max_events_per_class", "primary_events_per_class"), ("pairs_per_path", "primary_pairs_per_path"),
            ("learning_rate", "learning_rate"), ("threshold_quantile", "threshold_quantile"),
            ("ridge_alpha_grid", "ridge_alpha_grid"), ("termination_epochs", "termination_epochs")):
            self.assertEqual(parser.get_default(key), registry.CONTRACT[field])

    def test_global_alpha_and_risk_precision_are_sized_for_two_comparators(self):
        self.assertEqual(2 * 2 * registry.FAMILY_ALPHA, 0.05)
        alpha_i = registry.FAMILY_ALPHA / (3 * 8 * 2)
        self.assertLess(event_upper_bound(0, 512, alpha=alpha_i), 0.02)
        self.assertGreater(event_upper_bound(0, 16, alpha=alpha_i), 0.02)
        self.assertLessEqual(registry.validate_plan(registry.draft_plan())[
            "zero_harm_event_minimum_paths_at_registered_precision"], 512)

    def test_unknown_caps_and_unbound_artifacts_cannot_be_registered_as_ready(self):
        plan = registry.draft_plan()
        result = registry.validate_plan(plan)
        self.assertEqual(result["status"], "draft_not_locked")
        self.assertTrue(any("cpu_cap_seconds" in value for value in result["pending"]))
        plan["calibration"]["recipe_frozen"] = True
        plan["calibration"]["artifacts_frozen"] = True
        self.assertEqual(registry.validate_plan(plan)["status"], "draft_not_locked")
        plan["statistics"]["task_comparator_family_alpha"] = 0.025
        with self.assertRaisesRegex(ValueError, "contract changed"):
            registry.validate_plan(plan)

    def test_cli_round_trip_writes_only_a_structural_draft(self):
        with tempfile.TemporaryDirectory() as folder:
            path, checked = Path(folder) / "draft.json", Path(folder) / "checked.json"
            cli.main(["--output", str(path)])
            cli.main(["--input", str(path), "--output", str(checked)])
            result = json.loads(checked.read_text())
            self.assertEqual(result["plan"], json.loads(path.read_text())["plan"])
            self.assertEqual(result["readiness"]["status"], "draft_not_locked")
            self.assertFalse(result["readiness"]["performance_claim_supported"])

    def test_stress_failure_does_not_erase_a_supported_bounded_main_result(self):
        plan = registry.draft_plan()
        calibration = plan["calibration"]
        calibration.update(historical_seed_exposure_audit_complete=True, pilot_reports=["unit_fixture.json"],
                           recipe_frozen=True, artifacts_frozen=True)
        for row in calibration["per_task"].values():
            row.update(cpu_cap_seconds=10, environment_step_cap=1000, risk_distance=1,
                       termination_development_eligible=True)
        for cell in registry.seed_cells(plan):
            if cell["phase"] == "confirmation":
                for method in ("history", *registry.PRIMARY_COMPARATORS):
                    calibration["artifact_bindings"].append({"task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
                        "method": method, "budget_admitted": True, "controller_checkpoint": "unit_fixture.pt",
                        "source_report": "unit_fixture.json", "run_id": f"fixture_{method}_{cell['optimizer_seed']}"})
        roots = registry.validate_plan(plan)["confirmation_optimizer_roots"]
        summaries = [{"tasks": [task], "comparator_method": comparator, "family_alpha": registry.FAMILY_ALPHA,
            "risk_rate_margin": 0.02, "tail_ise_margin": 0.0, "expected_roots": roots[task],
            "panels": [{"task": task, "profile": profile, "status": "analyzed",
                "root_risk_counts": [{"optimizer_seed": root, "episodes": 512} for root in roots[task]],
                "joint_reliability_supported": profile == "in_distribution"} for profile in registry.PROFILES]}
            for task in registry.TASKS for comparator in registry.PRIMARY_COMPARATORS]
        result = registry.registered_claims(plan, summaries)
        self.assertTrue(result["C1_supported"])
        self.assertFalse(result["C2_supported"])
        self.assertIsNone(registry.registered_claims(plan, summaries[:-1])["C1_supported"])
        plan["calibration"]["artifacts_frozen"] = False
        self.assertIsNone(registry.registered_claims(plan, summaries)["C1_supported"])


if __name__ == "__main__":
    unittest.main()
