from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from freq_hrl.domains.transit import native_paired_fleet_calibration as learning
from scripts import run_native_transit_paired_fleet_stage296 as spec
from scripts import submit_native_transit_paired_fleet_stage296_scheduleurm as submit


def fixture(profiles=2):
    rows = [{"root": 397, "scenario": "low_noise", "replica": i, "probe_index": 8,
        "time_s": 0., "direction": True, "original_choice": 0, "nominal_s": [0., 360., 720.],
        "native_state_v2": [0.]*12+[.2], "context": {"queues": {"True": 10., "False": 5.}},
        "occupancy_context": {"historical_peak": 12, "active_buses": [
            {"direction": True, "load_fraction": .5, "predicted_remaining_s": 1000.}]},
        "candidates": [{"commitment": {"prefix_planned_s": [0., 360.+a*10, 720.],
            "prefix_fleet_cost": a*a/12.}, "fleet_cost": a/12.} for a in range(6)]} for i in range(3)]
    raw, proposal = np.zeros((profiles, 3), int), np.ones((profiles, 3), int)
    first = np.zeros((3, 8, 6, 4)); first[:, :, 1] = [-1.1, -1., -.1, -.2]
    return rows, raw, proposal, first, first.copy(), np.arange(3).reshape(1, 3, 1)


class PairedFleetTest(unittest.TestCase):
    def test_namespace_roles_and_banks_are_disjoint_from_confirmation(self):
        scenes = [spec.scene_seed(k) for r in spec.ROLES for k in spec.keys(r)]
        future = [spec.future_seed(k, p, d) for r in spec.ROLES for k in spec.keys(r) for p in spec.PROBES for d in range(64)]
        self.assertEqual(len(scenes), 270)
        self.assertEqual(len(set(scenes)), len(scenes)); self.assertEqual(len(set(future)), len(future))
        self.assertLess(max(future), 2**31)
        old_scenes = {spec.native.scene_seed(k) for r in ("new", "pilot") for k in spec.native.keys(r)}
        old_futures = {spec.native.future_seed(k, p, d) for r in ("new", "pilot") for k in spec.native.keys(r) for p in spec.PROBES for d in range(48)}
        self.assertFalse(set(scenes)&old_scenes); self.assertFalse(set(future)&old_futures)
        self.assertEqual([len(v) for v in spec.BANKS.values()], [8, 8, 16, 32])
        banks = list(map(set, spec.BANKS.values()))
        self.assertFalse(any(banks[i]&banks[j] for i in range(4) for j in range(i)))

    def test_features_use_preaction_state_and_PS_only_not_labels_or_scene_ids(self):
        rows, raw, proposal, first, screen, _ = fixture()
        x = learning.features(rows, raw, proposal, first, screen)
        self.assertEqual(x.shape, (2, 3, 26))
        for row in rows:
            row.update(root=999, scenario="not_a_feature", scene_seed=123, labels="forbidden_future_labels")
        np.testing.assert_array_equal(x, learning.features(rows, raw, proposal, first, screen))
        self.assertTrue(np.isfinite(x).all())

    def test_independent_positive_targets_do_not_inherit_negative_screen_risk(self):
        rows, raw, proposal, first, screen, _ = fixture()
        x = learning.features(rows, raw, proposal, first, screen)
        model = learning.fit(x, np.full(raw.shape, .25), proposal != raw)
        np.testing.assert_allclose(learning.predict(model, x), .25, atol=1e-14)
        self.assertEqual(model["ridge"], 1.)
        self.assertLess(model["normal_equation_error"], 1e-10)
        with self.assertRaisesRegex(ValueError, "No changed"):
            learning.fit(x, np.zeros(raw.shape), np.zeros(raw.shape, bool))

    def test_calibration_is_stratified_prefix_clustered_and_duplicate_invariant(self):
        cube = np.arange(6).reshape(2, 3, 1)
        targets = np.array([[1., 2., 3., 2., 3., 4.]])
        c = learning.calibrate(np.zeros_like(targets), targets, np.ones_like(targets, bool), cube)
        self.assertAlmostEqual(c["bias"], 2.5)
        self.assertAlmostEqual(c["prefix_cluster_se"], np.sqrt(1/6))
        repeated = learning.calibrate(np.zeros((16, 6)), np.repeat(targets, 16, axis=0), np.ones((16, 6), bool), cube)
        for key in ("bias", "prefix_cluster_se", "upward_adjustment"):
            self.assertAlmostEqual(c[key], repeated[key])
        self.assertGreater(c["upward_adjustment"], c["bias"])
        with self.assertRaisesRegex(ValueError, "No eligible"):
            learning.calibrate(targets, targets, np.zeros_like(targets, bool), cube)

    def test_service_proposals_unchanged_and_matching_counts_preserved(self):
        _, raw, proposal, _, screen, cube = fixture()
        score = np.array([[-1., 1., -1.]]*2)
        sealed = learning.seal(raw, raw, proposal, screen, score, np.ones_like(score), cube)
        choices = np.asarray(sealed["choices"])
        np.testing.assert_array_equal(choices[:, 2], proposal)
        np.testing.assert_array_equal(choices[:, 3], raw)
        np.testing.assert_array_equal(choices[:, 4], [[1, 0, 1]]*2)
        self.assertTrue(np.all((choices[:, 4] == proposal)|(choices[:, 4] == raw)))
        self.assertAlmostEqual(sealed["matching"]["expected_thin_changes"], 4.)
        self.assertEqual(learning.action_unions(sealed, [0, 0, 0]), [[0, 1]]*3)

    def test_all_raw_cannot_pass_service_retention_or_joint_claim(self):
        _, raw, proposal, _, screen, cube = fixture()
        panels = np.zeros((3, 32, 6, 4)); panels[:, :, 1] = [-1., 0., -1., -.2]
        sealed = learning.seal(raw, raw, proposal, screen, np.ones_like(raw), np.ones_like(raw), cube)
        result = learning.summarize(sealed, panels, cube)
        self.assertEqual(result["wait_retention_vs_zero_abstain"], {"all32": 0., "first16": 0., "last16": 0.})
        self.assertFalse(result["service_retention_supported"]); self.assertFalse(result["joint_transfer_supported"])

    def test_joint_needs_both_halves_and_profiles_are_not_seeds(self):
        _, raw, proposal, _, screen, cube = fixture(1)
        sealed = learning.seal(raw, raw, proposal, screen, -np.ones_like(raw), -np.ones_like(raw), cube)
        panels = np.zeros((3, 32, 6, 4)); panels[:, :, 1] = [-1., 0., -1., -.2]
        result = learning.summarize(sealed, panels, cube)
        self.assertTrue(result["joint_transfer_supported"])
        repeated = {**sealed, **{k: np.repeat(np.asarray(sealed[k]), 16, axis=0).tolist()
            for k in ("choices", "proposal", "matched_thin_probabilities")}}
        again = learning.summarize(repeated, panels, cube)
        for pair in result["comparisons"]:
            for interval in ("prefix_CI95", "conditional_MC_CI95"):
                np.testing.assert_allclose(list(result["comparisons"][pair][interval].values()),
                    list(again["comparisons"][pair][interval].values()), rtol=0, atol=1e-14)
        panels[:, :16, 1, 0], panels[:, 16:, 1, 0] = 1., -3.
        self.assertFalse(learning.summarize(sealed, panels, cube)["joint_transfer_supported"])

    def test_fit_freezes_model_before_calibration_and_never_opens_E32(self):
        rows, raw, proposal, first, screen, _ = fixture()
        x = learning.features(rows, raw, proposal, first, screen)
        target = np.zeros((3, 16, 6, 4)); target[:, :, 1] = [-.03, -.01, -.02, -.2]
        opened = []
        with tempfile.TemporaryDirectory() as temp, patch.object(spec, "ROOT", Path(temp)):
            def load_inputs(run, role):
                opened.append((role, "PS"))
                return rows, raw, raw, proposal, screen, x, np.ones(raw.shape, bool)

            def load_target(run, role, bank):
                self.assertIn(role, ("fit", "calibration")); self.assertEqual(bank, "target")
                if role == "calibration":
                    self.assertTrue((spec.directory(run)/"fit_raw/model_before_calibration.json").exists())
                opened.append((role, bank))
                return target, {}

            with patch.object(spec, "inputs", side_effect=load_inputs), patch.object(spec, "load_panel", side_effect=load_target):
                spec.fit("unit296")
            self.assertEqual(opened, [("fit", "PS"), ("fit", "target"), ("calibration", "PS"), ("calibration", "target"), ("test", "PS")])
            plan = spec.read(spec.directory("unit296")/"fit/result.json")
            sealed = spec.read(spec.directory("unit296")/"sealed_raw/test.json")
            self.assertFalse(plan["test_E32_labels_read"]); self.assertFalse(plan["Stage294_E32_labels_read"])
            self.assertEqual(plan["risk_fits"], 1)
            np.testing.assert_array_equal(np.asarray(sealed["choices"])[:, 4], proposal)
            self.assertEqual(plan["evaluation_native_episodes"], 288)

    def test_test_T16_is_forbidden_and_incomplete_evaluation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not part"):
            spec.load_panel("unit296", "test", "target")
        with tempfile.TemporaryDirectory() as temp, patch.object(spec, "ROOT", Path(temp)), patch.object(spec, "load_rows", return_value=[]):
            spec.write_json(spec.directory("unit296")/"fit/result.json", {"active_indices": [0]})
            spec.write_json(spec.directory("unit296")/"sealed_raw/test.json", {"states": [{"key": list(spec.keys("test")[0]), "probe": 8}]})
            with self.assertRaises(FileNotFoundError): spec.report("unit296")

    def test_training_panel_checks_physical_counts_seeds_identity_and_components(self):
        key, probe = spec.keys("fit")[0], spec.PROBES[0]
        receipt = {"contract": spec.contract(), "software_qualified": True, "key": list(key), "probe": probe,
            "candidates": list(range(6)), "draws": list(spec.BANKS["proposal"]),
            "native_episodes": 56, "exact_identities": 8, "state_futures": 8}
        labels = [{"draw": d, "future_seed": spec.future_seed(key, probe, d),
            "values": {str(a): [0., 0., 0., 0.] if a == 0 else [1., .25, .75, -.1] for a in range(6)}}
            for d in spec.BANKS["proposal"]]
        def load(path):
            if path.name == "rows.json": return [{"probe_index": probe, "original_choice": 0}]
            return receipt if path.name == "receipt.json" else labels
        with patch.object(spec, "keys", return_value=[key]), patch.object(spec, "PROBES", (probe,)), patch.object(spec, "read", side_effect=load):
            receipt["contract"] = spec.contract()
            panel, audit = spec.load_panel("unit296", "fit", "proposal")
            self.assertEqual(panel.shape, (1, 8, 6, 4)); self.assertEqual(audit["native_episodes"], 56)
            receipt["native_episodes"] = 55
            with self.assertRaisesRegex(ValueError, "receipt"): spec.load_panel("unit296", "fit", "proposal")
            receipt["native_episodes"] = 56
            labels[0]["future_seed"] += 1
            with self.assertRaisesRegex(ValueError, "future"): spec.load_panel("unit296", "fit", "proposal")
            labels[0]["future_seed"] -= 1
            for action, field, value in ((0, 3, 1.), (1, 0, float("nan")), (1, 2, 3.)):
                old = labels[0]["values"][str(action)][field]
                labels[0]["values"][str(action)][field] = value
                with self.subTest(action=action, field=field), self.assertRaisesRegex(ValueError, "Invalid"):
                    spec.load_panel("unit296", "fit", "proposal")
                labels[0]["values"][str(action)][field] = old

    def test_native_replay_archived_rng_and_registered_future_identity(self):
        calls, wrong = [], [False]
        def episode(*args, **kwargs):
            candidate = kwargs.get("candidate"); seed = kwargs["plan_class"].keywords["continuation_seed"]
            self.assertEqual(kwargs["reference"].prefix, "archived_prefix")
            calls.append((candidate, seed))
            row = {"supply": [], "probe": {"selected": candidate}, "service_cost_restricted": 1+(candidate or 0)*.05,
                "peak_fleet": 12+(candidate or 0), "restricted_wait_horizon_min": 2-(candidate or 0)*.2}
            plan = SimpleNamespace(prefix="archived_prefix", snapshot={"rng": {"torch": [1, 2]}},
                env=SimpleNamespace(scenario_tape=SimpleNamespace(seed=seed+int(wrong[0]), _streams={"used": True})))
            return row, plan
        source = SimpleNamespace(shared=SimpleNamespace(episode=episode, qualify=Mock(), policy_trace=Mock(return_value=[])),
            qualify_observer=Mock(), validate_continuations=Mock())
        key = spec.keys("fit")[0]
        with tempfile.TemporaryDirectory() as temp, patch.object(spec, "ROOT", Path(temp)), \
                patch.object(spec.native, "native_inputs", return_value=(source, {}, "fixed_checkpoint")), patch("torch.set_rng_state") as restore:
            path = spec.prefix_path("unit296", key)
            spec.write_json(path/"rows.json", [{"probe_index": 8, "original_choice": 0}])
            with (path/"prefix_refs.pkl").open("wb") as f:
                pickle.dump({"prefix": "archived_prefix", "snapshots": {8: {"rng": {"torch": [1, 2]}}}}, f)
            labels, receipt = spec.rollout("unit296", key, 8, (16, 17), [0, 1], path/"sample")
            self.assertEqual(receipt["native_episodes"], 6); self.assertEqual(restore.call_count, 6)
            self.assertEqual([c for c, _ in calls], [None, 0, 1]*2)
            self.assertEqual(labels[0]["values"]["0"], [0., 0., 0., 0.])
            wrong[0] = True
            with self.assertRaisesRegex(ValueError, "shared future"):
                spec.rollout("unit296", key, 8, (16,), [0, 1], path/"wrong")

    def test_scheduler_roles_resources_gates_and_minimal_staging(self):
        self.assertEqual(len(submit.tasks("unit296", "pilot")), 10)
        self.assertEqual(len(submit.tasks("unit296", "capture")), 260)
        self.assertEqual(len(submit.tasks("unit296", "menus")), 3120)
        targets = submit.tasks("unit296", "targets")
        self.assertEqual(len(targets), 1680)
        self.assertFalse(any("/test/" in t["signature"] for t in targets))
        t = submit.task_specification("unit296", "pilot", spec.keys("pilot")[0])
        self.assertEqual((t["cpu"], t["ram_mb"], t["require_node"]), (1, 1536, None))
        self.assertEqual(t["stage_excludes"], ["results", "**/__pycache__"])
        self.assertEqual(t["stage_input_paths"][-1], str(spec.directory("unit296")/"inputs"))
        self.assertNotIn(str(spec.directory("unit296")), t["stage_input_paths"])
        with self.assertRaisesRegex(ValueError, "Preceding phase"):
            submit.require_completed([{**t, "status": "running", "exit_code": None}], [t])
        submit.require_completed([{**t, "status": "done", "exit_code": 0}], [t])

    def test_cold_start_does_not_import_historical_native_stage_chain(self):
        result = subprocess.run([sys.executable, "-c", "from scripts import run_native_transit_paired_fleet_stage296; import sys; assert 'scripts.run_native_transit_value_expansion_stage187' not in sys.modules"],
            cwd=spec.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_parallel_collection_preserves_tasks_and_fit_waits_for_both_banks(self):
        run = "unit296"
        menus, targets = submit.tasks(run, "menus"), submit.tasks(run, "targets")
        self.assertEqual(submit.tasks(run, "collect"), menus+targets)
        self.assertEqual(len(menus)+len(targets), 4800)
        self.assertEqual(submit.preceding_phases("collect"), ("capture",))
        self.assertEqual(submit.preceding_phases("targets"), ("capture",))
        self.assertEqual(submit.preceding_phases("fit"), ("menus", "targets"))
        complete = [{**t, "status": "done", "exit_code": 0} for t in menus+targets]
        submit.require_completed(complete, menus+targets)
        for index in (0, len(menus)):
            with self.subTest(index=index):
                original = complete[index]
                complete[index] = {**original, "status": "running", "exit_code": None}
                with self.assertRaisesRegex(ValueError, "Preceding phase"):
                    submit.require_completed(complete, menus+targets)
                complete[index] = original


if __name__ == "__main__": unittest.main()
