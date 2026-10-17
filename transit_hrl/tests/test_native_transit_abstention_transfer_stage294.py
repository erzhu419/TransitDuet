from itertools import combinations
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from freq_hrl.domains.transit import native_abstention_transfer as rule
from scripts import run_native_transit_abstention_transfer_stage294 as spec
from scripts import submit_native_transit_abstention_transfer_stage294_scheduleurm as submit


class AbstentionTransferTest(unittest.TestCase):
    def test_fresh_scene_and_future_namespaces_do_not_collide(self):
        scenes = [spec.scene_seed(k) for role in ("new", "pilot") for k in spec.keys(role)]
        futures = [spec.future_seed(k, p, d) for role in ("new", "pilot") for k in spec.keys(role)
            for p in spec.PROBES for d in range(48)]
        self.assertEqual(len(scenes), 250)
        self.assertEqual(len(set(scenes)), 250)
        self.assertEqual(len(futures), len(set(futures)))
        self.assertLess(max(futures), 2**31)
        old = {1858000000+role*1000000+ri*100000+si*1000+r
            for role, n in enumerate((6, 4, 1)) for ri in range(2) for si in range(5) for r in range(n)}
        old_future = {1958000000+role*1000000+ri*100000+si*10000+r*1000+pi*100+d
            for role, n in enumerate((6, 4, 1)) for ri in range(2) for si in range(5)
            for r in range(n) for pi in range(6) for d in range(90)}
        self.assertFalse(old&set(scenes))
        self.assertFalse(old_future&set(futures))
        self.assertEqual([len(v) for v in spec.BANKS.values()], [8, 8, 32])

    def test_proposal_wait_feasibility_fallback_and_raw_priority(self):
        panels = np.zeros((4, 8, 6, 4))
        panels[:, :, 1, 0] = -3
        panels[0, :, 1, 3] = 1
        panels[1, :, 2, 0] = -2
        panels[2, :, 1, 0] = 0
        panels[3, :, 2, 0] = -4
        logits = np.zeros((4, 6))
        logits[1, 1] = 1
        logits[3, 2] = 1
        chosen = rule.propose(logits, np.array([0, 0, 0, 2]), np.zeros(4, int), panels)
        np.testing.assert_array_equal(chosen, [0, 2, 0, 2])
        logits[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "outside"):
            rule.propose(logits, np.ones(4, int), np.zeros(4, int), panels)

    def test_zero_events_rejected_signed_cancellation_retained(self):
        panels = np.zeros((4, 8, 6, 4))
        panels[1, :, 1, 1] = [1, -1]*4
        panels[2, :, 1, 1] = -1
        panels[3, :, 1, 1] = 1
        raw, proposal = np.zeros((1, 4), int), np.ones((1, 4), int)
        eligible, keep = rule.screen(proposal, raw, panels)
        np.testing.assert_array_equal(eligible, [[True, True, True, False]])
        np.testing.assert_array_equal(keep, [[False, True, True, False]])
        sealed = rule.seal(raw, raw, proposal, panels, np.arange(4).reshape(1, 4, 1))
        np.testing.assert_allclose(sealed["matched_thin_probabilities"], [[2/3, 2/3, 2/3, 0]])
        self.assertEqual(sealed["matching"]["kept_changes"], 2)

    def test_matching_is_per_profile_group_probe_not_pooled(self):
        cube = np.arange(12).reshape(2, 3, 2)
        eligible = np.ones((2, 12), bool)
        keep = np.zeros_like(eligible)
        keep[0, [0, 1, 3, 6]] = True
        probability, counts = rule.thin_probabilities(cube, eligible, keep)
        np.testing.assert_allclose(probability[0, cube], np.array([[[1/3, 2/3]]*3, [[1/3, 0]]*3]))
        self.assertEqual(probability[1].sum(), 0)
        self.assertAlmostEqual(probability.sum(), keep.sum())
        self.assertEqual(counts["strata"], 8)

    def test_expected_random_control_and_profile_duplicates_not_seeds(self):
        raw, proposal = np.zeros((1, 3), int), np.ones((1, 3), int)
        sealed = {"choices": np.stack((raw, raw, np.array([[1, 0, 0]]), raw), axis=1).tolist(),
            "proposal": proposal.tolist(), "matched_thin_probabilities": [[1/3]*3]}
        panels = np.zeros((3, 32, 6, 4))
        panels[:, :, 1, 0] = np.array([-2, 2, 2])[:, None]+(np.arange(32)-15.5)[None]/32
        expected = np.mean([panels[list(c), :, 1, 0].sum()/3/32 for c in combinations(range(3), 1)])
        result = rule.summarize(sealed, panels, np.arange(3).reshape(1, 3, 1))
        c = result["comparisons"]["matched_thin_vs_raw"]
        self.assertAlmostEqual(c["mean"]["cost"], expected)
        expected_se = np.sqrt(np.var(panels[:, :, 1, 0]/3, axis=1, ddof=1).sum()/32)/3
        self.assertAlmostEqual(c["conditional_MC_se"]["cost"], expected_se)
        ix = np.random.default_rng(294).choice(3, (5000, 3), replace=True)
        np.testing.assert_allclose(c["prefix_CI95"]["cost"], np.quantile((panels[:, :, 1, 0].mean(axis=1)/3)[ix].mean(axis=1), [.025, .975]))
        repeated = {k: np.repeat(np.asarray(v), 16, axis=0).tolist() for k, v in sealed.items()}
        again = rule.summarize(repeated, panels, np.arange(3).reshape(1, 3, 1))
        for contrast in result["comparisons"]:
            for field in ("mean", "conditional_MC_se"):
                np.testing.assert_allclose(list(result["comparisons"][contrast][field].values()),
                    list(again["comparisons"][contrast][field].values()), rtol=0, atol=1e-15)
            for field in ("prefix_CI95", "conditional_MC_CI95"):
                np.testing.assert_allclose(list(result["comparisons"][contrast][field].values()),
                    list(again["comparisons"][contrast][field].values()), rtol=0, atol=1e-15)

    def test_unions_include_random_support_but_skip_identity_states(self):
        raw = np.zeros((1, 3), int)
        sealed = {"choices": np.stack((raw, np.array([[0, 2, 0]]), raw, raw), axis=1),
            "proposal": [[1, 1, 1]], "matched_thin_probabilities": [[.5, 0, 0]]}
        self.assertEqual(rule.action_unions(sealed, [3, 3, 3]), [[0, 1, 3], [0, 2, 3], []])

    def test_cube_rejects_missing_or_duplicate_supported_probe_rows(self):
        rows = [{"root": 397, "scenario": "hour_burst", "replica": r, "probe_index": p}
            for r in range(3) for p in (8, 9)]
        self.assertEqual(spec.cube(rows).shape, (1, 3, 2))
        with self.assertRaises((KeyError, ValueError)):
            spec.cube(rows[:-1])
        with self.assertRaises(ValueError):
            spec.cube(rows+[rows[0]])

    def test_scheduler_gate_and_small_input_staging(self):
        task = submit.task_specification("unit294", "menus", spec.keys()[0], spec.PROBES[0], "proposal")
        self.assertEqual((task["cpu"], task["ram_mb"], task["require_node"]), (1, 1536, None))
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertNotIn(str(spec.directory("unit294")), task["stage_input_paths"])
        self.assertEqual(task["stage_input_paths"][-1], str(spec.directory("unit294")/"inputs"))
        self.assertEqual(task["result_dir"], str(spec.task_receipt_dir("unit294", "menus", spec.keys()[0], spec.PROBES[0], "proposal")))
        self.assertIn("/receipts/capture/", submit.task_specification("unit294", "capture", spec.keys()[0])["result_dir"])
        self.assertIn("/receipts/pilot/", submit.task_specification("unit294", "pilot", spec.keys("pilot")[0])["result_dir"])
        self.assertIn("/receipts/evaluate/", submit.task_specification("unit294", "evaluate", index=0, chunk=0)["result_dir"])
        self.assertEqual(len(submit.tasks("unit294", "capture")), 240)
        self.assertEqual(len(submit.tasks("unit294", "menus")), 2880)
        previous = {**task, "status": "running", "exit_code": None}
        with self.assertRaisesRegex(ValueError, "Preceding phase"):
            submit.require_completed([previous], [task])
        previous.update(status="done", exit_code=0)
        submit.require_completed([previous], [task])

    def test_cold_entry_does_not_import_historical_stage_chain(self):
        command = "from scripts import run_native_transit_abstention_transfer_stage294; import sys; assert 'scripts.run_native_transit_frozen_prefix_stage287' not in sys.modules; assert 'scripts.run_native_transit_value_expansion_stage187' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", command], cwd=spec.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_report_refuses_incomplete_evaluation(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(spec, "ROOT", Path(temp)):
            spec.write_json(spec.directory("unit294")/"seal/result.json", {"active_indices": [0]})
            spec.write_json(spec.directory("unit294")/"sealed_raw/primary.json", {"states": [{"key": list(spec.keys()[0]), "probe": 8, "candidates": [0, 1]}]})
            with self.assertRaises(FileNotFoundError):
                spec.report("unit294")

    def test_native_rollout_restores_archived_rng_and_checks_shared_tape(self):
        calls = []
        wrong_seed = False

        def episode(*args, **kwargs):
            candidate = kwargs.get("candidate")
            seed = kwargs["plan_class"].keywords["continuation_seed"]
            self.assertEqual(kwargs["reference"].prefix, "archived_prefix")
            calls.append((candidate, seed))
            row = {"supply": [], "probe": {"selected": candidate}, "service_cost_restricted": 1+(candidate or 0)*.05,
                "peak_fleet": 12+(candidate or 0), "restricted_wait_horizon_min": 2-(candidate or 0)*.2}
            plan = SimpleNamespace(prefix="archived_prefix", snapshot={"rng": {"torch": [1, 2]}},
                env=SimpleNamespace(scenario_tape=SimpleNamespace(seed=seed+int(wrong_seed), _streams={"used": True})))
            return row, plan

        shared = SimpleNamespace(episode=episode, qualify=Mock(), policy_trace=Mock(return_value=[]))
        source = SimpleNamespace(shared=shared, qualify_observer=Mock(), validate_continuations=Mock())
        key = spec.keys()[0]
        with tempfile.TemporaryDirectory() as temp, patch.object(spec, "ROOT", Path(temp)), \
                patch.object(spec, "native_inputs", return_value=(source, {}, "fixed_checkpoint")), \
                patch("torch.set_rng_state") as restore:
            path = spec.prefix_path("unit294", key)
            spec.write_json(path/"rows.json", [{"probe_index": 8, "original_choice": 0}])
            with (path/"prefix_refs.pkl").open("wb") as f:
                pickle.dump({"prefix": "archived_prefix", "snapshots": {8: {"rng": {"torch": [1, 2]}}}}, f)
            labels, receipt = spec.rollout("unit294", key, 8, (16, 17), [0, 1], path/"evaluation")
            self.assertEqual(receipt["native_episodes"], 6)
            self.assertEqual(receipt["exact_identities"], 2)
            self.assertEqual(restore.call_count, 6)
            self.assertEqual([c for c, _ in calls], [None, 0, 1]*2)
            self.assertEqual({s for _, s in calls}, {spec.future_seed(key, 8, d) for d in (16, 17)})
            np.testing.assert_allclose(labels[0]["values"]["1"], [.05, 1/12, .05-1/12, -.2])
            self.assertEqual(labels[0]["values"]["0"], [0., 0., 0., 0.])
            self.assertEqual(source.validate_continuations.call_args.kwargs["candidates"], [0, 1])
            wrong_seed = True
            with self.assertRaisesRegex(ValueError, "shared future"):
                spec.rollout("unit294", key, 8, (16,), [0, 1], path/"bad")

    def test_joint_success_needs_both_interval_types_and_both_halves(self):
        raw, proposal = np.zeros((1, 3), int), np.ones((1, 3), int)
        sealed = {"choices": np.stack((raw, raw, proposal, raw), axis=1), "proposal": proposal,
            "matched_thin_probabilities": np.zeros((1, 3))}
        panels = np.zeros((3, 32, 6, 4))
        panels[:, :, 1] = [-1, 0, -1, -.5]
        layout = np.arange(3).reshape(1, 3, 1)
        self.assertTrue(rule.summarize(sealed, panels, layout)["joint_transfer_supported"])
        panels[:, :16, 1, 0], panels[:, 16:, 1, 0] = 1, -3
        result = rule.summarize(sealed, panels, layout)
        self.assertTrue(result["primary_both_CI_passed"])
        self.assertFalse(result["primary_both_halves_passed"])
        self.assertFalse(result["joint_transfer_supported"])
        panels[:, :, 1, 0] = np.array([-100, 99, -2])[:, None]
        self.assertFalse(rule.summarize(sealed, panels, layout)["primary_both_CI_passed"])


if __name__ == "__main__":
    unittest.main()
