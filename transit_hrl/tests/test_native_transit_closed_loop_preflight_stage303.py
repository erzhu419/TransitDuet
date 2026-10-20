import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_closed_loop_preflight_stage303 as spec
from scripts import submit_native_transit_closed_loop_preflight_stage303_scheduleurm as submit


class ClosedLoopPreflightTests(unittest.TestCase):
    def test_prior_choices_are_applied_and_copied_before_later_decision(self):
        class Base:
            def __init__(self):
                self.decisions = []; self.probe_index = 1; self.probe = {}
            def select_candidate(self, candidates): return 0
        history = {0: 2}; cls = spec.replay_plan_class(Base, history); history[0] = 3
        plan = cls()
        self.assertEqual(plan.select_candidate([]), 2)
        plan.decisions.append(2)
        self.assertEqual(plan.select_candidate([]), 0)
        self.assertEqual(plan.applied_choices, {0: 2, 1: 0})

    def test_scene_and_inner_future_namespaces_are_fresh_and_bounded(self):
        scenes = {spec.scene_seed(r, s) for r in spec.source.ROOTS for s in spec.source.SCENARIOS}
        inner = {spec.future_seed(r, s, p, d) for r in spec.source.ROOTS for s in spec.source.SCENARIOS for p in spec.PROBES for d in range(16)}
        old_keys = spec.parent.keys()+spec.parent.keys("pilot")+spec.source.previous.keys()+spec.source.keys()
        self.assertEqual(len(scenes), 10); self.assertEqual(len(inner), 320)
        self.assertFalse(scenes & {spec.source.scene_seed(k) for k in old_keys})
        self.assertFalse(inner & {spec.source.future_seed(k, p, d) for k in old_keys for p in spec.source.PROBES for d in range(64)})
        self.assertFalse(scenes & inner); self.assertLess(max(inner), 2**31)

    def test_inner_future_replaces_tape_only_at_current_probe(self):
        class Tape:
            def __init__(self, seed, version=1): self.seed, self.version = seed, version
        class Env:
            scenario_tape = Tape(7); current_time = 10.
        class Base:
            def __init__(self):
                self.decisions = []; self.probe_index = 1; self.probe = {}; self.env = Env()
            def select_candidate(self, candidates): return 0
        plan = spec.replay_plan_class(Base, {0: 2}, continuation_seed=13)()
        plan.select_candidate([]); self.assertEqual(plan.env.scenario_tape.seed, 7)
        plan.decisions.append(2); plan.select_candidate([])
        self.assertEqual(plan.env.scenario_tape.seed, 13)
        self.assertEqual(plan.probe["continuation_start_time_s"], 10.)

    def test_frozen_cutoffs_preserve_service_and_joint_selection(self):
        raw = np.zeros((16, 1), int); proposal = np.ones_like(raw)
        screen = np.zeros((1, 8, 6, 4)); screen[:, :, 1, 1] = -.1; screen[:, :, 1, 2] = -.2
        registration = {"contract": {"locked_policy": {"service": {"threshold": -.15}, "joint": {"threshold": -.15}}}}
        with patch.object(spec.parent.native, "controls", return_value=(raw, raw, np.zeros((16, 1, 6)))), \
                patch.object(spec.parent.native, "proposals", return_value=proposal), \
                patch.object(spec.parent.learning, "features", return_value=np.zeros((16, 1, 26))), \
                patch.object(spec.parent.learning, "predict", return_value=np.full((16, 1), .1)), \
                patch.object(spec.source, "read", return_value=registration):
            self.assertEqual(spec.choose("unit", "raw", {}, None, None), 0)
            self.assertEqual(spec.choose("unit", "service_only", {}, screen, screen), 1)
            self.assertEqual(spec.choose("unit", "joint_budgeted", {}, screen, screen), 0)

    def test_pilot_is_unpinned_and_has_no_training(self):
        job = submit.task("unit_stage303", 397, "low_noise", "service_only")
        self.assertEqual((job["cpu"], job["ram_mb"], job["vram"]), (1, 1536, 0))
        self.assertIsNone(job["require_node"])
        self.assertEqual(job["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertIn("unit_stage303/inputs", job["stage_input_paths"][-1])
        with patch.object(spec.source, "read", return_value={}):
            contract = spec.contract()
        self.assertEqual(contract["budget"]["native_episodes"], 10*5+20*227)
        self.assertEqual(contract["actor_updates"], 0)


if __name__ == "__main__": unittest.main()
