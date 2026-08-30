import unittest
from argparse import Namespace

from scripts import mujoco_v14_18_router_mechanism_screen_spec as spec
from scripts.analyze_mujoco_v14_18_router_mechanism_screen import analyze_payloads
from scripts.submit_mujoco_v14_18_router_probe_scheduleurm import (
    build_probe_command,
    build_scheduler_spec,
    selected_cells,
)


def _probe_payload(seed_offset=0.0, failing_strength=None):
    candidates = []
    baseline = 10.0 + seed_offset
    for strength in spec.ROUTER_STRENGTHS:
        merit = baseline if strength == 0.5 else baseline * (1.0 - strength / 10.0)
        candidates.append({
            "upper_gain": 1.0,
            "lower_gain": 1.0,
            "router_strength": strength,
            "reward_violation_count": int(strength == failing_strength),
            "frequency_violation_count": int(100 * merit),
            "frequency_violation_merit": merit,
            "worst_frequency_violation": merit / 2.0,
        })
    return {
        "profile": spec.PROFILE,
        "gains": list(spec.ACTOR_GAINS),
        "router_strengths": list(spec.ROUTER_STRENGTHS),
        "candidates": candidates,
    }


class MujocoV1418RouterMechanismScreenTest(unittest.TestCase):
    def _args(self):
        return Namespace(
            run_name="v14_18_unit",
            anchor_run_name=spec.ANCHOR_RUN_NAME,
            python_executable="python3",
            priority="normal",
            nodes=["node001", "node002", "node003", "node004", "node005", "node006"],
        )

    def test_launcher_is_dynamic_unpinned_and_stages_one_anchor(self):
        self.assertEqual(len(selected_cells()), 9)
        environment, seed = selected_cells()[0]
        payload = build_scheduler_spec(self._args(), environment, seed)
        self.assertIsNone(payload["require_node"])
        self.assertEqual(payload["cpu"], 1)
        self.assertEqual(payload["ram_mb"], 768)
        self.assertEqual(set(payload["allowed_nodes"]), set(self._args().nodes))
        self.assertEqual(len(payload["stage_input_paths"]), 1)
        self.assertEqual(len(payload["wait_for_files"]), 2)
        self.assertTrue(payload["reroute_on_node_down"])

    def test_command_freezes_router_only_grid_and_done_marker(self):
        environment, seed = selected_cells()[0]
        command = build_probe_command(self._args(), environment, seed)
        self.assertIn("--gains 1.0", command)
        self.assertIn("--router-strengths 0.5,0.6,0.7,0.8,0.9,1.0", command)
        self.assertIn("--profile v14_17_anchor", command)
        self.assertTrue(command.endswith("&& echo DONE"))

    def test_analysis_nominates_one_global_strength_only_on_nine_of_nine(self):
        payloads = [
            (environment, seed, _probe_payload(index / 100.0))
            for index, (environment, seed) in enumerate(selected_cells())
        ]
        result = analyze_payloads(payloads)
        self.assertEqual(result["status"], "global_router_strength_nominated")
        self.assertEqual(result["nominated_global_router_strength"], 1.0)
        self.assertEqual(result["untouched_cell_count"], 8)

        environment, seed, _ = payloads[-1]
        payloads[-1] = (environment, seed, _probe_payload(failing_strength=1.0))
        result = analyze_payloads(payloads)
        self.assertNotEqual(result["nominated_global_router_strength"], 1.0)

        blocked = _probe_payload()
        for candidate in blocked["candidates"]:
            if candidate["router_strength"] != spec.BASELINE_ROUTER_STRENGTH:
                candidate["reward_violation_count"] = 1
        payloads[-1] = (environment, seed, blocked)
        result = analyze_payloads(payloads)
        self.assertEqual(result["status"], "no_global_router_strength_nominated")
        self.assertIsNone(result["nominated_global_router_strength"])

    def test_analysis_rejects_incomplete_cell_registry(self):
        with self.assertRaisesRegex(ValueError, "cell registry mismatch"):
            analyze_payloads([
                (environment, seed, _probe_payload())
                for environment, seed in selected_cells()[:-1]
            ])


if __name__ == "__main__":
    unittest.main()
