import unittest
from argparse import Namespace

from scripts import mujoco_v14_19_bidirectional_router_adapter_screen_spec as spec
from scripts.analyze_mujoco_v14_19_bidirectional_router_adapter_screen import (
    analyze_payloads,
)
from scripts.submit_mujoco_v14_19_bidirectional_router_adapter_scheduleurm import (
    build_probe_command,
    build_scheduler_spec,
    selected_cells,
)


def _probe_payload(environment, blocked=False):
    baseline = 10.0
    candidates = []
    for strength in spec.ROUTER_STRENGTHS:
        distance = strength - spec.BASELINE_ROUTER_STRENGTH
        if environment == "Walker2d-v5":
            merit = baseline * (1.0 + distance)
        else:
            merit = baseline * (1.0 - distance)
        if blocked and strength != spec.BASELINE_ROUTER_STRENGTH:
            merit = baseline + abs(distance)
        candidates.append({
            "upper_gain": 1.0,
            "lower_gain": 1.0,
            "router_strength": strength,
            "reward_violation_count": 0,
            "frequency_violation_count": int(10 * merit),
            "frequency_violation_merit": merit,
            "worst_frequency_violation": merit / 2.0,
        })
    return {
        "profile": spec.PROFILE,
        "gains": list(spec.ACTOR_GAINS),
        "router_strengths": list(spec.ROUTER_STRENGTHS),
        "candidates": candidates,
    }


class MujocoV1419BidirectionalRouterAdapterScreenTest(unittest.TestCase):
    def _args(self):
        return Namespace(
            run_name="v14_19_unit",
            anchor_run_name=spec.ANCHOR_RUN_NAME,
            python_executable="python3",
            priority="normal",
            nodes=["node001", "node002", "node003", "node004", "node005", "node006"],
        )

    def test_launcher_uses_frozen_bidirectional_grid_and_dynamic_nodes(self):
        self.assertEqual(len(selected_cells()), 9)
        environment, seed = selected_cells()[0]
        command = build_probe_command(self._args(), environment, seed)
        self.assertIn(
            "--router-strengths 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
            command,
        )
        self.assertTrue(command.endswith("&& echo DONE"))
        scheduler = build_scheduler_spec(self._args(), environment, seed)
        self.assertIsNone(scheduler["require_node"])
        self.assertEqual(scheduler["cpu"], 1)
        self.assertEqual(set(scheduler["allowed_nodes"]), set(self._args().nodes))

    def test_one_selector_adapts_direction_and_requires_nine_of_nine(self):
        payloads = [
            (environment, seed, _probe_payload(environment))
            for environment, seed in selected_cells()
        ]
        result = analyze_payloads(payloads)
        self.assertEqual(
            result["status"],
            "bidirectional_router_adapter_mechanism_supported",
        )
        self.assertEqual(result["supported_cell_count"], 9)
        self.assertEqual(result["selected_direction_counts"], {
            "higher": 6,
            "lower": 3,
        })

        environment, seed, _ = payloads[-1]
        payloads[-1] = (environment, seed, _probe_payload(environment, blocked=True))
        result = analyze_payloads(payloads)
        self.assertEqual(
            result["status"],
            "bidirectional_router_adapter_mechanism_not_supported",
        )
        self.assertEqual(result["supported_cell_count"], 8)


if __name__ == "__main__":
    unittest.main()
