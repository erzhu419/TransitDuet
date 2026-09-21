import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.experiments.pointmaze_exogenous_routing_attribution import (
    POINTMAZE_EXOGENOUS_ROUTING_ALGORITHM_PATH,
    POINTMAZE_EXOGENOUS_ROUTING_METHODS,
    POINTMAZE_EXOGENOUS_ROUTING_PROTOCOL_VERSION,
    build_pointmaze_exogenous_routing_model,
    exogenous_routing_method_spec,
    main,
)
from freq_hrl.experiments.pointmaze_exogenous_validation import (
    pointmaze_exogenous_dimensions,
)
from freq_hrl.experiments.pointmaze_goal_validation import DEFAULT_ENV_ID


class PointMazeExogenousRoutingStageSixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.time_scale = PhysicalTimeScaleContract(
            dt_seconds=0.01,
            upper_period_seconds=0.25,
            history_seconds=0.32,
            fast_period_seconds=0.04,
        )

    def test_method_matrix_is_the_preregistered_factorial_and_controls(self):
        self.assertEqual(len(POINTMAZE_EXOGENOUS_ROUTING_METHODS), 8)
        self.assertEqual(
            exogenous_routing_method_spec("flat_exogenous_multiscale_all"),
            ("flat", "multiscale_all"),
        )
        self.assertEqual(
            exogenous_routing_method_spec("hrl_exogenous_multiscale_routed"),
            ("hrl", "multiscale_routed_masked"),
        )
        self.assertEqual(
            exogenous_routing_method_spec("hrl_exogenous_multiscale_swapped"),
            ("hrl", "multiscale_swapped_masked"),
        )

    def test_representation_only_changes_inputs_not_initial_models(self):
        def assert_nested_equal(reference, candidate, path="state"):
            self.assertIs(type(reference), type(candidate), msg=path)
            if isinstance(reference, dict):
                self.assertEqual(reference.keys(), candidate.keys(), msg=path)
                for name in reference:
                    assert_nested_equal(
                        reference[name], candidate[name], f"{path}.{name}"
                    )
            elif torch.is_tensor(reference):
                self.assertTrue(
                    torch.equal(reference, candidate),
                    msg=f"initialization differs at {path}",
                )
            elif isinstance(reference, np.ndarray):
                np.testing.assert_array_equal(reference, candidate)
            else:
                self.assertEqual(reference, candidate, msg=path)

        models = {}
        capacities = {}
        for method in POINTMAZE_EXOGENOUS_ROUTING_METHODS:
            _, representation = exogenous_routing_method_spec(method)
            dimensions = pointmaze_exogenous_dimensions(
                env_id=DEFAULT_ENV_ID,
                horizon=64,
                time_scale=self.time_scale,
                representation=representation,
            )
            self.assertEqual(
                (dimensions.flat, dimensions.upper, dimensions.lower),
                (134, 134, 134),
            )
            model, capacity = build_pointmaze_exogenous_routing_model(
                method=method,
                dimensions=dimensions,
                reference_hidden_dim=128,
                learning_rate=3e-4,
                optimizer_seed=184001,
            )
            models[method] = model
            capacities[method] = capacity

        for architecture in ("flat", "hrl"):
            methods = [
                method for method in POINTMAZE_EXOGENOUS_ROUTING_METHODS
                if exogenous_routing_method_spec(method)[0] == architecture
            ]
            reference = models[methods[0]].state_dict()
            for method in methods[1:]:
                candidate = models[method].state_dict()
                assert_nested_equal(reference, candidate, architecture)
                self.assertEqual(capacities[methods[0]], capacities[method])

    def test_dry_run_records_routing_contract_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "protocol.json"
            status = main([
                "--methods",
                "flat_exogenous_history",
                "hrl_exogenous_multiscale_routed",
                "--iterations", "2",
                "--horizon", "64",
                "--optimizer-seed", "184001",
                "--train-seeds", "1790011",
                "--selection-seeds", "1790101",
                "--eval-seeds", "1790211",
                "--output", str(output),
                "--dry-run",
            ])
            self.assertEqual(status, 0)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["status"], "dry_run")
            self.assertEqual(
                payload["protocol"]["protocol_version"],
                POINTMAZE_EXOGENOUS_ROUTING_PROTOCOL_VERSION,
            )
            self.assertEqual(
                payload["protocol"]["algorithm_path"],
                POINTMAZE_EXOGENOUS_ROUTING_ALGORITHM_PATH,
            )
            self.assertEqual(payload["cells"], [])


if __name__ == "__main__":
    unittest.main()
