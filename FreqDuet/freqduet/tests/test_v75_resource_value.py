from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import freqduet_resource_observation as resources
import run_freqduet_v75_resource_labels as panel
import run_freqduet_v75_resource_value as value
from test_v73_fleet_state import environment


def example():
    env = environment()
    env.timetables[2].launch_time = 900.
    snapshot = panel.previous.telemetry.capture_fleet(env)
    capture = dict(identity=(800., 1, 1, 4, True), observation=dict(target_headway_s=500., pending_commands=[]))
    return capture, snapshot


class ResourceValueTest(unittest.TestCase):
    def test_return_supply_flips_direction_and_excludes_controlled_vehicle(self):
        capture, snapshot = example()
        snapshot["fleet"][0].update(on_route=True, direction=False, active_trip_id=2, actual_launch_s=100.,
            progress_m=1000., terminal_return_s=None, state="DWELLING", remaining_command_hold_s=20., remaining_service_s=0.)
        capture["observation"]["pending_commands"] = [dict(bus_id=0, action_s=10.)]
        encoded = dict(zip(resources.FEATURES, resources.encode_resources(capture, snapshot)))
        self.assertEqual(encoded["origin_other_inbound"], 1 / 3)
        self.assertEqual(encoded["return_other_inbound"], 0.)
        self.assertEqual(encoded["origin_inbound_pending_hold"], 10 / 135)
        self.assertEqual(encoded["origin_inbound_nearest_remaining"], .5)
        self.assertEqual(encoded["return_ready"], 1 / 3)

    def test_vehicle_and_plan_order_do_not_change_resource_encoding(self):
        capture, snapshot = example()
        first = resources.encode_resources(capture, snapshot)
        snapshot["fleet"].reverse()
        snapshot["plans"].reverse()
        np.testing.assert_array_equal(first, resources.encode_resources(capture, snapshot))
        self.assertEqual(len(first), 38)

    def test_future_actual_launch_return_and_trip_ids_do_not_enter_features(self):
        capture, snapshot = example()
        first = resources.encode_resources(capture, snapshot)
        snapshot["plans"][1]["actual_launch_s"] = 999999.
        snapshot["fleet"][1]["predicted_return_s"] = -99999.
        snapshot["future_demand"] = [99999.]
        snapshot["plans"][1]["trip_id"] += 1000
        np.testing.assert_array_equal(first, resources.encode_resources(capture, snapshot))

    def test_due_and_future_plans_use_published_release_not_unobserved_actual(self):
        capture, snapshot = example()
        encoded = dict(zip(resources.FEATURES, resources.encode_resources(capture, snapshot)))
        self.assertEqual(encoded["origin_due"], 1 / 3)
        self.assertEqual(encoded["origin_next_release_wait"], 0.)
        self.assertEqual(encoded["return_plans_15m"], 1 / 3)
        self.assertEqual(encoded["return_next_release_wait"], 100 / 3600)

    def test_primary_retains_complete_original_state_and_adds_only_resources(self):
        data = dict(states=np.arange(12.).reshape(3, 4), resources=np.ones((3, 38)),
                    fleet_states=np.ones((3, 5)), plan_states=np.ones((3, 6)),
                    days=np.ones(3), cases=np.ones(3), outcomes=np.ones((3, 32, 7, 11)))
        np.testing.assert_array_equal(value.inputs(data, "structured")[:, :4], data["states"])
        self.assertEqual(value.inputs(data, "structured").shape, (3, 42))
        self.assertEqual(value.inputs(data, "full_fleet").shape, (3, 15))

    def test_fitting_excludes_all_validation_days_from_scaling_and_targets(self):
        rng = np.random.default_rng(75)
        data = dict(states=rng.normal(size=(6, 3)), resources=rng.normal(size=(6, 38)),
            native=np.arange(6), days=np.repeat([1, 2, 3], 2), outcomes=rng.normal(size=(6, 32, 7, 11)))
        first = value.fit_candidate(data, "structured", (1, 2))
        changed = deepcopy(data)
        for key in ("states", "resources", "outcomes"):
            changed[key][data["days"] == 3] = 1e9
        second = value.fit_candidate(changed, "structured", (1, 2))
        np.testing.assert_array_equal(first.center, second.center)
        np.testing.assert_array_equal(first.coefficients, second.coefficients)

    def test_new_day_and_future_panels_are_disjoint_and_keep_full_action_budget(self):
        old_days = set(panel.source.DEMAND_SEEDS)
        self.assertFalse(set(panel.TRAIN_DAYS) & set(panel.VALID_DAYS))
        self.assertFalse(old_days & set(panel.DAYS))
        self.assertEqual((len(panel.TRAIN_DAYS), len(panel.VALID_DAYS), len(panel.REFERENCES), len(panel.CONTEXTS), len(panel.ITEMS)), (32, 12, 176, 528, 16896))
        self.assertEqual(len(set(panel.future_seed(c, f) for c, f in panel.ITEMS)), len(panel.ITEMS))
        self.assertEqual(panel.joint.ACTIONS, (0, 5, 10, 15, 20, 30, 45))

    def test_gates_require_all_controls_and_every_joint_constraint(self):
        contrasts = {c: {m: {loss: dict(ci95=[-2., -1.]) for loss in ("mae", "mse")}
            for m in value.old.PRIMARY} for c in value.CONTROLS}
        effects = {m: dict(ci95=[-2., -1.]) for m in value.model.METRICS}
        self.assertTrue(all(value.gates(contrasts, effects).values()))
        contrasts["expanded44/full_fleet"]["cv_squared"]["mse"]["ci95"][1] = .01
        self.assertFalse(value.gates(contrasts, effects)["prediction_gate_passed"])
        effects["trips_incomplete"]["ci95"][1] = .01
        self.assertFalse(value.gates(contrasts, effects)["joint_effect_gate_passed"])


if __name__ == "__main__":
    unittest.main()
