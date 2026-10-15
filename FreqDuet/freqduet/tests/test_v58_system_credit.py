from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from freqduet_system_credit import Command, Decision, SystemCreditStream, observable_decision


def observation(time, bus=1, trip=10):
    return Decision(time, bus, trip, 3, True, (0.5, 0.2), 360, 1, 2)


class SystemCreditTest(unittest.TestCase):
    def test_overlapping_vehicle_lifetimes_do_not_duplicate_cost(self):
        stream = SystemCreditStream()
        stream.decide(observation(0), 15)
        stream.advance(0, 10, 2, 3, .5, 1, 1)
        stream.acknowledge([Command(0, 1, 10, 3, 15)])
        stream.decide(observation(10, bus=2, trip=20), 0)
        stream.advance(10, 10, 2, 3, .5, 1, 1)
        stream.acknowledge([Command(10, 2, 20, 3, 0)])
        stream.finish(20, service_cleared=False, unserved=2, lf_unserved=.5)
        audit = stream.audit()
        self.assertEqual(audit["exposure"]["waiting_person_s"], 40)
        self.assertEqual(audit["exposure"]["onboard_person_s"], 60)
        self.assertEqual(audit["cross_trip_bootstraps"], 1)
        self.assertAlmostEqual(audit["reward_sum"], -100 / 3600)

    def test_simultaneous_decisions_see_prior_pending_commands_only(self):
        stream = SystemCreditStream()
        first = stream.decide(observation(0), 15)
        second = stream.decide(observation(0, bus=2, trip=20), 45)
        self.assertEqual(first.pending_commands, ())
        self.assertEqual(second.pending_commands, (Command(0, 1, 10, 3, 15),))
        self.assertEqual(stream.transitions[0].reward, 0)
        self.assertEqual(stream.transitions[0].bootstrap_discount, 1)
        stream.advance(0, 1, 0, 0, 0, 0, 0)
        stream.acknowledge([Command(0, 2, 20, 3, 45), Command(0, 1, 10, 3, 15)])
        stream.decide(observation(1, bus=3, trip=30), 0)
        self.assertEqual(stream.active[0].pending_commands, ())

    def test_trip_return_does_not_terminate_system_bootstrap(self):
        stream = SystemCreditStream()
        stream.decide(observation(0), 0)
        stream.advance(0, 5, 1, 0, 1, 0, 0)
        stream.acknowledge([Command(0, 1, 10, 3, 0)])
        stream.returned(1, 10, 4)
        stream.decide(observation(5, bus=2, trip=20), 15)
        stream.advance(5, 5, 1, 0, 1, 0, 0)
        stream.acknowledge([Command(5, 2, 20, 3, 15)])
        stream.finish(10, service_cleared=True, unserved=1, lf_unserved=1)
        self.assertEqual([t.done for t in stream.transitions], [False, True])
        self.assertEqual([t.bootstrap_discount for t in stream.transitions], [1, 0])
        self.assertEqual(stream.audit()["return_crossing_bootstraps"], 1)

    def test_uncontrolled_prefix_and_censored_unserved_tail_are_conserved(self):
        stream = SystemCreditStream()
        stream.advance(0, 5, 2, 0, .5, 0, 0)
        stream.decide(observation(5), 0)
        stream.advance(5, 5, 2, 0, .5, 0, 0)
        stream.acknowledge([Command(5, 1, 10, 3, 0)])
        stream.finish(20, service_cleared=True, unserved=2, lf_unserved=.5)
        stream.audit()
        np.testing.assert_array_equal(stream.prefix, [10, 0, 2.5, 0, 0])
        np.testing.assert_array_equal(stream.transitions[0].exposure, [30, 0, 7.5, 0, 0])
        np.testing.assert_array_equal(stream.total, [40, 0, 10, 0, 0])

    def test_future_or_repeated_tick_cannot_be_charged(self):
        stream = SystemCreditStream()
        stream.advance(0, 1, 1, 0, .5, 0, 0)
        with self.assertRaisesRegex(ValueError, "recorded once"):
            stream.advance(0, 1, 1, 0, .5, 0, 0)

    def test_truncated_service_cannot_fake_a_zero_tail(self):
        with self.assertRaisesRegex(ValueError, "truncated"):
            SystemCreditStream().finish(10, service_cleared=False, unserved=0, lf_unserved=0)

    def test_action_projection_mismatch_blocks_candidate_data(self):
        stream = SystemCreditStream()
        stream.decide(observation(0), 45)
        stream.advance(0, 1, 0, 0, 0, 0, 0)
        with self.assertRaisesRegex(ValueError, "physical execution"):
            stream.acknowledge([Command(0, 1, 10, 3, 15)])

    def test_lf_hf_split_preserves_total_exposure(self):
        stream = SystemCreditStream()
        stream.advance(0, 6, 3, 4, 1, 2, 0)
        stream.finish(6, service_cleared=False, unserved=3, lf_unserved=1)
        stream.audit()
        wait, onboard, low_wait, low_onboard, _ = stream.total
        self.assertEqual(low_wait + low_onboard, 18)
        self.assertEqual(wait + onboard - low_wait - low_onboard, 24)

    def test_invalid_frequency_ownership_is_not_silently_clipped(self):
        with self.assertRaisesRegex(ValueError, "total/LF"):
            SystemCreditStream().advance(0, 1, 1, 0, 2, 0, 0)

    def test_nonterminal_clock_discount_does_not_depend_on_number_of_buses(self):
        stream = SystemCreditStream()
        stream.decide(observation(0), 0)
        stream.advance(0, 3600, 1, 0, 1, 0, 0)
        stream.acknowledge([Command(0, 1, 10, 3, 0)])
        stream.decide(observation(3600, bus=2, trip=20), 0)
        self.assertEqual(stream.transitions[0].bootstrap_discount, 1)
        self.assertEqual(stream.transitions[0].reward, -1)

    def test_latent_queues_and_future_returns_cannot_enter_decision_state(self):
        bus = NS(trip_id=10, direction=True, future_return_time=9000)
        env = NS(current_time=5, latent_queue=500, _ready_vehicle_count=lambda direction: 1 if direction else 2)
        runner = NS(env=env, _bus_for_agent=lambda key: bus,
                    _pending_lower_action_context={1: NS(target_headway_s=360)},
                    _augment_lower_state=lambda obs, previous: [*obs, previous])
        before = observable_decision(runner, [1, 3, .2], 1, 15)
        env.latent_queue = 900
        bus.future_return_time = 5000
        self.assertEqual(before, observable_decision(runner, [1, 3, .2], 1, 15))


if __name__ == "__main__":
    unittest.main()
