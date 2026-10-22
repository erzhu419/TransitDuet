import unittest

import run_freqduet_v217_upper_rule_factorial as v
import audit_freqduet_v217_native_cycles as audit


class FactorialTests(unittest.TestCase):
    def test_two_shared_fixed_paths_per_day_not_ten_actor_repeats(self):
        self.assertEqual(len(v.ITEMS), 32)
        self.assertEqual(len(set(v.ITEMS)), 32)
        self.assertEqual(len(v.old.SEEDS), 10)

    def test_factorial_interaction_has_declared_direction(self):
        levels = {k: {m: x for m in v.old.METRICS} for k, x in
            dict(learned_rule=7., learned_zero=5., fixed_rule=9., fixed_zero=6.).items()}
        result = v.effects(levels)
        self.assertEqual(result['upper_with_rule']['journey_min'], -2.)
        self.assertEqual(result['interaction']['journey_min'], -1.)

    def test_positive_opportunity_requires_native_smaller_action_and_valid_evidence(self):
        q = dict(time_s=10., event_seq=3, snapshot_read_s=10., snapshot_event_seq=3,
            ready=[], active=[], plans=[], terminal_masked=False, forward_valid=True,
            follower_valid=True, base_bin=2, feasible_bins=[0, 1, 2])
        self.assertTrue(audit.eligible(q))
        for replacement in (dict(base_bin=0), dict(feasible_bins=[2]), dict(follower_valid=False),
                            dict(snapshot_event_seq=1), dict(terminal_masked=True)):
            self.assertFalse(audit.eligible(dict(q, **replacement)))

    def test_empty_cycles_are_retained(self):
        data = dict(capture=dict(fresh_upper=[dict(event_seq=2, time_s=10., trip_id=4, direction=True)],
                                 queries=[], events=[]),
            trace=dict(upper_callbacks=[dict(time_s=10., trip_id=4, commands=[{}], changes=[])],
                       actual_launches=[], vehicle_returns=[]))
        result = audit.summarize(data)
        self.assertEqual(result['native_cycles'], 1)
        self.assertEqual(result['cycles_with_positive'], 0)


if __name__ == '__main__':
    unittest.main()
