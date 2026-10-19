import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_freqduet_v129_terminal_recovery import transition, turnover_split


class RecoveryTests(unittest.TestCase):
    def chain(self, references, actuals, visible):
        rows = []
        for r, a, v in zip(references, actuals, visible):
            rows.append(transition(r, a, v, rows[-1] if rows else None))
        return rows

    def test_hard_gap_preserves_delay_without_recovery_slack(self):
        rows = self.chain([0., 360., 720., 1080.], [0., 460., 820., 1180.], [0., 460., 0., 0.])
        self.assertEqual([r['delay_s'] for r in rows], [0., 100., 100., 100.])
        self.assertEqual(sum(r['new_supply_s'] for r in rows), 100.)
        self.assertEqual(rows[-1]['hard_spacing_wait_s'], 100.)

    def test_longer_published_gap_consumes_inherited_delay(self):
        rows = self.chain([0., 360., 820.], [0., 460., 820.], [0., 460., 0.])
        self.assertEqual(rows[-1]['planned_slack_s'], 100.)
        self.assertEqual(rows[-1]['planned_recovery_s'], 100.)
        self.assertEqual(rows[-1]['delay_s'], 0.)

    def test_shorter_plan_adds_delay_without_new_vehicle_shortage(self):
        rows = self.chain([0., 300., 660.], [0., 360., 720.], [0., 0., 0.])
        self.assertEqual(rows[1]['short_plan_addition_s'], 60.)
        self.assertEqual(sum(r['new_supply_s'] for r in rows), 0.)
        self.assertEqual(rows[-1]['immediate_release_advance_s'], 60.)

    def test_same_vehicle_can_carry_prior_launch_delay(self):
        row = transition(600., 700., 700.)
        split = turnover_split(row, dict(prior_trip_id=0, first_dispatch_visible_s=700.),
            {0: dict(actual_s=100.)}, {0: dict(return_s=699., holding_s=50., nonholding_elapsed_s=549.)}, {0: 0.})
        self.assertEqual(split['prior_trip_delay_s'], 100.)
        self.assertEqual(split['new_supply_without_prior_lateness_s'], 0.)
        self.assertEqual(split['prior_lateness_exposure_s'], 100.)

    def test_unexplained_wait_is_retained_and_impossible_clock_rejected(self):
        self.assertEqual(transition(0., 10., 0.)['unexplained_s'], 10.)
        self.assertEqual(transition(0., 10., 0.)['hard_spacing_wait_s'], 0.)
        with self.assertRaises(AssertionError):
            transition(100., 100., 101.)


if __name__ == '__main__':
    unittest.main()
