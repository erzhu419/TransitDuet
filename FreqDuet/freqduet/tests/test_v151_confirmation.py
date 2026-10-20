from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v151_confirmation as audit


class ConfirmationReadoutTests(unittest.TestCase):
    def test_point_channels_close_with_shared_initial_state(self):
        points = dict(v149=100., events=90., joint=80.)
        result = audit.attribution(points, -3.)
        self.assertEqual(result, dict(event_shift_s=-10., current_residence_shift_s=-3.,
            downstream_shift_s=-7., boundary_shift_s=0.))

    def test_opaque_baseline_stays_a_whole_point_shift(self):
        result = audit.attribution(dict(v149=100., events=100., joint=80.))
        self.assertEqual(result['boundary_shift_s'], -20.)
        self.assertEqual(result['current_residence_shift_s'], 0.)
        self.assertEqual(result['downstream_shift_s'], 0.)

    def test_absolute_cell_bias_is_not_mean_absolute_error(self):
        rows = [dict(episode=[i], joint_bias_s=j, v149_bias_s=5.) for i, j in enumerate((10., -10.))]
        self.assertEqual(audit.describe(rows)['metrics']['joint_delta_absolute_bias_s'], -5.)

    def test_partition_budget_uses_original_episode_weights(self):
        def row(episode, phase, mechanism, shift):
            values = dict.fromkeys(audit.CHANNELS, 0.)
            values['downstream_shift_s'] = shift
            return dict(episode=[episode], values=values, labels=['scope/'+phase, 'mechanism/'+mechanism])
        rows = [row(1, 'active', 'downstream_only', -2.), row(1, 'clearance', 'downstream_only', -6.),
            row(2, 'active', 'downstream_only', 10.)]
        result = audit.budgets(rows)
        self.assertEqual(sum(v['metrics']['downstream_shift_s'] for k, v in result.items() if k.startswith('scope/')), 3.)
        self.assertEqual(sum(v['metrics']['fraction'] for k, v in result.items() if k.startswith('mechanism/')), 1.)


if __name__ == '__main__':
    unittest.main()
