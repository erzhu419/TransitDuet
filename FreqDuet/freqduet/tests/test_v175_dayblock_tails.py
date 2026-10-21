import ast
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import preregister_freqduet_v175_dayblock_tails as stage


def model():
    base = dict(execution_source='f5d152f0bc',
        frozen=dict(point_source='1b783ea8f3', native_window_s=3600., tails=[.1, .1]),
        support=dict(minimum_calibration_days=19, low_support='permanent abstention'),
        reservation=dict(calibration_days=list(stage.CALIBRATION), actors=list(range(12)),
            sampler_seeds={str(d): 1650000000+i for i, d in enumerate(stage.CALIBRATION)}),
        conditioning=dict(candidate_key=['query_phase', 'query_time_stratum', 'native_stage_kind']),
        target=dict(joint_unit='same atom'), estimator=dict(day_mass_ranks='unchanged'),
        gates=dict(above_upper95_max=.1, below_upper95_max=.1, coverage_lower95_min=.8,
            strict_score_scopes=['all', 'clearance']), comparator={}, inference=dict(scopes=list(stage.prior.SCOPES)),
        isolation={})
    return dict(status='sealed', execution_source='a5b45df520', calibration_days=list(stage.CALIBRATION),
        actors=list(range(12)), frozen_protocol=base)


class DayBlockProtocolTests(unittest.TestCase):
    def register(self, source=None, fresh=None):
        with patch.dict('os.environ', FREQDUET_DIAGNOSTIC_COMMIT='test-source'):
            return stage.registration(model() if source is None else source,
                dict(collisions=[]) if fresh is None else fresh)

    def test_protocol_preserves_frozen_method_and_reserves_independent_days(self):
        source = model()
        before = deepcopy(source)
        result = self.register(source)
        self.assertEqual(source, before)
        for field in ('frozen', 'support', 'target', 'conditioning', 'estimator', 'gates'):
            self.assertEqual(result[field], source['frozen_protocol'][field])
        reservation = result['reservation']
        self.assertEqual(len(reservation['confirmation_days']), 64)
        self.assertFalse(set(reservation['confirmation_days']) & set(stage.CALIBRATION))
        self.assertFalse(set(reservation['confirmation_days']) & set(reservation['excluded_confirmation_days']))
        self.assertEqual(len(set(reservation['sampler_seeds'].values())), 128)
        self.assertFalse(any(result['execution'].values()))
        self.assertEqual(result['upper_tail_envelope']['candidate_count'], 1)
        self.assertIn('L_new=L_V172', result['upper_tail_envelope']['interval'])
        self.assertIn('never collapse duplicates', result['upper_tail_envelope']['repeated_days'])
        self.assertEqual(reservation['confirmation_paths'], 1536)
        self.assertIn('1536', result['isolation']['confirmation'])

    def test_collision_or_changed_window_blocks_registration(self):
        with self.assertRaises(AssertionError):
            self.register(fresh=dict(collisions=[3352001]))
        source = model()
        source['frozen_protocol']['frozen']['native_window_s'] = 7200.
        with self.assertRaises(AssertionError):
            self.register(source)

    def test_membership_uses_query_state_and_union_not_stacked_corrections(self):
        fields = model()['frozen_protocol']['conditioning']['candidate_key']
        self.assertEqual(stage.key_scopes(['clearance', 'hour13', 'moving-launch'], fields), stage.TARGETS)
        self.assertEqual(stage.key_scopes(['active', 'hour13', 'moving-launch'], fields), (stage.TARGETS[1],))
        self.assertEqual(stage.key_scopes(['active', 'hour2', 'moving-launch'], fields), ())

    def test_exact_label_boundary_and_score_decomposition_baseline(self):
        x = stage.metrics(100., 110., [90., 110.], [80., 120.], [90., 110.])
        self.assertEqual(x[0], 0.)
        self.assertEqual(x[2], 1.)
        self.assertEqual(x[6], -20.)
        self.assertEqual(x[8], 0.)
        y = stage.metrics(100., np.nextafter(110., np.inf), [90., 110.], [80., 120.], [90., 110.])
        self.assertEqual(y[0], 1.)

    def test_missing_and_duplicate_calibration_cells_are_not_dropped(self):
        rows = [dict(actor=1, day=d, finite_fraction=.9, **{m: float(d) for m in stage.METRICS})
            for d in stage.CALIBRATION]
        self.assertEqual(stage.summarize(rows, [1])['cells'], 64)
        for bad in (rows[:-1], rows+[rows[0]]):
            with self.assertRaises(AssertionError):
                stage.summarize(bad, [1])

    def test_freshness_checks_names_without_opening_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage.write(root/'inventory.json', dict(day_references={}, groups=[]))
            path = root/'capture/residuals'/f'1_{stage.CONFIRMATION[0]}_old.jsonl.gz'
            path.parent.mkdir(parents=True)
            path.touch()
            args = SimpleNamespace(inventory_file=root/'inventory.json', v172_root=root/'base/seal',
                out_root=root/'new/protocol', parent=[root], generator_root=[])
            with patch.object(stage.ledger.archives, 'result_roots', return_value=[root]):
                with self.assertRaisesRegex(AssertionError, 'collision'):
                    stage.freshness(args)

    def test_no_fit_collection_or_confirmation_outcome_entry(self):
        tree = ast.parse(Path(stage.__file__).read_text())
        calls = {(n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id)
            for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
        self.assertTrue(calls.isdisjoint({'fit_key', 'inverse_day_mass', 'quantile', 'percentile',
            'collect', 'rollout', 'train', 'attach_labels'}))
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr in
            ('v168_root', 'v173_root', 'v174_root') for n in ast.walk(tree)))


if __name__ == '__main__':
    unittest.main()
