from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v174_upper_tails as audit


class TailDiagnosticsTests(unittest.TestCase):
    def record(self, truth=120.):
        return (dict(point_s=100.), dict(finite=True, reference_finite=True, secondary_finite=True,
            bounds=[80., 110.], reference_bounds=[60., 130.], secondary_bounds=[80., 110.]),
            dict(status='resolved', full_eta_s=truth))

    def test_exact_endpoint_label_not_reconstructed_truth(self):
        q, p, label = self.record(110.)
        x = dict(zip(audit.METRICS, audit.diagnostics(q, p, label)))
        self.assertEqual(x['above'], 0.)
        label['full_eta_s'] = np.nextafter(110., np.inf)
        self.assertEqual(audit.diagnostics(q, p, label)[0], 1.)

    def test_new_repaired_and_identical_reference_decomposition(self):
        inputs = self.record()
        before = deepcopy(inputs)
        x = dict(zip(audit.METRICS, audit.diagnostics(*inputs)))
        self.assertEqual((x['coarse_new_miss'], x['coarse_upper_delta_s']), (1., -20.))
        self.assertEqual((x['v167_new_miss'], x['v167_repaired_miss'], x['v167_score_delta_s']), (0., 0., 0.))
        q, p, label = inputs
        p = dict(p, bounds=p['reference_bounds'], reference_bounds=p['bounds'])
        y = dict(zip(audit.METRICS, audit.diagnostics(q, p, label)))
        self.assertEqual(y['coarse_repaired_miss'], 1.)
        self.assertEqual(inputs, before)

    def test_episode_weighting_not_query_pooling(self):
        x = audit.diagnostics(*self.record())
        y = audit.diagnostics(*self.record(100.))
        result = audit.collapse({(1,): np.r_[9., 9.*x], (2,): np.r_[1., y]})
        self.assertEqual(result['above'], .5)
        self.assertEqual(result['coarse_new_miss'], .5)

    def test_missing_finite_evidence_is_not_dropped(self):
        q, p, label = self.record()
        for bad_p, bad_label in ((dict(p, reference_finite=False), label),
                (p, dict(status='unresolved'))):
            with self.assertRaises(AssertionError):
                audit.diagnostics(q, bad_p, bad_label)
        with self.assertRaises(AssertionError):
            audit.collapse({})

    def test_descriptive_variance_partition(self):
        a = np.array([-1., 1.])[:, None]
        d = np.array([-2., 0., 2.])[None, :]
        x = a+d
        result = audit.variation(x)
        self.assertAlmostEqual(result['actor_variance'], 1.)
        self.assertAlmostEqual(result['day_variance'], 8./3.)
        self.assertEqual(result['interaction_variance'], 0.)
        self.assertIsNone(audit.variation(np.zeros((2, 3)))['fractions'])

    def test_shared_day_bootstrap_matches_existing_and_rejects_missing_cells(self):
        rows = [dict(actor=a, day=d, endpoint_reproduced=True,
            **{m: float(a+d) for m in audit.METRICS}) for a in (1, 2) for d in (10, 11, 12)]
        result = audit.summarize(rows, [1, 2], [10, 11, 12])
        frame = pd.DataFrame([dict(train_seed=r['actor'], eval_seed=r['day'], value=r['above']) for r in rows])
        draws = audit.saved.prior.hierarchical_bootstrap(frame, 'value', draws=5000)
        np.testing.assert_array_equal(result['estimates']['above']['ci95'], np.percentile(draws, [2.5, 97.5]))
        with self.assertRaises(AssertionError):
            audit.summarize(rows[:-1], [1, 2], [10, 11, 12])

    def test_saved_stream_reproduces_abstention_denominators_and_endpoints(self):
        q = dict(actor=1, day=10, mode='old', query_key=[1, 2, True], query_time_s=46800.,
            episode=[2, 4, 2, True], candidate_key=[1, 'old', 'clearance'], point_s=100.,
            denominator_eligible=True, native_supported=True,
            native_stage=dict(reason='not_applicable', kind='moving-launch'),
            condition_values=dict(actor=1, mode='old', query_phase='clearance'), observable=dict(hour=13))
        entry = dict(status='finite', day_count=64, qlow_s=-10., qhigh_s=10.)
        tables = ({(1, 'old', 'clearance', 'hour13'): entry},
            {(1, 'old', 'clearance'): dict(entry, qlow_s=-30., qhigh_s=30.)},
            {(1, 'old', 'clearance'): entry})
        protocol = dict(comparator=dict(key=['actor', 'mode', 'query_phase']),
            reservation=dict(sampler_seeds={'10': 123}))
        unsupported = dict(q, query_key=[3, 2, True], episode=[3, 5, 2, True],
            candidate_key=None, native_supported=False)
        rows = []
        for query in (q, unsupported):
            p = audit.saved.prediction(query, *tables, protocol)
            p['candidate_key'] = None if p['candidate_key'] is None else list(p['candidate_key'])
            rows.append((query, p, dict(status='resolved', full_eta_s=120.)))
        cells, counts = audit.saved.reduce_rows(rows)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = '1_10_old.jsonl.gz'
            for subdir, records in (
                    ('queries', [r[0] for r in rows]), ('interval_predictions', [r[1] for r in rows]),
                    ('residuals', [dict(query_key=r[0]['query_key'], query_time_s=r[0]['query_time_s'],
                        label=r[2]) for r in rows])):
                dest = root/'confirmation'/subdir/name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(dest, 'wt') as handle:
                    for record in records:
                        handle.write(json.dumps(record)+'\n')
            audit.write(root/'cells/1_10_old.json', dict(execution_source='0151b7c719',
                map_source=audit.saved.MAP_SOURCE, sampler_seed=123, cells=cells, scoring_counts=counts))
            result = audit.scan_path(SimpleNamespace(v173_root=root), protocol, tables, 1, 10, 'old')
            self.assertEqual([r['finite_fraction'] for r in result], [.5, .5])
            self.assertTrue(all(r['endpoint_reproduced'] and r['above'] == 1. for r in result))


if __name__ == '__main__':
    unittest.main()
