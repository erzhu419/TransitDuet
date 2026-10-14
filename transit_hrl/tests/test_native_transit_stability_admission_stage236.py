import copy
import json
import unittest
from unittest.mock import patch

from freq_hrl.domains.transit import native_stability_admission as admission
from scripts import run_native_transit_stability_admission_stage236 as spec
from scripts.submit_native_transit_stability_admission_stage236_scheduleurm import task_specification
from tests import test_native_transit_prefix_stability_stage235 as fixtures

diagnostic, joint = admission.diagnostic, admission.joint


def gate_fixture(candidate='1@0.25'):
    curve = {'joint_score': -.1, 'by_root': {str(r): {'cases': 6, 'mean_delta_cost': -.1,
        'mean_delta_wait_min': -.2, 'mean_delta_unsafe': 0.} for r in (397, 401)}}
    return {'full': {m: {'candidate': candidate, 'curve': copy.deepcopy(curve)} for m in admission.MODES},
        'future_halves': [{'modes': {m: {'candidate': '16@1', 'fixed_curve': copy.deepcopy(curve)} for m in admission.MODES}} for _ in range(2)],
        'deletions': [{'modes': {m: {'candidate': '3@1', 'fixed_curve': copy.deepcopy(curve)} for m in admission.MODES}} for _ in range(10)]}


def cell_fixture(train, held, partition, direction):
    source, curves, ledger = fixtures.source_for(train, held, partition, direction)
    for rows in source['scopes'].values():
        for row in rows:
            ds = row['decisions']
            for name in admission.FROZEN: ds.setdefault(name, dict(ds['raw']))
            for d in ds.values(): d['changed'] = d['selected'] != row['original_choice']
    with patch.object(joint, 'collect_paths', return_value=(curves, ledger, {'max_abs_difference': 0., 'changed_choices': 0})):
        stable = spec.previous.fit_cell(train, {}, source, {})
    return source, stable


class StabilityAdmissionTest(unittest.TestCase):
    def test_separate_future_prefix_and_joint_fixed_candidate_admission(self):
        d = gate_fixture()
        d['future_halves'][0]['modes']['mean_selected']['fixed_curve']['by_root']['401']['mean_delta_cost'] = .001
        x = admission.decide(d, 'mean_selected')
        self.assertEqual(x['accepted'], {'future_only': False, 'prefix_only': True, 'joint': False})
        d = gate_fixture()
        d['deletions'][4]['modes']['mean_selected']['fixed_curve']['by_root']['397']['mean_delta_wait_min'] = .001
        x = admission.decide(d, 'mean_selected')
        self.assertEqual(x['accepted'], {'future_only': True, 'prefix_only': False, 'joint': False})
        self.assertEqual(admission.decide(gate_fixture(), 'mean_selected')['accepted'], dict.fromkeys(admission.FILTERS, True))
        self.assertEqual(x['candidate'], '1@0.25')

    def test_zero_and_roundoff_boundaries_safe_risk_and_pooled_strictness(self):
        d = gate_fixture()
        for family in ('future_halves', 'deletions'):
            for block in d[family]:
                for mode in admission.MODES:
                    for x in block['modes'][mode]['fixed_curve']['by_root'].values():
                        x['mean_delta_cost'], x['mean_delta_wait_min'] = 0., 7e-17
        for mode in admission.MODES: self.assertTrue(admission.decide(d, mode)['accepted']['joint'])
        d['future_halves'][0]['modes']['safe_selected']['fixed_curve']['by_root']['401']['mean_delta_unsafe'] = 1/60
        self.assertFalse(admission.decide(d, 'safe_selected')['accepted']['future_only'])
        self.assertTrue(admission.decide(d, 'mean_selected')['accepted']['future_only'])
        d['full']['mean_selected']['curve']['joint_score'] = 0.
        self.assertFalse(any(admission.decide(d, 'mean_selected')['accepted'].values()))
        for candidate in (None, 'zero'):
            self.assertFalse(any(admission.decide(gate_fixture(candidate), 'mean_selected')['accepted'].values()))

    def test_accept_reuses_exact_candidate_reject_reuses_exact_raw(self):
        rows, registered = fixtures.fixtures.old.old.old.fixtures.cohort.fixture()
        views = spec.training.cohort.source.matched_views(rows, registered)
        train, held = spec.training.cohort.split_rows(views['mixed_b'], 0)
        source, stable = cell_fixture(train, held, 'first_last', 0)
        before = copy.deepcopy(source)
        cell = admission.admit_cell(source, stable)
        for scope, rs in cell['scopes'].items():
            for r, old in zip(rs, source['scopes'][scope]):
                for a in joint.ANCHORS:
                    for m in admission.MODES:
                        for f in admission.FILTERS:
                            chosen = admission.control_name(a, m) if cell['admission'][a][m]['accepted'][f] else 'raw'
                            self.assertEqual(r['decisions'][admission.variant_name(a, m, f)], old['decisions'][chosen])
                for name in admission.FROZEN: self.assertEqual(r['decisions'][name], old['decisions'][name])
        self.assertEqual(source, before)
        self.assertTrue(cell['admission']['raw']['mean_selected']['accepted']['future_only'])
        self.assertFalse(cell['admission']['raw']['mean_selected']['accepted']['prefix_only'])
        self.assertTrue(cell['admission']['raw']['safe_selected']['accepted']['joint'])

    def test_outer_labels_blind_candidate_and_score_binding_rejection(self):
        rows, registered = fixtures.fixtures.old.old.old.fixtures.cohort.fixture()
        views = spec.training.cohort.source.matched_views(rows, registered)
        train, held = spec.training.cohort.split_rows(views['mixed_a'], 0)
        source, stable = cell_fixture(train, held, 'first_last', 0)
        gates = {a: {m: admission.decide(stable['anchors'][a], m) for m in admission.MODES} for a in joint.ANCHORS}
        altered = copy.deepcopy(source)
        for row in altered['scopes']['held_other']:
            for name, d in row['decisions'].items():
                if name != 'raw': d['cost'] += 100.
        with self.assertRaisesRegex(ValueError, 'outer physical scores changed'): admission.admit_cell(altered, stable)
        rebound = copy.deepcopy(stable)
        rs = altered['scopes']['held_other']
        for a in joint.ANCHORS:
            for m in admission.MODES:
                rebound['outer'][a][m]['held_other'] = {str(root): diagnostic.outer_stats([r for r in rs if r['root'] == root],
                    admission.control_name(a, m)) for root in {r['root'] for r in rs}}
        after = admission.admit_cell(altered, rebound)
        self.assertEqual(after['admission'], gates)
        self.assertNotEqual(after['scopes']['held_other'], admission.admit_cell(source, stable)['scopes']['held_other'])
        mismatched = copy.deepcopy(source)
        mismatched['selection']['raw']['mean_level'] = '16@1'
        with self.assertRaisesRegex(ValueError, 'fixed candidates differ'): admission.admit_cell(mismatched, stable)

    def test_complete_summary_all_scopes_sources_and_action_effects(self):
        rows, registered = fixtures.fixtures.old.old.old.fixtures.cohort.fixture()
        views = spec.training.cohort.source.matched_views(rows, registered)
        cells = {}
        for key in spec.KEYS:
            train, held = spec.training.cohort.split_rows(views[key[0]], key[1])
            source, stable = cell_fixture(train, held, key[2], key[3])
            cells[key] = admission.admit_cell(source, stable)
        report = spec.summarize(cells)
        self.assertEqual(report['source_reproduction'], {'candidate_comparisons': 128, 'choice_comparisons': 207360,
            'score_comparisons': 414720, 'changed_choices': 0, 'max_abs_difference': 0., 'outer_metric_records': 768})
        self.assertEqual(len(admission.VARIANTS), 39)
        for v in report['views'].values():
            for scope in spec.SCOPES:
                self.assertEqual(v['scopes'][scope]['prediction_cases'], 480)
                self.assertEqual(v['scopes'][scope]['unique_states'], 120)
                self.assertEqual(set(v['scopes'][scope]['variants']), set(admission.VARIANTS))
            counts = v['admission_counts']['raw']['mean_selected']['strategies']
            self.assertEqual(counts['future_only'], {'accepted_cells': 8, 'effectful_held_cells': 8})
            self.assertEqual(counts['joint'], {'accepted_cells': 0, 'effectful_held_cells': 0})
            variants = v['scopes']['held_other']['variants']
            self.assertEqual(variants[admission.variant_name('raw', 'mean_selected', 'joint')], variants['raw'])
        self.assertEqual(report['mixed_pooled']['scopes']['held_other']['unique_states'], 240)
        cells[spec.KEYS[0]]['scopes']['held_other'].pop()
        with self.assertRaisesRegex(ValueError, 'Dropped admission scope/state'): spec.summarize(cells)

    def test_admission_without_outer_action_change_is_not_counted_as_effectful(self):
        rows, registered = fixtures.fixtures.old.old.old.fixtures.cohort.fixture()
        views = spec.training.cohort.source.matched_views(rows, registered)
        train, held = spec.training.cohort.split_rows(views['root_401'], 0)
        source, stable = cell_fixture(train, held, 'first_last', 0)
        for scope, rs in source['scopes'].items():
            for r in rs:
                r['decisions'][admission.control_name('raw', 'safe_selected')] = dict(r['decisions']['raw'])
            stable['outer']['raw']['safe_selected'][scope] = {str(root): diagnostic.outer_stats([r for r in rs if r['root'] == root],
                admission.control_name('raw', 'safe_selected')) for root in {r['root'] for r in rs}}
        cell = admission.admit_cell(source, stable)
        key = ('root_401', 0, 'first_last', 0)
        with patch.object(spec.admission, 'metrics', return_value={}):
            counts = spec.aggregate([(key, cell)])['admission_counts']['raw']['safe_selected']['strategies']
        self.assertEqual(counts['joint'], {'accepted_cells': 1, 'effectful_held_cells': 0})

    def test_budget_protocol_and_unbound_linux_scheduler(self):
        c = spec.contract()
        self.assertEqual(c, json.loads(json.dumps(c)))
        for key in ('network_updates', 'new_raw_reference_fits', 'new_residual_fits', 'new_native_episodes', 'new_outer_prediction_calls'):
            self.assertEqual(c['budget'][key], 0)
        with patch.object(spec.previous, 'contract', wraps=spec.previous.contract) as parent:
            spec.contract()
            self.assertEqual(parent.call_count, 1)
        task = task_specification('unit_stability_admission')
        self.assertEqual((task['cpu'], task['ram_mb']), (1, 3072))
        self.assertIsNone(task['require_node'])
        self.assertEqual(task['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
        self.assertNotIn('_raw', task['result_dir'])


if __name__ == '__main__': unittest.main()
