import contextlib
import gzip
import io
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import controllers
import freqduet_flow_controller as flow
from freqduet_arrival_metric_control import ArrivalMetricGraph, choose_current
from freqduet_consistent_flow import ConsistentFlowController
from freqduet_flow_controller import FlowController
from freqduet_indexed_continuation import IndexedNativeGraph
from profile_freqduet_v277_solver import load_module
from test_fast_continuation import fixture

entry = load_module(Path(__file__).with_name('run.py'), 'accelerated_public_entry')
acceptance = load_module(Path(__file__).with_name('final_acceptance.py'), 'accelerated_final_acceptance')


def recorded_gaps(native):
    keys = {(d, s.station) for d, route in native.snapshot.routes.items() for s in route if not s.terminal}
    last, gaps = {}, []
    for direction, station, _, now, _ in native.observed['arrivals']:
        key = direction, station
        if key not in keys:
            continue
        if key in last:
            gaps.append(now-last[key])
        last[key] = now
    return gaps


class EntryTests(unittest.TestCase):
    def prepared(self, name, native):
        controller = object.__new__(controllers.CONTROLLERS[name])
        controller.homogeneous, controller.proposals = False, {(0, 0, 1):10}
        controller.env = SimpleNamespace(headway_events=SimpleNamespace(headways_s=recorded_gaps(native)))
        controller.prefix, controller.evidence = object(), object()
        return controller

    def test_all_three_controller_builders_keep_complete_solver_outputs_and_budgets(self):
        native = fixture()
        for name in controllers.CONTROLLERS:
            with self.subTest(controller=name):
                controller = self.prepared(name, native)
                info = {'fixture':1}
                if name == 'native_beam':
                    with patch.object(ConsistentFlowController, 'build_model', return_value=(native, info)):
                        model, returned = controller.build_model(None, ())
                    reference = IndexedNativeGraph(native, native.observed)
                else:
                    with patch.object(FlowController, 'build_model', return_value=(native, info)), \
                         patch.object(controllers, 'inputs', return_value=native.observed):
                        model, returned = controller.build_model(None, ())
                    reference = ArrivalMetricGraph(native, native.observed)
                self.assertIs(returned, info)
                actual = controller.select_plan(model)
                expected = (choose_current(reference) if name == 'arrival_current' else
                            flow.choose(reference, proposals=controller.proposals))
                self.assertEqual(actual, expected)
                self.assertLessEqual(actual['evaluations'], 7 if name == 'arrival_current' else 128)

    def test_arrival_metric_still_rejects_a_wrong_recorded_history(self):
        native = fixture()
        controller = self.prepared('arrival_current', native)
        controller.env.headway_events.headways_s.append(0.)
        with patch.object(FlowController, 'build_model', return_value=(native, {})), \
             patch.object(controllers, 'inputs', return_value=native.observed):
            with self.assertRaises(AssertionError):
                controller.build_model(None, ())

    def test_actor_is_forwarded_to_the_original_benchmark_audit(self):
        with patch.object(entry.fixed.BenchmarkAudit, '__init__', return_value=None) as initialize:
            credit = object()
            audit = entry.Audit(12023, 'arrival_current', credit)
        initialize.assert_called_once_with(12023, 'fixed_bilateral025')
        self.assertEqual(audit.controller, 'arrival_current')
        self.assertIs(audit.credit, credit)

    def test_policy_family_must_be_selected_explicitly(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                entry.parser().parse_args(['run', '--root', '/unused', '--actor', '12023',
                                          '--day', '2099001', '--sampler', '333'])

    def test_new_dates_and_actors_flow_through_the_public_entry_without_archives(self):
        for actor, day in ((12023, 2099001), (12035, 2099002)):
            with self.subTest(actor=actor, day=day), TemporaryDirectory() as directory:
                args = entry.parser().parse_args(['run', '--root', str(Path(directory)/'new_result'),
                    '--actor', str(actor), '--day', str(day), '--sampler', '333',
                    '--controller', 'arrival_current'])
                credit = SimpleNamespace(stream=SimpleNamespace(total=[10., 20., 0., 0., 30.]))
                bridge = SimpleNamespace(queries=[], predictions=[], summary=lambda:{'queries':0})
                audit = SimpleNamespace(bridge=bridge, dispatches=[], arrivals=[], departures=[], decisions=[],
                    callback_wall=[.01], holding_person_s=0., dispatch={}, delays={}, contract={}, coverage={},
                    execution={'done_reason':'service_cleared'}, chain={})
                with patch.object(entry, 'preflight', return_value={'reference_trajectory_required':False}), \
                     patch.object(entry.fixed.pilot, 'CreditObserver', return_value=credit), \
                     patch.object(entry, 'Audit', return_value=audit) as make_audit, \
                     patch.object(entry.fixed.pilot, 'rollout', return_value=({}, {}, [], [])) as rollout, \
                     patch.object(entry.previous, 'label_resources'):
                    result = entry.run_one(args)
                self.assertEqual(rollout.call_args.args[:5], (args.root, actor, day, 'stochastic', 333))
                make_audit.assert_called_once_with(actor, 'arrival_current', credit)
                self.assertEqual((result['actor'], result['day']), (actor, day))
                self.assertFalse(result['reference_trajectory_required'])
                self.assertEqual(result['training_updates'], 0)
                self.assertTrue((args.root/'summary/results.json').is_file())

    def test_existing_output_is_not_overwritten_or_replayed(self):
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(root=Path(directory), actor=12023, day=2099001,
                                   sampler=333, controller='native_beam')
            with patch.object(entry, 'preflight', return_value={}), \
                 patch.object(entry.fixed.pilot, 'rollout') as rollout:
                with self.assertRaises(FileExistsError):
                    entry.run_one(args)
            rollout.assert_not_called()


class AcceptanceTests(unittest.TestCase):
    def stats(self):
        metrics = ('passenger_person_s', 'journey_min', 'cv_squared', 'readiness_delay_s',
                   'public_launch_delay_total_s', 'holding_s_per_trip', 'trips_unlaunched', 'trips_incomplete')
        return {m:dict(mean=0. if m.startswith('trips_') else -1.,
                       ci95=[0., 0.] if m.startswith('trips_') else [-2., -.1],
                       by_day={str(d):0. if m.startswith('trips_') else -1. for d in range(4)}) for m in metrics}

    def test_frozen_strict_and_nonharm_rules_are_not_relaxed(self):
        value = dict(metrics=self.stats())
        value['metrics']['passenger_person_s']['ci95'] = [-1., 0.]
        self.assertIn('passenger_person_s', acceptance.failed_metrics(value, 'fixed360_bilateral025'))
        self.assertNotIn('passenger_person_s', acceptance.failed_metrics(value, 'frozen_v276_apc'))
        value['metrics']['readiness_delay_s']['ci95'] = [-1., 0.]
        self.assertIn('readiness_delay_s', acceptance.failed_metrics(value, 'frozen_v275_apc'))
        value['metrics']['trips_incomplete']['by_day']['0'] = 1.
        self.assertIn('trips_incomplete', acceptance.failed_metrics(value, 'frozen_v276_apc'))

    def test_incomplete_matrix_does_not_create_a_successful_gate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'summary').mkdir()
            (root/'summary/results.json').write_text(json.dumps(dict(status='incomplete_failed_slots', missing=[1])))
            check = root/'entry.json'
            check.write_text(json.dumps(dict(status='ready_for_new_frozen_evaluations')))
            value = acceptance.accept(root, check)
            self.assertEqual(value['status'], 'original_matrix_incomplete')
            self.assertNotIn('frozen_eligible', value)

    def test_all_four_days_are_read_and_a_negative_original_decision_stays_negative(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'summary').mkdir()
            (root/'traces').mkdir()
            days = list(range(4))
            arms = ('fixed360_bilateral025', 'frozen_v275_apc', 'frozen_v276_apc',
                    'arrival_cv_beam', 'arrival_cv_current')
            rows = []
            for i, (arm, day) in enumerate((a, d) for a in arms for d in days):
                rows.append(dict(index=i, day=day, arm=arm, results={'journey_min':1.}, wall_s=1.))
                with gzip.open(root/'traces'/f'p{i}.pkl.gz', 'wb') as stream:
                    pickle.dump({field:[] for field in ('holds', 'dispatches', 'arrivals', 'departures', 'queries', 'uniforms')}, stream)
            effects = {}
            for arm in arms[3:]:
                for base in arms[:3]:
                    values = self.stats()
                    values['cv_squared']['ci95'] = [-1., .1]
                    effects[arm+'_vs_'+base] = dict(metrics=values, joint_gate=False)
            old = dict(status='twenty_development_paths_complete', by_day=rows, effects=effects,
                eligible={arm:False for arm in arms[3:]}, current_vs_beam={}, decision='close_no_expansion', source='fixture')
            (root/'summary/results.json').write_text(json.dumps(old))
            check = root/'entry.json'
            check.write_text(json.dumps(dict(status='ready_for_new_frozen_evaluations')))
            value = acceptance.accept(root, check)
            self.assertEqual(len(value['same_metric_pairs']), 4)
            self.assertEqual(value['frozen_decision'], 'close_no_expansion')
            self.assertFalse(any(value['frozen_eligible'].values()))
            self.assertFalse(value['main_policy_changed'])


if __name__ == '__main__':
    unittest.main()
