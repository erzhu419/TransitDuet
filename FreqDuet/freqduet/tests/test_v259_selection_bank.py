import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch

import freqduet_v259_selection_bank as bank
import run_freqduet_v259_selection_bank as runner
import test_v257_atomic_smoke as frozen


class SyntheticPaths(unittest.TestCase):
    def setUp(self):
        self.saved = random.getstate(),np.random.get_state(),torch.get_rng_state()
        self.fixtures = []

    def tearDown(self):
        for f in reversed(self.fixtures):
            f.audit.detach()
        random.setstate(self.saved[0]);np.random.set_state(self.saved[1]);torch.set_rng_state(self.saved[2])

    def make(self,unit,branch='native',reference=None,ctx=None,**kw):
        cls = bank.selection_audit(frozen.Base,frozen.repeated,frozen.event,frozen.joint,unit,ctx)
        # Adapt only the old c0 toy fixture's two hard-coded constructor arguments.
        def build(_seed,name,uniforms,ref):
            uniforms.seed = unit['prefix_sampler']
            return cls(unit['actor'],name,uniforms,ref)
        with patch.object(frozen,'Audit',build):
            f = frozen.Fixture(branch,reference,**kw)
        f.env.scenario_tape = frozen.ScenarioTape(unit['demand_day'])
        f.env.scenario_tape.normal_stream(0,1,'road',1)
        self.fixtures.append(f)
        return f

    def completed(self,f):
        data = f.complete()
        for name in bank.METRICS:
            data['results'].setdefault(name,0.)
        data['results'].update(base_actors_unchanged=True,training_updates=0,dual_updates=0)
        return data

    def reference(self,data):
        return {k:data[k] for k in ('unit','anchor','future')}

    def test_frozen_contexts_roots_prefix_and_uniform_object(self):
        first = None
        for c,future in ((0,0),(0,7),(1,0),(2,3),(3,7)):
            with self.subTest(context=c,future=future):
                unit = bank.selection_unit(c,future)
                ctx = dict(context=0,anchor=first['anchor']) if c == 0 and future > 0 else None
                fixture = self.make(unit,ctx=ctx)
                data = self.completed(fixture)
                self.assertTrue(bank.validate_path(data,unit,frozen.same)['passed'])
                self.assertIs(fixture.rollout_kwargs['event_uniforms'],fixture.uniforms)
                self.assertIs(fixture.rollout_kwargs['probability_rule'],fixture.audit)
                self.assertEqual(fixture.policy.calls,3)
                self.assertEqual(data['future']['root'],unit['root'])
                self.assertEqual(data['future']['event_seed'],unit['streams']['lower_event_and_sampling'])
                self.assertEqual(fixture.returned_values,[True,True,True])
                self.assertEqual(data['anchor']['scenario']['seed'],unit['demand_day'])
                if first is None:
                    first = data
        self.assertEqual(bank.core.SMOKE_ROOT,3540999999)

    def test_all_legal_arms_execution_and_diagnostics_without_outcome_filtering(self):
        unit = bank.selection_unit(3,6)
        native = self.completed(self.make(unit))
        plan = bank.action_plan(native['anchor']['query']['native_bin'],list(range(7)))
        branches = dict(native=native)
        for branch in plan['physical'][1:]:
            branches[branch] = self.completed(self.make(unit,branch,self.reference(native)))
        result = bank.validate_unit(branches,unit,frozen.same,frozen.event.alignment)
        self.assertTrue(result['passed'])
        self.assertEqual(len(branches),7)
        self.assertFalse(result['offline_choices_made'])
        self.assertEqual(result['evaluation_outcomes_opened'],0)
        for name,row in result['paths'].items():
            self.assertEqual(row['coupling']['uniform_mismatches'],0)
            h = row['holding']
            self.assertEqual(h['direct_executed_delta_s']+h['later_same_cycle_delta_s']+
                             h['other_cycles_delta_s'],h['all_cycles_delta_s'])
            self.assertEqual(row['supply']['status'],'missing_first_next_launch')
            self.assertIsNone(row['supply']['first_next_delta_s'])
        self.assertTrue(any(r['delta_vs_native']['passenger_person_s'] > 0 for r in result['paths'].values()))

    def test_alias_mask_and_forced_native_are_never_physical_S_arms(self):
        unit = bank.selection_unit(0,1)
        full = self.completed(self.make(unit))
        native_bin = full['anchor']['query']['native_bin']
        masked = self.completed(self.make(unit,mask=[0,native_bin]))
        plan = bank.action_plan(native_bin,masked['anchor']['query']['feasible_bins'])
        self.assertEqual(len(plan['physical']),2)
        cls = bank.selection_audit(frozen.Base,frozen.repeated,frozen.event,frozen.joint,unit)
        for name in (f'b{native_bin}','forced_native',*plan['masked']):
            with self.subTest(label=name),self.assertRaises(AssertionError):
                cls(unit['actor'],name,frozen.event.EventUniforms(unit['prefix_sampler']),self.reference(masked))

    def test_wrong_context_future_or_evaluation_reference_is_rejected(self):
        unit = bank.selection_unit(0,0)
        native = self.completed(self.make(unit))
        for bad in (dict(unit,phase='evaluation'),dict(unit,root=3540000020)):
            with self.assertRaises(AssertionError):
                bank.selection_audit(frozen.Base,frozen.repeated,frozen.event,frozen.joint,bad)
        cls = bank.selection_audit(frozen.Base,frozen.repeated,frozen.event,frozen.joint,bank.selection_unit(0,1))
        with self.assertRaises(AssertionError):
            cls(unit['actor'],'b0',frozen.event.EventUniforms(unit['prefix_sampler']),self.reference(native))
        with self.assertRaises(AssertionError):
            bank.selection_audit(frozen.Base,frozen.repeated,frozen.event,frozen.joint,unit,
                                 dict(context=1,anchor=native['anchor']))

    def test_prefix_mismatch_stops_before_fork_and_missing_anchor_has_no_replacement(self):
        unit = bank.selection_unit(2,0)
        native = self.completed(self.make(unit))
        bad = self.reference(copy.deepcopy(native))
        bad['anchor']['theta'][0] = 99.
        f = self.make(unit,'b0',bad)
        with self.assertRaises(AssertionError):
            f.complete()
        self.assertIsNone(f.audit.fork_anchor)
        self.assertEqual(f.env.scenario_tape.seed,unit['demand_day'])
        missing = self.make(unit,supported=False)
        with self.assertRaises(AssertionError):
            missing.complete()
        self.assertIsNone(missing.audit.fork_anchor)

    def test_native_actor_and_prefix_sampler_are_bound_to_the_unit(self):
        unit = bank.selection_unit(3,0)
        cls = bank.selection_audit(frozen.Base,frozen.repeated,frozen.event,frozen.joint,unit)
        with self.assertRaises(AssertionError):
            cls(12037,'native',frozen.event.EventUniforms(unit['prefix_sampler']))
        with self.assertRaises(AssertionError):
            cls(unit['actor'],'native',frozen.event.EventUniforms(402112137))

    def test_execution_command_or_ledger_corruption_is_not_admitted(self):
        unit = bank.selection_unit(0,0)
        original = self.completed(self.make(unit))
        for part in ('action','commanded_H','missing_departure'):
            data = copy.deepcopy(original)
            if part == 'action':
                data['actions'][-1]['action'] = (data['actions'][-1]['action']+1)%7
            elif part == 'commanded_H':
                data['ledger']['commanded_holding_s'] += 5
            else:
                data['payloads']['v248']['departures'] = []
            with self.subTest(part=part),self.assertRaises(AssertionError):
                bank.validate_path(data,unit,frozen.same)


class MetadataAndBudget(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/bank.COLLECTION
        self.source = 'synthetic'
        runner.write(self.root/'selection_authorization.json',bank.authorization(self.source))

    def pass_path(self,unit,branch,mask=None):
        runner.write(self.root/'checks'/bank.unit_key(unit)/(branch+'.json'),
            dict(source=self.source,unit=unit,branch=branch,passed=True,
                 plan=bank.action_plan(1,list(range(7)) if mask is None else mask)))

    def test_all_root_and_derived_streams_are_disjoint_from_E_and_smoke(self):
        rows = bank.SPEC['futures']
        seeds = [set(r['streams'].values()) for r in rows]
        self.assertEqual(sum(map(len,seeds)),len(set().union(*seeds)))
        self.assertEqual(len([r for r in rows if r['phase']=='selection']),32)
        self.assertEqual(len([r for r in rows if r['phase']=='evaluation']),32)
        for c in range(4):
            for f in range(8):
                u = bank.selection_unit(c,f)
                self.assertEqual(u['phase'],'selection')
                self.assertIsNone(u['half'])
        for c,f in ((4,0),(0,8),(-1,0)):
            with self.assertRaises(AssertionError):
                bank.selection_unit(c,f)

    def test_32_units_reach_224_unique_starts_not_256(self):
        for c in range(4):
            for f in range(8):
                unit = bank.selection_unit(c,f)
                for branch in bank.action_plan(1,list(range(7)))['physical']:
                    bank.claim_slot(self.root,unit,branch,self.source)
                    self.pass_path(unit,branch)
                plan = bank.action_plan(1,list(range(7)))
                runner.write(self.root/'units'/(bank.unit_key(unit)+'.json'),dict(source=self.source,
                    unit=unit,passed=True,plan=plan,paths={b:{} for b in plan['physical']}))
        self.assertEqual(len(list((self.root/'started').glob('*/*.json'))),224)
        result = runner.aggregate(NS(out_root=self.root,source=self.source))
        self.assertTrue(result['complete'])
        self.assertEqual(result['physical_starts'],224)
        self.assertFalse(result['evaluation_authorized'])
        self.assertFalse(result['contrasts_sealed'])
        with self.assertRaises(AssertionError):
            bank.claim_slot(self.root,bank.selection_unit(0,0),'native',self.source)

    def test_missing_units_are_not_silently_dropped_or_marked_complete(self):
        result = runner.aggregate(NS(out_root=self.root,source=self.source))
        self.assertFalse(result['complete'])
        self.assertEqual(len(result['missing_units']),32)
        self.assertEqual(result['evaluation_outcomes_opened'],0)

    def test_alias_mask_does_not_refund_or_replace_slots(self):
        unit = bank.selection_unit(1,2)
        bank.claim_slot(self.root,unit,'native',self.source)
        self.pass_path(unit,'native',[0,1,3])
        for branch in ('b1','b2','b4','b5','b6','forced_native'):
            with self.assertRaises(AssertionError):
                bank.claim_slot(self.root,unit,branch,self.source)
        self.assertEqual(len(list((self.root/'started').glob('*/*.json'))),1)

    def test_failed_attempt_burns_slot_and_has_no_automatic_retry(self):
        unit = bank.selection_unit(0,0)
        a = NS(out_root=self.root,source=self.source)
        output = io.StringIO()
        with redirect_stdout(output),redirect_stderr(output),patch.object(
                runner,'interfaces',side_effect=RuntimeError('synthetic load failure')) as load:
            with self.assertRaises(SystemExit):
                runner.collect_path(a,unit,'native',None)
            with self.assertRaises(AssertionError):
                runner.collect_path(a,unit,'native',None)
            self.assertEqual(load.call_count,1)
        failed = runner.read(self.root/'checks'/bank.unit_key(unit)/'native.json')
        self.assertEqual(failed['failure'],'RuntimeError: synthetic load failure')
        self.assertIn('synthetic load failure',output.getvalue())
        with self.assertRaises(AssertionError):
            bank.claim_slot(self.root,unit,'b0',self.source)
        self.assertEqual(len(list((self.root/'started').glob('*/*.json'))),1)

    def test_missing_authorization_blocks_before_interfaces_or_claims(self):
        (self.root/'selection_authorization.json').unlink()
        with patch.object(runner,'interfaces') as load,self.assertRaises(AssertionError):
            runner.collect_unit(NS(out_root=self.root,source=self.source))
        load.assert_not_called()
        self.assertFalse((self.root/'started').exists())

    def test_supply_extractor_uses_first_actual_launch_not_later_matching_trip(self):
        cycle = dict(cycle_key=[0,7,13],launch_s=5.,return_s=30.,executed_holding_s=10.,return_event_seq=50)
        data = dict(anchor=dict(query=dict(cycle_key=[0,7,13])),payloads=dict(
            v235=dict(cycles=[cycle]),legacy=dict(events=[
                dict(kind='launch',bus_id=1,trip_id=88,event_seq=51,time_s=31.),
                dict(kind='launch',bus_id=0,trip_id=8,event_seq=52,time_s=32.),
                dict(kind='launch',bus_id=0,trip_id=9,event_seq=90,time_s=50.)],dispatches=[
                dict(bus=0,trip=8,actual_s=32.,public_baseline_s=31.),
                dict(bus=0,trip=9,actual_s=50.,public_baseline_s=45.)])))
        result = bank.anchor_supply(data)
        self.assertEqual(result['next_trip'],8)
        self.assertEqual(result['other_elapsed_s'],15.)
        data['payloads']['legacy']['events'] = []
        self.assertIsNone(bank.anchor_supply(data)['next_trip'])


if __name__ == '__main__':
    unittest.main()
