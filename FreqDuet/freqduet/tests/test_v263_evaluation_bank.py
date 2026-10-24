import copy
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch

import freqduet_v263_evaluation_bank as bank
import run_freqduet_v263_evaluation_bank as runner
import test_v257_atomic_smoke as frozen


def lock_for(context, native_bin=1, feasible=None, primary='native'):
    chosen = native_bin if primary == 'native' else int(primary[1:])
    return dict(context=context, native_bin=native_bin, chosen_branch=primary, chosen_bin=chosen,
        contrast_id=f'c{context}:{primary}_vs_native_b{native_bin}',
        plan=bank.action_plan(native_bin, list(range(7)) if feasible is None else feasible))


class SyntheticPaths(unittest.TestCase):
    def setUp(self):
        self.saved = random.getstate(), np.random.get_state(), torch.get_rng_state()
        self.fixtures = []

    def tearDown(self):
        for fixture in reversed(self.fixtures):
            fixture.audit.detach()
        random.setstate(self.saved[0]); np.random.set_state(self.saved[1]); torch.set_rng_state(self.saved[2])

    def fixture(self, cls, unit, branch, reference, **kw):
        def build(_seed, name, uniforms, ref):
            uniforms.seed = unit['prefix_sampler']
            return cls(unit['actor'], name, uniforms, ref)
        with patch.object(frozen, 'Audit', build):
            f = frozen.Fixture(branch, reference, **kw)
        f.env.scenario_tape = frozen.ScenarioTape(unit['demand_day'])
        f.env.scenario_tape.normal_stream(0, 1, 'road', 1)
        self.fixtures.append(f)
        return f

    def completed(self, fixture):
        data = fixture.complete()
        for name in bank.METRICS:
            data['results'].setdefault(name, 0.)
        data['results'].update(base_actors_unchanged=True, training_updates=0, dual_updates=0)
        return data

    def s0(self, context, mask=None):
        unit = bank.selection.selection_unit(context, 0)
        cls = bank.selection.selection_audit(frozen.Base, frozen.repeated, frozen.event, frozen.joint, unit)
        return self.completed(self.fixture(cls, unit, 'native', None, mask=mask))

    def make(self, unit, branch='native', reference=None, ctx=None, lock=None, **kw):
        if ctx is None:
            s = self.s0(unit['context'], kw.get('mask'))
            ctx = dict(unit=s['unit'], anchor=s['anchor'])
        if lock is None:
            q = ctx['anchor']['query']
            lock = lock_for(unit['context'], q['native_bin'], q['feasible_bins'])
        cls = bank.evaluation_audit(frozen.Base, frozen.repeated, frozen.event, frozen.joint, unit, ctx, lock)
        return self.fixture(cls, unit, branch, reference, **kw)

    def reference(self, data):
        return {k:data[k] for k in ('unit','anchor','future','locked_contrast')}

    def test_contexts_E_roots_halves_prefix_and_exact_uniform_object(self):
        for c,f in ((0,0), (0,7), (1,3), (2,4), (3,7)):
            unit = bank.evaluation_unit(c,f)
            fixture = self.make(unit)
            data = self.completed(fixture)
            self.assertTrue(bank.validate_path(data, unit, data['locked_contrast'], frozen.same)['passed'])
            self.assertIs(fixture.rollout_kwargs['event_uniforms'], fixture.uniforms)
            self.assertIs(fixture.rollout_kwargs['probability_rule'], fixture.audit)
            self.assertEqual(data['future']['root'], 3540000000+100*c+20+f)
            self.assertEqual(data['future']['event_seed'], unit['streams']['lower_event_and_sampling'])
            self.assertEqual(unit['half'], 'A' if f < 4 else 'B')
            self.assertEqual(data['anchor']['scenario']['seed'], unit['demand_day'])
            self.assertEqual(fixture.policy.calls, 3)
            self.assertEqual(fixture.returned_values, [True,True,True])
        self.assertEqual(bank.core.SMOKE_ROOT, 3540999999)

    def test_all_legal_bins_retained_without_E_winner_reselection(self):
        unit = bank.evaluation_unit(1,6)
        s = self.s0(1)
        q = s['anchor']['query']
        primary = 'b2' if q['native_bin'] != 2 else 'b3'
        lock = lock_for(1, q['native_bin'], q['feasible_bins'], primary)
        ctx = dict(unit=s['unit'], anchor=s['anchor'])
        native = self.completed(self.make(unit, ctx=ctx, lock=lock))
        paths = dict(native=native)
        for branch in lock['plan']['physical'][1:]:
            paths[branch] = self.completed(self.make(unit, branch, self.reference(native), ctx, lock))
        other = next(b for b in paths if b not in ('native',primary))
        paths[other]['results']['passenger_person_s'] = -1000000.
        before = copy.deepcopy(lock)
        result = bank.validate_unit(paths, unit, lock, frozen.same, frozen.event.alignment)
        self.assertEqual(result['primary_branch'], primary)
        self.assertEqual(result['primary_contrast_id'], lock['contrast_id'])
        self.assertEqual(lock, before)
        self.assertEqual(len(result['paths']), 7)
        self.assertFalse(result['selection_recomputed'])
        self.assertTrue(any(p['delta_vs_native']['passenger_person_s'] > 0 for p in result['paths'].values()))
        for name,p in result['paths'].items():
            self.assertEqual(p['coupling']['uniform_mismatches'], 0)
            h = p['holding']
            self.assertEqual(h['direct_executed_delta_s']+h['later_same_cycle_delta_s']+
                h['other_cycles_delta_s'], h['all_cycles_delta_s'])
            self.assertIsNone(p['supply']['first_next_delta_s'])
            for request in paths[name]['requests']:
                if not request['is_anchor']:
                    self.assertEqual(request['selected_bin'], request['native_bin'])

    def test_native_primary_still_enumerates_every_legal_bin(self):
        unit = bank.evaluation_unit(0,2)
        n = self.completed(self.make(unit))
        lock, ctx = n['locked_contrast'], dict(unit=bank.selection.selection_unit(0,0), anchor=n['anchor'])
        paths = dict(native=n)
        for b in lock['plan']['physical'][1:]:
            paths[b] = self.completed(self.make(unit,b,self.reference(n),ctx,lock))
        result = bank.validate_unit(paths,unit,lock,frozen.same,frozen.event.alignment)
        self.assertEqual(result['primary_branch'],'native')
        self.assertEqual(len(result['paths']),7)

    def test_mask_alias_and_forced_native_never_start(self):
        unit = bank.evaluation_unit(0,1)
        s = self.s0(0)
        native_bin = s['anchor']['query']['native_bin']
        mask = [native_bin, next(b for b in range(7) if b != native_bin)]
        n = self.completed(self.make(unit, mask=mask))
        lock = n['locked_contrast']
        ctx = dict(unit=bank.selection.selection_unit(0,0), anchor=n['anchor'])
        cls = bank.evaluation_audit(frozen.Base,frozen.repeated,frozen.event,frozen.joint,unit,ctx,lock)
        self.assertEqual(len(lock['plan']['physical']),2)
        for name in (f'b{native_bin}','forced_native',*lock['plan']['masked']):
            with self.subTest(branch=name),self.assertRaises(AssertionError):
                cls(unit['actor'],name,frozen.event.EventUniforms(unit['prefix_sampler']),self.reference(n))

    def test_S_unit_root_wrong_half_or_derived_stream_is_rejected(self):
        u = bank.evaluation_unit(2,4)
        for bad in (bank.selection.selection_unit(2,4),dict(u,root=3540000204),
                    dict(u,half='A'),dict(u,streams=dict(u['streams'],lower_event_and_sampling=0))):
            with self.assertRaises(AssertionError):
                bank.evaluation_audit(frozen.Base,frozen.repeated,frozen.event,frozen.joint,bad,None,None)

    def test_S0_prefix_mask_mismatch_or_missing_anchor_stops_before_fork(self):
        unit = bank.evaluation_unit(3,0)
        s = self.s0(3)
        q = s['anchor']['query']
        lock = lock_for(3,q['native_bin'],q['feasible_bins'])
        ctx = dict(unit=s['unit'],anchor=copy.deepcopy(s['anchor']))
        ctx['anchor']['theta'][0] = 99.
        f = self.make(unit,ctx=ctx,lock=lock)
        with self.assertRaises(AssertionError): f.complete()
        self.assertIsNone(f.audit.fork_anchor)
        self.assertEqual(f.env.scenario_tape.seed,unit['demand_day'])
        ctx = dict(unit=s['unit'],anchor=s['anchor'])
        for kw in (dict(supported=False),dict(mask=[0,q['native_bin']] )):
            f = self.make(unit,ctx=ctx,lock=lock,**kw)
            with self.assertRaises(AssertionError): f.complete()
            self.assertIsNone(f.audit.fork_anchor)

    def test_wrong_actor_sampler_or_cross_E_reference_is_rejected(self):
        u = bank.evaluation_unit(3,0)
        n = self.completed(self.make(u))
        ctx = dict(unit=bank.selection.selection_unit(3,0),anchor=n['anchor'])
        cls = bank.evaluation_audit(frozen.Base,frozen.repeated,frozen.event,frozen.joint,u,ctx,n['locked_contrast'])
        for actor,sampler in ((12037,u['prefix_sampler']),(u['actor'],0)):
            with self.assertRaises(AssertionError): cls(actor,'native',frozen.event.EventUniforms(sampler))
        bad = self.reference(n); bad['unit'] = bank.evaluation_unit(3,1)
        with self.assertRaises(AssertionError):
            cls(u['actor'],'b0',frozen.event.EventUniforms(u['prefix_sampler']),bad)

    def test_actual_execution_or_frozen_choice_corruption_is_not_admitted(self):
        u = bank.evaluation_unit(0,0)
        n = self.completed(self.make(u)); lock = n['locked_contrast']
        for part in ('sample','ledger','departure','S_choice'):
            d = copy.deepcopy(n)
            if part == 'sample': d['actions'][-1]['action'] = (d['actions'][-1]['action']+1)%7
            if part == 'ledger': d['ledger']['commanded_holding_s'] += 5
            if part == 'departure': d['payloads']['v248']['departures'] = []
            if part == 'S_choice': d['locked_contrast']['chosen_bin'] = (lock['chosen_bin']+1)%7
            with self.subTest(part=part),self.assertRaises(AssertionError):
                bank.validate_path(d,u,lock,frozen.same)


class MetadataAndBudget(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/bank.COLLECTION
        self.source = 'synthetic'
        self.locks = [lock_for(c,primary='b2' if c == 1 else 'native') for c in range(4)]
        runner.write(self.root/'evaluation_authorization.json',bank.authorization(self.source,self.locks))

    def pass_path(self,u,b):
        lock = self.locks[u['context']]
        runner.write(self.root/'checks'/bank.unit_key(u)/(b+'.json'),dict(source=self.source,
            unit=u,branch=b,passed=True,locked_contrast=lock,plan=lock['plan']))

    def test_namespace_isolation_and_fixed_E_halves(self):
        inventories = [set(r['streams'].values()) for r in bank.SPEC['futures']]
        self.assertEqual(sum(map(len,inventories)),len(set().union(*inventories)))
        for c in range(4):
            for f in range(8):
                u = bank.evaluation_unit(c,f)
                self.assertEqual(u['phase'],'evaluation')
                self.assertEqual(u['half'],'A' if f<4 else 'B')
                self.assertNotEqual(u['root'],bank.selection.selection_unit(c,f)['root'])
        for c,f in ((4,0),(-1,0),(0,8)):
            with self.assertRaises(AssertionError): bank.evaluation_unit(c,f)

    def test_seal_is_required_and_choices_are_copied_without_reselection(self):
        summary = dict(source=bank.S_SEAL_SOURCE,collector_source=bank.S_SOURCE,
            status='v262_all_S_complete_contrasts_sealed_E_closed',complete_units=32,failed_starts=0,
            contrasts_sealed=True,offline_choices_made=True,evaluation_authorized=False,
            evaluation_outcomes_opened=0,contrasts=self.locks)
        seal = dict(summary=summary);before = copy.deepcopy(seal)
        locks = bank.choices_from_seal(seal)
        self.assertEqual(locks,self.locks)
        locks[0]['plan']['physical'].pop()
        self.assertEqual(seal,before)
        summary['contrasts_sealed'] = False
        with self.assertRaises(AssertionError): bank.choices_from_seal(seal)

    def test_32_units_224_starts_and_locked_primary_not_an_E_winner(self):
        for c in range(4):
            for f in range(8):
                u = bank.evaluation_unit(c,f);lock = self.locks[c]
                for b in lock['plan']['physical']:
                    bank.claim_slot(self.root,u,b,self.source,self.locks);self.pass_path(u,b)
                runner.write(self.root/'units'/(bank.unit_key(u)+'.json'),dict(source=self.source,
                    unit=u,passed=True,plan=lock['plan'],locked_contrast=lock,
                    paths={b:{} for b in lock['plan']['physical']},primary_branch=lock['chosen_branch'],
                    primary_contrast_id=lock['contrast_id'],all_arms_retained=True,selection_recomputed=False))
        a = NS(out_root=self.root,source=self.source)
        result = runner.aggregate(a,self.locks)
        self.assertTrue(result['complete']);self.assertEqual(result['physical_starts'],224)
        self.assertFalse(result['selection_recomputed'])
        p = self.root/'units/c0_evaluation_f0.json';d=runner.read(p);d['primary_branch']='b0';runner.write(p,d)
        with self.assertRaises(AssertionError): runner.aggregate(a,self.locks)
        with self.assertRaises(AssertionError):
            bank.claim_slot(self.root,bank.evaluation_unit(0,0),'native',self.source,self.locks)

    def test_missing_units_do_not_become_a_complete_bank(self):
        result = runner.aggregate(NS(out_root=self.root,source=self.source),self.locks)
        self.assertFalse(result['complete']);self.assertEqual(len(result['missing_units']),32)

    def test_aliases_masks_or_S_units_cannot_consume_E_slots(self):
        self.locks[0]=lock_for(0,feasible=[0,1,3])
        runner.write(self.root/'evaluation_authorization.json',bank.authorization(self.source,self.locks))
        u=bank.evaluation_unit(0,0)
        bank.claim_slot(self.root,u,'native',self.source,self.locks);self.pass_path(u,'native')
        for b in ('b1','b2','b4','b5','b6','forced_native'):
            with self.assertRaises(AssertionError): bank.claim_slot(self.root,u,b,self.source,self.locks)
        with self.assertRaises(AssertionError):
            bank.claim_slot(self.root,bank.selection.selection_unit(0,0),'native',self.source,self.locks)
        self.assertEqual(len(list((self.root/'started').glob('*/*.json'))),1)

    def test_failed_attempt_burns_slot_and_never_retries(self):
        u=bank.evaluation_unit(0,0);a=NS(out_root=self.root,source=self.source)
        output=io.StringIO()
        with redirect_stdout(output),redirect_stderr(output),patch.object(
                runner,'interfaces',side_effect=RuntimeError('synthetic load failure')) as load:
            with self.assertRaises(SystemExit): runner.collect_path(a,u,'native',None,self.locks)
            with self.assertRaises(AssertionError): runner.collect_path(a,u,'native',None,self.locks)
            self.assertEqual(load.call_count,1)
        failure=runner.read(self.root/'checks'/bank.unit_key(u)/'native.json')
        self.assertEqual(failure['failure'],'RuntimeError: synthetic load failure')
        self.assertIn('synthetic load failure',output.getvalue())
        with self.assertRaises(AssertionError): bank.claim_slot(self.root,u,'b0',self.source,self.locks)
        self.assertEqual(len(list((self.root/'started').glob('*/*.json'))),1)

    def test_missing_or_changed_authorization_blocks_before_S_reads_and_runtime(self):
        p=self.root/'evaluation_authorization.json';p.unlink()
        with patch.object(runner,'read') as reads,patch.object(runner,'interfaces') as load:
            with self.assertRaises(AssertionError): runner.collect_unit(NS(out_root=self.root,source=self.source))
            reads.assert_not_called();load.assert_not_called()
        wrong=bank.authorization(self.source,self.locks);wrong['locked_contrasts'][0]=lock_for(0,primary='b0')
        runner.write(p,wrong)
        with self.assertRaises(AssertionError):
            bank.claim_slot(self.root,bank.evaluation_unit(0,0),'native',self.source,self.locks)
        self.assertFalse((self.root/'started').exists())


if __name__ == '__main__':
    unittest.main()
