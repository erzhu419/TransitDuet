import ast
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v188_hour2_clocks as stage


class ClockTests(unittest.TestCase):
    def test_strict_prefix_clocks_and_old_prefix_equivalence(self):
        atoms,extras = {},{}
        for i in range(9):
            ep = (i,i,2,True)
            atoms[ep] = ((7000.+i,i),7300.+i,3,np.array([-10.,2.,0.]))
            extras[ep] = (90.,100.,7280.+i,'moving-launch')
        old = stage.prior.previous.Prefix(atoms)
        new = stage.PrefixClocks(atoms,extras)
        self.assertIsNone(new.query(7308.,'moving-launch')['mean'])
        h,ref = new.query(7309.,'service'),old.query(7309.,None)['global']
        self.assertEqual(h['n'],ref['n'])
        self.assertEqual(h['latest'],ref['latest'])
        np.testing.assert_array_equal(h['mean'],ref['mean'])
        self.assertEqual(h['prefix_before_hour'],1.)
        self.assertEqual(h['prefix_same_stage'],0.)
        self.assertEqual(h['prefix_label_delay_s'],300.)
        self.assertEqual(h['prefix_ready_after_arrival_s'],20.)
        self.assertAlmostEqual(h['prefix_query_age_s'],h['prefix_completion_age_s']+300.)
        self.assertEqual(new.query(10909.,'service')['n'],0)

    def test_recorded_native_inputs_reconstruct_partial_running(self):
        route = dict(ids=[1,2,3],positions=np.array([0.,100.,200.]))
        history = stage.native.CompletedMeans([(('running',True,2),10.,10.),
            (('running',True,2),20.,15.),(('running',True,3),30.,20.)])
        rows = {(True,2):[(10.,0.,10.),(20.,5.,15.)],(True,3):[(30.,10.,20.)]}
        q = dict(query_time_s=100.,condition_values=dict(direction=True),
            observable=dict(position=dict(origin_index=0,target_index=2,progress_m=25.)),
            prediction_coarse=dict(running=29.375),support=dict(completed_components=[
                dict(station=i,components=dict(running=history.estimate(('running',True,i),100.))) for i in (2,3)]))
        before = repr(q)
        result = stage.absorbed(q,{True:route},rows,history)
        self.assertEqual(repr(q),before)
        self.assertEqual(result['native_latest_age_s'],75.)
        self.assertEqual(result['native_before_hour'],1.)
        q['support']['completed_components'][0]['components']['running']['latest'] = 100.
        with self.assertRaises(AssertionError):
            stage.absorbed(q,{True:route},rows,history)

    def test_covariance_identity_and_prediction_level_control(self):
        moments = []
        for x in (-1.,1.):
            for noise in (-1.,1.):
                for truth in (-1.,1.):
                    for service in (-1.,1.):
                        v = np.zeros(stage.N)
                        for k,value in dict(prefix_running=x,prefix_service=service,
                                predicted_running_s=x+noise,truth_running_s=truth,
                                running_underestimate_s=truth-x-noise).items():
                            v[stage.VARS.index(k)] = value
                        moments.append(stage.moment(v))
        cov = stage.centered(np.mean(moments,axis=0))
        x,e,t,p = map(stage.VARS.index,('prefix_running','running_underestimate_s','truth_running_s','predicted_running_s'))
        self.assertAlmostEqual(cov[x,e],cov[x,t]-cov[x,p])
        self.assertLess(stage.correlation(cov,'running_underestimate_s'),0.)
        conditioned = stage.adjusted(cov,stage.CONTROLS['forecast_level'])
        self.assertAlmostEqual(float(stage.correlation(conditioned,'running_underestimate_s')),0.)
        duplicated = stage.adjusted(cov,('prefix_service','predicted_running_s','predicted_running_s'))
        np.testing.assert_allclose(conditioned,duplicated,atol=1e-12)

    def test_summary_entry_and_missing_variation(self):
        actors,days,rows = (1,2),(10,11),[]
        for a in actors:
            for day in days:
                moments = []
                for x in (-1.,1.):
                    for s in (-1.,1.):
                        v = np.zeros(stage.N)
                        v[0:2] = [x,s]
                        for k,value in dict(predicted_running_s=x+s,truth_running_s=day,
                                running_underestimate_s=day-x-s,above=float(x > 0.)).items():
                            v[stage.VARS.index(k)] = value
                        moments.append(stage.moment(v))
                rows.append(dict(actor=a,day=day,groups=[dict(level='geometry_time',key=['3','moving-launch'],
                    moments=np.mean(moments,axis=0).tolist())]))
        result = stage.summarize(rows,actors,days,'geometry_time')
        self.assertEqual(result['running_clock_mass'],1.)
        self.assertEqual(set(result['blocks']),set(stage.CONTROLS))
        self.assertEqual(len(result['blocks']['service_only']['estimates']['above']['ci95']),2)
        self.assertTrue(np.isnan(stage.correlation(np.zeros((stage.N,stage.N)),'above')))

    def test_no_collection_or_fitting_calls(self):
        tree = ast.parse(Path(stage.__file__).read_text())
        calls = {n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree)
            if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
        self.assertTrue(calls.isdisjoint({'collect','collect_path','attach_labels','fit_key','shared_fit',
            'save_queries','train','envelope','temporal_fit','prepare','rollout'}))
        self.assertEqual(stage.DAYS,tuple(range(2951001,2951065)))


if __name__ == '__main__':
    unittest.main()
