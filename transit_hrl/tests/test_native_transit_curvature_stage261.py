import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_curvature_response as learning
from scripts import run_native_transit_curvature_stage261 as spec
from scripts import submit_native_transit_curvature_stage261_scheduleurm as submit
from tests import test_native_transit_calibration_ablation_stage260 as fixtures


class CurvatureTest(unittest.TestCase):
    def test_design_matches_joint_and_fixed_difference_curvature(self):
        rng = np.random.default_rng(31)
        z = rng.normal(size=(12,6,3))
        precision = rng.uniform(.2,2.,size=(12,15,2))
        eligible = np.ones((12,6),bool);eligible[:6,5]=False
        _,_,m,c = learning.source.shrink.geometry(z,precision,eligible)
        for penalty in (None,0.,.7):
            dual=.3
            f=learning.factor(z,precision,eligible,penalty,dual)
            joint=m+dual*np.eye(3) if penalty is None else np.block([[m+dual*np.eye(3),c],[c,m+penalty*np.eye(3)]])
            np.testing.assert_allclose(f['inverse_basis']@f['inverse_basis'].T,np.linalg.inv(joint),rtol=1e-11,atol=1e-11)
            l=learning.leverage(f,np.arange(6))
            self.assertGreaterEqual(l['maximum'],0.)
            self.assertLessEqual(l['maximum'],1.+1e-12)
            self.assertLessEqual(l['trace'],f['rank']+1e-12)

    def test_block_leverage_detects_prefix_only_identified_direction(self):
        z=np.zeros((12,6,2))
        z[:6,:,0]=np.arange(6);z[6:,:,1]=np.arange(6)
        f=learning.factor(z,np.ones((12,15,2)),np.ones((12,6),bool),None,0.)
        l=learning.leverage(f,np.arange(6))
        self.assertAlmostEqual(l['maximum'],1.)
        self.assertAlmostEqual(l['data_trace_fraction'],.5)

    def test_low_block_extends_ties_and_retains_null_modes(self):
        basis,s=learning.low_subspace(np.diag([0.,1.,1.,10.]))
        self.assertEqual((s['null_modes'],s['positive_modes'],s['low_modes']),(1,3,3))
        np.testing.assert_allclose(basis@basis.T,np.diag([1.,1.,1.,0.]))
        _,isotropic=learning.low_subspace(np.eye(4))
        self.assertEqual(isotropic['low_dimension_fraction'],1.)
        _,empty=learning.low_subspace(np.zeros((2,2)))
        self.assertIsNone(empty['low_cut_to_maximum'])

    def test_cached_active_cap_influence_identity_without_new_solves(self):
        data,c,prior,cap,geometry=fixtures.CalibrationAblationTest().numeric_fixture()
        c['target']*=5
        indices,removed=np.arange(6,12),np.arange(6)
        full,_=learning.source.fit_fixed(data,c,np.arange(12),prior,cap,geometry)
        fixed,_=learning.source.fit_fixed(data,c,indices,prior,cap,geometry)
        x=learning.source.source.joint.relative(data['x'],data['originals'])
        active=0
        with patch.object(learning.source.source.source.optimizer,'quadratic_fit',side_effect=AssertionError('Diagnostic refit')):
            for level,(basis,coordinates) in geometry.items():
                calibration={'target':c['target'],'precision':c['precision'],'penalty':c['penalties'][level]}
                for model in ('free_old','shrunk'):
                    full_heads=np.asarray(full[model]['weights'][level])@coordinates.T
                    fixed_heads=np.asarray(fixed[model]['weights'][level])@coordinates.T
                    initial=np.asarray(prior['weights'][level])@coordinates.T
                    context=learning.full_context(x@basis,calibration,data['raw_logits']<=0,initial,full_heads,model,.8)
                    response=learning.prefix_context(x@basis,calibration,data['raw_logits']<=0,initial,fixed_heads,indices,removed,model,.8,context)
                    self.assertLess(response['diagnostics']['influence_equation_residual'],1e-8)
                    active+=response['diagnostics']['prefix_dual']>0
        self.assertGreater(active,0)

    def test_cross_term_and_unsupported_exposure_are_not_dropped(self):
        identity={'dimension':2,'inverse_basis':np.eye(2),'supported':np.eye(2)}
        restricted={'dimension':2,'inverse_basis':np.array([[1.],[0.]]),'supported':np.array([[1.],[0.]])}
        context={'delta':np.array([1.,-1.]),'low':np.array([1.,0.]),'full_factor':identity,'prefix_factor':restricted}
        s=learning.region_stats(np.array([[[1.,1.]]]),np.array([[True]]),np.array([[True]]),context,1.)
        self.assertEqual(s['squared_response'],{'total':0.,'low':1.,'rest':1.,'cross':-2.})
        self.assertEqual(s['exposure']['prefix_factor']['unsupported_squared_sum'],1.)

    def test_summary_preserves_inactive_denominator_and_geometry(self):
        identity={'dimension':2,'inverse_basis':np.eye(2),'supported':np.eye(2)}
        context={'delta':np.array([1.,1.]),'low':np.array([1.,0.]),'full_factor':identity,'prefix_factor':identity}
        stats=learning.region_stats(np.ones((1,6,2)),np.array([[True]]),np.ones((1,6),bool),context,1.)
        stats.update(state_cases=1,disagreements=1)
        spectral={'low_dimension_fraction':.5,'parameter_energy':2.,'low_parameter_energy':1.,'full_dual':0.,'prefix_dual':1.,
            'low_parameter_fraction':.5,'low_cut_to_maximum':.1,'minimum_positive_to_maximum':.1,
            'full_effective_condition':10.,'prefix_effective_condition':20.,'block_leverage':{'maximum':.8,'data_trace_fraction':.2}}
        a={'cell':['mixed_a',0,'even_odd',0],'bank':0,'active':True,'regions':{'held':stats},'spectral':spectral}
        zero={'feasible_action_cases':6,'squared_response':dict.fromkeys(('total','low','rest','cross'),0.),'exposure':{},'state_cases':1,'disagreements':0}
        b={**a,'active':False,'regions':{'held':zero}}
        s=spec.summarize([a,b])
        self.assertEqual((s['prefix_model_exposures'],s['active_prefix_models']),(2,1))
        self.assertEqual(s['regions']['held']['state_model_exposures'],2)
        self.assertAlmostEqual(s['regions']['held']['cost_RMS'],np.sqrt(2.))
        self.assertEqual(s['spectral']['low_parameter_energy_fraction'],.5)

    def test_main_reads_no_evaluation_panels_and_fits_nothing(self):
        with patch('sys.argv',['runner','--run-name','unit_curvature']), \
                patch.object(spec.cache.cached.training.cohort.source.source,'load_data',return_value=({'training_data':[]},None,[],None)), \
                patch.object(spec.cache.cached.training.cohort.source,'matched_views',return_value={}), \
                patch.object(spec,'analyze_cell',return_value={}) as analyze,patch.object(spec,'report',return_value={}), \
                patch.object(spec.Path,'read_text',return_value='{}'),patch.object(spec,'write_json'), \
                patch.object(spec.cache.sampling,'read_panels',side_effect=AssertionError('Evaluation panels opened')), \
                patch.object(learning.source,'fit_fixed',side_effect=AssertionError('Refit')), \
                patch.object(spec.cache.learning,'select',side_effect=AssertionError('Reselection')):
            spec.main()
        self.assertEqual(analyze.call_count,32)

    def test_scheduler_single_unpinned_task_requires_completed_parent(self):
        t=submit.task_specification('unit_curvature')
        self.assertEqual(t['cpu'],1);self.assertIsNone(t['require_node'])
        self.assertEqual(t['allowed_nodes'],[f'node{i:03}' for i in range(1,7)])
        self.assertFalse(any('/results/' in p for p in t['stage_input_paths']))
        parent={'protocol':spec.source.EXPERIMENT_PROTOCOL,'contract':spec.source.contract(),'coverage_qualified':True}
        receipt={'scheduler':{'status_counts':{'done':33},'exit_code_counts':{'0':33}}}
        with patch('sys.argv',['submit','--run-name','unit_curvature']),patch.object(submit,'inventory',return_value=[]), \
                patch.object(submit.Path,'exists',return_value=False), \
                patch.object(submit.Path,'read_text',autospec=True,side_effect=lambda p:json.dumps(receipt if p.name=='completion_receipt.json' else parent)), \
                patch.object(submit.subprocess,'check_output',return_value='revision'),patch.object(submit,'write_json'),patch.object(submit,'execute_bulk') as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]),1)
            bulk.reset_mock();receipt['scheduler']['exit_code_counts']={'1':33}
            with self.assertRaises(SystemExit):submit.main()
            bulk.assert_not_called()


if __name__=='__main__':unittest.main()
