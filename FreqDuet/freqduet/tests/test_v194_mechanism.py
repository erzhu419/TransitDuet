from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_freqduet_v194_mechanism as stage


class MechanismTests(unittest.TestCase):
    def fixture(self,now=12.,joint=False):
        path=dict(dispatch={1:dict(actual_s=0.)},
            arrivals={(1,1,False):dict(time_s=10.,realized_holding=0.),
                (1,2,False):dict(time_s=35.,realized_holding=3.)},
            departures={(1,1,False):dict(time_s=20.,realized_holding=3.),
                (1,2,False):dict(time_s=42.,realized_holding=4.)},
            ready={(1,1,False):dict(ready_time_s=14.,bus_id=99),
                (1,2,False):dict(ready_time_s=40.,bus_id=99)})
        geometry={False:dict(ids=[0,1,2],index={0:0,1:1,2:2})}
        p=stage.accounting.parts.zeros()
        if joint:
            p.update(running=12.,downstream_service=4.,current_ready_joint=3.)
            truth=dict(running=15.,service=5.,residence=5.)
        elif now==12.:
            p.update(running=12.,current_service=1.,downstream_service=4.,holding=2.,execution_wait=2.)
            truth=dict(running=15.,service=7.,residence=6.)
        else:
            p.update(running=6.,downstream_service=4.)
            truth=dict(running=10.,service=5.,residence=0.)
        coarse=stage.accounting.coarse(p)
        q=dict(query_time_s=now,episode=[1,99,2,False],point_s=sum(p.values()),prediction_parts=p,
            inseparable_current_ready_key=[1,1,False] if joint else None)
        r=dict(label=dict(available_time_s=40.,full_eta_s=40.-now),truth_coarse=truth,
            residual_s={k:coarse[k]-truth[k] for k in truth})
        item=((0,2,0),None,dict(latest=9.,mean_age_min=.1),dict(mean=12.),.5 if now==25. else 1.)
        return q,r,item,path,geometry,{(1,2,False):(20.,35.)}

    def test_service_holding_and_execution_wait_stay_separate(self):
        values,facts=stage.component_labels(*self.fixture())
        np.testing.assert_allclose(values,[3.,0.,2.,1.,1.,0.,7.])
        np.testing.assert_allclose(facts[2:6],[23.,23.,28.,23.])

    def test_current_joint_residence_is_not_pure_holding(self):
        values,_=stage.component_labels(*self.fixture(now=15.,joint=True))
        np.testing.assert_allclose(values,[3.,0.,1.,0.,0.,2.,6.])

    def test_first_running_truth_excludes_already_elapsed_travel(self):
        values,facts=stage.component_labels(*self.fixture(now=25.))
        np.testing.assert_allclose(values,[4.,0.,1.,0.,0.,0.,5.])
        np.testing.assert_allclose(facts[2:6],[10.,10.,15.,10.])

    def test_signal_strict_completion_and_absorption_identity(self):
        rows=np.array([[12.,10.,4.,1.,10.],[100.,10.,8100.,11.,20.],[200.,10.,36100.,21.,30.]])
        prefix=SimpleNamespace(groups={(0,2,0):([10.,20.,30.],np.vstack((np.zeros(5),np.cumsum(rows,axis=0))))})
        i,a,j=stage.innovation_parts(prefix,(0,2,0),20.,11.,.5)
        np.testing.assert_allclose([i,a,j],[1.,.5,.5])
        self.assertAlmostEqual(i-a,j)

    def test_extended_moments_recover_old_coordinates(self):
        v=np.arange(stage.N,dtype=float)/7.
        np.testing.assert_allclose(stage.old_moment(stage.moment(v)),stage.prior.moment(v[:stage.prior.N]),atol=1e-12)

    def test_original_episode_denominators_do_not_renormalize_selection(self):
        values=stage.collapse({(1,):{(0,2,0):np.ones(2)},(2,):{(0,2,0):np.ones(2)}},
            {(1,):np.array([4.,4.]),(2,):np.array([2.,2.])})
        np.testing.assert_allclose(values[(0,2,0)],[.375,.375])


if __name__=='__main__':
    unittest.main()
