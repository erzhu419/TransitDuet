from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import preregister_freqduet_v189_leg_innovation as stage


class InnovationTests(unittest.TestCase):
    def test_pre_entry_forecast_and_strict_arrival_clock(self):
        rows = {(True,2):[(100.,0.,100.),(300.,150.,150.),(500.,350.,150.)]}
        history = stage.prior.native.CompletedMeans([(('running',d,s),end,value)
            for (d,s),values in rows.items() for end,start,value in values])
        prefix = stage.Innovations(rows,history)
        self.assertEqual((prefix.total,prefix.with_baseline),(3,2))
        self.assertIsNone(prefix.query((True,2,0),300.,125.))
        result = prefix.query((True,2,0),301.,125.)
        self.assertEqual(result['n'],1)
        self.assertEqual(result['innovation'],50.)
        self.assertEqual(result['absorbed'],25.)
        self.assertEqual(result['remaining'],25.)
        self.assertIsNone(prefix.query((True,2,0),4101.,125.))
        self.assertIsNone(prefix.query((True,3,0),301.,125.))

    def test_entry_hour_alignment_and_native_absorption_identity(self):
        rows = {(True,2):[(100.,0.,100.),(3700.,3500.,200.),(3900.,3750.,150.)]}
        history = stage.prior.native.CompletedMeans([(('running',True,2),end,value)
            for end,start,value in rows[True,2]])
        prefix = stage.Innovations(rows,history)
        # At 4000 the native window has exactly the two supported traversals.
        self.assertEqual(history.estimate(('running',True,2),4000.)['count'],2)
        older = prefix.query((True,2,0),4000.,175.)
        newer = prefix.query((True,2,1),4000.,175.)
        self.assertEqual(older['n'],1)
        self.assertEqual(newer['n'],1)
        self.assertAlmostEqual((older['remaining']+newer['remaining'])/2.,0.)
        # Matching one entry hour is not the entire native cohort.
        self.assertNotEqual(newer['remaining'],0.)

    def test_future_entry_is_not_a_query_feature(self):
        self.assertEqual(stage.visible_entry(100.,12.,90.),(90.,'observed_entry'))
        for later in (100.,101.,10000.,None):
            self.assertEqual(stage.visible_entry(100.,12.,later),(112.,'predicted_entry'))

    def test_support_days_are_not_pooled_across_actors(self):
        rows = []
        for actor in (1,2):
            for i,day in enumerate(stage.DAYS):
                v = np.zeros(len(stage.FIELDS))
                if (actor == 1 and i < 18) or (actor == 2 and i >= 18):
                    v[:3] = 1.
                rows.append(dict(actor=actor,day=day,scopes=[dict(scope='hour2',eligible_episodes=1,
                    finite_fraction=1.,groups=[dict(key=[1,2,2],values=v.tolist())])]))
        result = stage.summarize(rows,(1,2),'old','hour2')
        self.assertEqual(result['keys_by_count']['9'],dict(observed=2,at_least19=1))
        self.assertAlmostEqual(result['below19_supported_mass']['at_least_nine'],18./128.)
        self.assertEqual(len(result['actor_margins']),2)
        self.assertEqual(len(result['day_margins']),64)
        rows[0]['scopes'][0]['eligible_episodes'] = 0
        result = stage.summarize(rows,(1,2),'old','hour2')
        self.assertFalse(result['complete_crossed_grid'])
        self.assertIsNone(result['rates'])
        self.assertIsNotNone(result['observed_nonempty_path_rates'])

    def test_zero_support_keys_are_not_counted_as_observed(self):
        rows = [dict(actor=1,day=day,scopes=[dict(scope='hour2',eligible_episodes=1,
            finite_fraction=1.,groups=[dict(key=[1,2,2],values=[1.,1.,0.,1.,0.,0.,1.])])]) for day in stage.DAYS]
        result = stage.summarize(rows,(1,),'old','hour2')
        self.assertEqual(result['keys_by_count']['1'],dict(observed=1,at_least19=1))
        self.assertEqual(result['keys_by_count']['9'],dict(observed=0,at_least19=0))


if __name__ == '__main__':
    unittest.main()
