from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import numpy as np
import torch

import run_freqduet_v272_graph_pilot as pilot
from test_v271_native_event_graph import fake_env


class PilotTests(unittest.TestCase):
    def test_unique_twelve_paths_and_eight_sequential_units(self):
        indices = [i for shard in pilot.SHARDS for i in shard]
        self.assertEqual(sorted(indices), list(range(12)))
        self.assertEqual(len(set(pilot.ITEMS)), 12)
        self.assertEqual(len(pilot.SHARDS), 8)
        self.assertEqual([len(s) for s in pilot.SHARDS], [2]*4+[1]*4)

    def test_selected_law_uses_legal_native_bin_without_mutation(self):
        original = torch.tensor([[0.,0.,1.,0.,0.,0.,0.]])
        chosen = pilot.selected_law(original, 45, (0,10,45))
        self.assertEqual(chosen.tolist(), [[0.,0.,0.,0.,0.,0.,1.]])
        self.assertEqual(original.tolist(), [[0.,0.,1.,0.,0.,0.,0.]])
        with self.assertRaises(AssertionError):
            pilot.selected_law(original, 5, (0,10,45))

    def test_actual_execution_matches_requests_not_native_recommendations(self):
        decisions = [dict(time_s=200.,bus=0,trip=0,station=1,action_s=10,executed_request_s=5)]
        holds = [dict(time_s=200.,bus_id=0,trip_id=0,station_id=1,action_s=5.)]
        self.assertEqual(pilot.executed_chain(decisions, holds), dict(verified_commands=1,changed_from_native=1))
        holds[0]['action_s'] = 10.
        with self.assertRaises(AssertionError):
            pilot.executed_chain(decisions, holds)

    def audit(self, arm):
        law = torch.tensor([[0.,0.,1.,0.,0.,0.,0.]])
        class Base:
            def __init__(self,*args):
                pass
            def __call__(self,*args):
                return law
        fixed = NS(BenchmarkAudit=Base, pilot=NS(step=NS(ACTIONS=np.array([0,5,10,15,20,30,45]))))
        credit = NS(stream=NS(pending_commands=[]))
        audit = pilot.make_audit(fixed, arm, credit)
        audit.env = fake_env()
        audit.runner = NS(_pilot_agent=0, _bus_for_agent=lambda _:audit.env.bus_all[0])
        policy = NS(feasible_action_mask=lambda _:torch.ones((1,7),dtype=torch.bool))
        return audit, policy, law

    def test_baseline_returns_same_original_tensor(self):
        audit, policy, law = self.audit(pilot.ARMS[0])
        self.assertIs(audit(policy,None,None,None),law)
        self.assertEqual(audit.decisions[0]['executed_request_s'],10)

    def test_graph_callback_reaches_native_output_and_preserves_base_law(self):
        for arm in pilot.ARMS[1:]:
            audit, policy, law = self.audit(arm)
            audit.bridge = NS(observe=Mock(return_value=dict(shadow_action_s=5)))
            result = audit(policy,None,None,None)
            self.assertEqual(int(result.argmax()),1)
            self.assertEqual(int(law.argmax()),2)
            self.assertEqual(audit.decisions[0]['action_s'],10)
            self.assertEqual(audit.decisions[0]['executed_request_s'],5)
            audit.bridge.observe.assert_called_once_with(audit.env.bus_all[0],(0,5,10,15,20,30,45),10)

    def test_sampler_seeds_are_aliases_under_deterministic_laws(self):
        from run_freqduet_v69_event_coupling import EventUniforms, categorical_index
        a = EventUniforms(27011001).draw(4,6,True)[1]
        b = EventUniforms(27011002).draw(4,6,True)[1]
        self.assertNotEqual(a,b)
        for i in range(7):
            law = np.eye(7)[i]
            self.assertEqual(categorical_index(law,a),categorical_index(law,b))

    def test_statistics_use_four_days_and_keep_service_constraints(self):
        cube = {(d,a):{m:float(10+i) for m in pilot.METRICS}
                for d in pilot.DAYS for i,a in enumerate((pilot.ARMS[1],pilot.ARMS[0],pilot.ARMS[2]))}
        for row in cube.values():
            row['trips_unlaunched'] = row['trips_incomplete'] = 0.
        result = pilot.summarize(cube)
        self.assertTrue(all(r['joint_development_gate'] for r in result.values()))
        cube[pilot.DAYS[-1],pilot.ARMS[1]]['cv_squared'] = 100.
        result = pilot.summarize(cube)
        self.assertTrue(all(not r['joint_development_gate'] for r in result.values()))


if __name__ == '__main__':
    unittest.main()
