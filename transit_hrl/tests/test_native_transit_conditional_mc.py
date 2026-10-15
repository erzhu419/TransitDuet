import copy
import json
import unittest

import numpy as np

from freq_hrl.domains.transit import native_frozen_future as learning


class ConditionalMCTest(unittest.TestCase):
    def test_shared_future_duplication_does_not_inflate_sample_size(self):
        rows = [{'root': 397, 'scene_seed': 1, 'probe_index': i} for i in (8, 9)]
        choices = {v: [1, 1] for v in learning.VARIANTS}
        choices['raw'] = [0, 0]
        cell = {'blocks': [{'states': rows, 'choices': choices}]}
        panels = {}
        for row in rows:
            values = np.zeros((8, 6, 4))
            values[:, 1, 0] = np.arange(8.)
            values[:, 1, 1] = np.arange(8.)/4
            values[:, 1, 2] = np.arange(8.)*3/4
            values[:, 1, 3] = -np.arange(8.)
            panels[learning.state_key(row)] = {'values': values}
        first = learning.conditional_mc([cell], panels)
        duplicated = learning.conditional_mc([cell, copy.deepcopy(cell)], panels)
        v = first['variants']['joint_raw_half_safe']
        self.assertEqual(first['fixed_states'], 2)
        self.assertEqual(duplicated['state_exposures'], 4)
        self.assertEqual(first['variants'], duplicated['variants'])
        self.assertAlmostEqual(v['fields']['cost']['mean'], 3.5)
        self.assertAlmostEqual(v['fields']['cost']['conditional_mc_se'], np.sqrt(6./16))
        self.assertEqual(v['sampled_fleet_event_state_draws'], 14)
        self.assertEqual(v['changed_states'], 2)
        self.assertEqual(first, json.loads(json.dumps(first)))
        self.assertEqual(first['variants']['raw']['fields']['cost']['conditional_mc_se'], 0.)


if __name__ == '__main__': unittest.main()
