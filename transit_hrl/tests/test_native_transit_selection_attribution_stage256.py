import json
import sys
import unittest
from unittest.mock import patch

from freq_hrl.domains.transit import native_selection_attribution as learning
from scripts import run_native_transit_selection_attribution_stage256 as spec
from scripts import submit_native_transit_selection_attribution_stage256_scheduleurm as submit


def metrics(cost):
    return dict(zip(learning.FIELDS, (cost, 2*cost, cost/4, 3*cost/4, 0., .01 if cost else 0.)))


def piece(before, after, category):
    return {"before": metrics(before), "after": metrics(after), "category": category, "rekey_kind": "rank" if category == "rekey" else None,
        "old_key_under_new_gate": None, "gate_candidates": {b: {"candidate_exposures": 16, "eligible": 1, "strict_gain_only": 2, "nonharm_some_gain": 1}
            for b in ("prefix_refit", "fixed_bundle")}}


def curve(cost, wait, unsafe=0.):
    stats = {"delta_cost": cost, "delta_wait_min": wait, "delta_unsafe": unsafe, "scaled_cost": cost, "scaled_wait": wait}
    return {"score": max(cost, wait), "by_root": {"397": stats}}


class SelectionAttributionTest(unittest.TestCase):
    def test_transition_and_whole_group_denominator(self):
        self.assertEqual([learning.transition(a,b) for a,b in ((None,None),('a','a'),('a',None),(None,'a'),('a','b'))],
            ['unchanged','unchanged','removed','added','rekey'])
        rows = [piece(-4,0,'removed'), piece(0,1,'added'), piece(-2,-5,'rekey'), piece(-1,-1,'unchanged'), piece(0,0,'unchanged')]
        x = learning.account(rows)
        self.assertAlmostEqual(x['change']['delta_cost'], .4)
        self.assertEqual(x['categories']['removed']['before_conditional_mean']['delta_cost'], -4.)
        self.assertEqual(x['categories']['removed']['contribution']['delta_cost'], .8)
        self.assertEqual(x['categories']['added']['contribution']['delta_cost'], .2)
        self.assertEqual(x['categories']['rekey']['contribution']['delta_cost'], -.6)
        self.assertLessEqual(x['decomposition_max_abs_difference'],1e-12)
        empty = learning.account([piece(0,0,'unchanged')])['categories']['removed']
        self.assertIsNone(empty['before_conditional_mean'])
        self.assertEqual(empty['contribution']['delta_cost'],0.)

    def test_unchanged_keys_and_cost_accounting_detect_bad_joins(self):
        rows = [piece(-1,-2,'unchanged'), piece(-2,-1,'unchanged')]
        with self.assertRaisesRegex(ValueError,'unchanged key'): learning.account(rows)
        row = piece(0,-1,'added')
        row['after']['delta_service'] = 0.
        with self.assertRaisesRegex(ValueError,'cost-component'): learning.account([row])

    def test_neutral_root_strict_gain_is_not_point_harm(self):
        pooled = curve(-1.,-1.)
        pooled['by_root']['401'] = curve(0.,0.)['by_root']['397']
        pooled['score'] = 0.
        s = {'pooled':pooled,'halves':[pooled,pooled]}
        self.assertTrue(learning.rule.nonharm(pooled))
        g = learning.gate(s,'pooled_safe')
        self.assertEqual(g['blocks'],['strict_gain'])
        self.assertTrue(g['nonharm_some_gain'])
        bad = {'pooled':curve(.1,-1.,.2),'halves':[pooled,pooled]}
        blocks = learning.gate(bad,'pooled_safe')['blocks']
        self.assertIn('pooled:397:delta_cost',blocks)
        self.assertIn('pooled:397:delta_unsafe',blocks)
        half = {'pooled':curve(-1.,-1.),'halves':[curve(.1,-1.),curve(-1.,-1.)]}
        self.assertTrue(learning.gate(half,'pooled_safe')['eligible'])
        self.assertFalse(learning.gate(half,'half_safe')['eligible'])
        ss = {'zero':s,'candidate':s}
        self.assertEqual(learning.gate_counts(ss,None,'pooled_safe')['strict_gain_only'],1)
        with self.assertRaisesRegex(ValueError,'cached selection'):learning.gate_counts(ss,'candidate','pooled_safe')

    def test_summaries_preserve_banks_views_and_reproduce_source_means(self):
        records = []
        for view in spec.VIEWS:
            rows = []
            for scope in ('train60','held60'):
                for bank in (0,1):
                    for arm in spec.ARMS:
                        for mode in spec.FILTERS:
                            rows.append({**(piece(-4,0,'removed') if bank == 0 else piece(-2,-5,'rekey')),
                                'scope':scope,'bank':bank,'arm':arm,'filter':mode})
            records.append({'key':[view,0],'pieces':rows})
        expected = {'groups':{}}
        for group in ('mixed_pooled',*spec.VIEWS):
            expected['groups'][group] = {}
            for scope in ('train60','held60'):
                expected['groups'][group][scope] = {}
                for label, before, after in (('pooled',-3,-2.5),('bank0',-4,0),('bank1',-2,-5)):
                    expected['groups'][group][scope][label] = {'outcomes':{
                        f'{bundle}_{arm}_{mode}':metrics(cost) for bundle,cost in (('prefix_refit',before),('fixed_bundle',after))
                        for arm in spec.ARMS for mode in spec.FILTERS}, 'alignment_contrasts':{
                            f'{arm}_{mode}':{f:metrics(after)[f]-metrics(before)[f] for f in ('delta_cost','delta_wait_min','delta_risk','delta_service')}
                            for arm in spec.ARMS for mode in spec.FILTERS}}
                    expected['groups'][group][scope][label]['outcomes'] = {v:{**x,'changed_rate': .005 if label == 'pooled' and v.startswith('fixed_bundle') else x['changed_rate']}
                        for v,x in expected['groups'][group][scope][label]['outcomes'].items()}
        output, maximum = spec.summarize(records,expected)
        name = f'{spec.ARMS[0]}_pooled_safe'
        self.assertEqual(output['mixed_pooled']['held60']['pooled'][name]['bank_cell_exposures'],4)
        self.assertEqual(output['mixed_pooled']['held60']['bank0'][name]['bank_cell_exposures'],2)
        self.assertEqual(maximum,0.)
        expected['groups']['mixed_pooled']['held60']['pooled']['alignment_contrasts'][name]['delta_cost'] += 1.
        with self.assertRaisesRegex(ValueError,'group-mean'):spec.summarize(records,expected)

    def test_scheduler_requires_completion_not_a_positive_claim(self):
        task = submit.task_specification('unit_attribution')
        self.assertEqual(task['cpu'],1)
        self.assertIsNone(task['require_node'])
        self.assertEqual(task['allowed_nodes'],[f'node{i:03}' for i in range(1,7)])
        self.assertFalse(any('/results/' in p for p in task['stage_input_paths']))
        self.assertEqual(spec.contract(),json.loads(json.dumps(spec.contract())))
        self.assertEqual(spec.contract()['budget']['new_critic_fits'],0)
        report = {'protocol':spec.source.EXPERIMENT_PROTOCOL,'contract':spec.source.contract(),'coverage_qualified':True}
        receipt = {'scheduler':{'status_counts':{'done':33},'exit_code_counts':{'0':33}}}
        def read(p):return json.dumps(receipt if p.name=='completion_receipt.json' else report)
        with patch.object(sys,'argv',['submit','--run-name','unit_attribution']),patch.object(submit,'inventory',return_value=[]), \
                patch.object(submit.Path,'exists',return_value=False),patch.object(submit.Path,'read_text',autospec=True,side_effect=read), \
                patch.object(submit.subprocess,'check_output',return_value='revision'),patch.object(submit,'write_json'),patch.object(submit,'execute_bulk') as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]),1)
            bulk.reset_mock();receipt['scheduler']['exit_code_counts']={'0':32,'1':1}
            with self.assertRaises(SystemExit):submit.main()
            bulk.assert_not_called()


if __name__=='__main__':unittest.main()
