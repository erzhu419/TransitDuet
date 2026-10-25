#!/usr/bin/env python3
"""Frozen S contrasts on the complete E bank; summary-only analysis, no selector."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import unittest

import numpy as np


DRAW_COUNT, SEED = 5000, 253
COLLECTOR_SOURCE, SHARD_SOURCE = 'e414070e27', '85399b1050'
CORE = ('passenger_person_s', 'journey_min', 'cv_squared', 'readiness_delay_s', 'public_launch_delay_total_s')
COHORTS = dict(full=tuple(range(8)), A=tuple(range(4)), B=tuple(range(4, 8)))
HOLDING = ('direct_executed_delta_s', 'later_same_cycle_delta_s', 'other_cycles_delta_s', 'all_cycles_delta_s')


def read(path):
    return json.loads(path.read_text())


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def crossed_weights(contexts):
    actors, days = (sorted({r[k] for r in contexts}) for k in ('actor', 'demand_day'))
    assert len(contexts) == len(actors)*len(days)
    assert len({(r['actor'], r['demand_day']) for r in contexts}) == len(contexts)
    rng = np.random.default_rng(SEED)
    ac = rng.multinomial(len(actors), np.full(len(actors), 1/len(actors)), size=DRAW_COUNT)
    dc = rng.multinomial(len(days), np.full(len(days), 1/len(days)), size=DRAW_COUNT)
    return np.asarray([ac[:, actors.index(r['actor'])]*dc[:, days.index(r['demand_day'])]
        for r in contexts]).T/len(contexts)


def intervals(values, weights, conditional_counts):
    # values: context x future x contrast x metric; every contrast shares draws.
    point = values.mean(axis=(0, 1))
    crossed = np.einsum('dc,cam->dam', weights, values.mean(axis=1))
    conditional = np.einsum('dcf,cfam->dam', conditional_counts, values)/(values.shape[0]*values.shape[1])
    return point, {k:np.percentile(v, [2.5, 97.5], axis=0) for k,v in
        (('crossed_actor_day', crossed), ('conditional_future', conditional))}


def joint_gate(stats, constraints, use_ci):
    value = lambda k: stats[k]['ci95'][1] if use_ci else stats[k]['mean']
    return bool(value('passenger_person_s') < 0 and all(value(k) <= 0 for k in constraints))


def signal_noise(old, new, weights):
    """No clipping: a negative debiased signal/cross-product remains negative."""
    mu, variance = new.mean(axis=1), new.var(axis=1, ddof=1)
    left, right = new[:, :4].mean(axis=1), new[:, 4:].mean(axis=1)
    old_mu = old.mean(axis=1)
    per_context = dict(mean_effect_squared=(mu**2).mean(axis=1),
        single_future_noise_variance=variance.mean(axis=1),
        debiased_signal_squared=(mu**2-variance/8).mean(axis=1),
        E_half_cross_product=(left*right).mean(axis=1),
        S_E_cross_product=(old_mu*mu).mean(axis=1))
    result = {}
    for key, values in per_context.items():
        result[key] = dict(mean=values.mean(axis=0).tolist(),
            ci95=np.percentile(weights@values, [2.5, 97.5], axis=0).T.tolist())
    signal, noise = (per_context[k].mean(axis=0) for k in ('debiased_signal_squared', 'single_future_noise_variance'))
    result['single_future_snr'] = [float(s/n) if n > 0 else None for s,n in zip(signal, noise)]
    result['mean8_future_snr'] = [float(8*s/n) if n > 0 else None for s,n in zip(signal, noise)]
    informative = (left != 0) | (right != 0)
    result['informative_context_action_pairs'] = informative.sum(axis=(0, 1)).tolist()
    result['same_nonzero_E_half_sign_pairs'] = ((np.sign(left) == np.sign(right)) &
        (left != 0) & (right != 0)).sum(axis=(0, 1)).tolist()
    return result


def as_stats(point, bounds, metrics, arm):
    return {method:{m:dict(mean=float(point[arm,j]), ci95=ci[:,arm,j].tolist())
        for j,m in enumerate(metrics)} for method,ci in bounds.items()}


def effect_tensor(rows, locks, bank):
    physical = locks[0]['plan']['physical']
    assert all(r['plan']['physical'] == physical for r in locks)
    values = []
    for c in range(4):
        futures = []
        for f in range(8):
            row = rows[bank.unit_key(bank.evaluation_unit(c, f))]
            arms = [locks[c]['chosen_branch'], *physical]
            deltas = []
            for arm in arms:
                path, native = row['paths'][arm], row['paths']['native']
                delta = [path['metrics'][k]-native['metrics'][k] for k in bank.METRICS]
                np.testing.assert_allclose(delta, [path['delta_vs_native'][k] for k in bank.METRICS], atol=1e-8, rtol=0)
                deltas.append(delta)
            futures.append(deltas)
        values.append(futures)
    return np.asarray(values), ['locked_primary', *physical]


def mechanism(rows, locks, bank, futures):
    paths = [rows[bank.unit_key(bank.evaluation_unit(c,f))]['paths'][locks[c]['chosen_branch']]
        for c in range(4) for f in futures]
    active = [rows[bank.unit_key(bank.evaluation_unit(c,f))]['paths'][locks[c]['chosen_branch']]
        for c in range(4) if locks[c]['chosen_branch'] != 'native' for f in futures]
    supply = {}
    for k in ('return_delta_s', 'first_next_delta_s', 'return_to_next_wait_delta_s', 'public_delay_delta_s'):
        values = [p['supply'][k] for p in active if p['supply'].get(k) is not None]
        supply[k] = dict(supported=len(values), missing=len(active)-len(values),
            mean=float(np.mean(values)) if values else None,
            earlier=sum(v < 0 for v in values), later=sum(v > 0 for v in values))
    return dict(holding_context_average={k:float(np.mean([p['holding'][k] for p in paths])) for k in HOLDING},
        active_paths=len(active), supply_status_counts=dict(Counter(p['supply']['status'] for p in active)),
        active_supply=supply, query_due_matches=sum(p['supply'].get('query_due_obligation_match') is True for p in active))


def analyze(a):
    import freqduet_v263_evaluation_bank as bank
    import run_freqduet_v263_evaluation_bank as collector
    from preregister_freqduet_v253_action_supply import CONSTRAINTS
    assert 'source_e414070e27' in bank.__file__ and 'source_e414070e27' in collector.__file__
    root = a.results_root
    seal = read(root/'protocol_v6_v262_selection_contrasts/sealed_S.json')
    locks = bank.choices_from_seal(seal)
    collection = root/bank.COLLECTION
    aggregate = collector.aggregate(argparse.Namespace(out_root=collection, source=COLLECTOR_SOURCE), locks)
    assert aggregate['complete'] and len(aggregate['units']) == 32 and aggregate['physical_starts'] == 224
    admission = read(root/'protocol_v6_v265_evaluation_remaining/summary/e0_admission.json')
    assert admission['source'] == SHARD_SOURCE and admission['passed'] and admission['locked_contrasts'] == locks
    for shard in range(8):
        row = read(root/'protocol_v6_v265_evaluation_remaining/shards'/f'shard{shard}.json')
        expected = [bank.unit_key(bank.evaluation_unit(c,f)) for c,f in admission['shards'][shard]]
        assert row['source'] == SHARD_SOURCE and row['collector_source'] == COLLECTOR_SOURCE
        assert row['status'] == 'done' and row['planned_units'] == expected
        assert [r['unit'] for r in row['units']] == expected
        assert all(r['returncode'] == 0 and r['passed'] for r in row['units'])
        assert not row['automatic_retry'] and not row['selection_recomputed'] and row['training_updates'] == 0
    rows = aggregate['units']
    query_counts = []
    for row in rows.values():
        unit, lock = row['unit'], locks[row['unit']['context']]
        key = bank.unit_key(unit)
        for branch, path in row['paths'].items():
            marker = read(collection/'started'/key/(branch+'.json'))
            check = read(collection/'checks'/key/(branch+'.json'))
            assert marker == dict(source=COLLECTOR_SOURCE, unit=unit, branch=branch, locked_contrast=lock, retry_authorized=False)
            assert check['branch'] == branch and not check['retry_authorized']
            for name in ('passed','plan','locked_contrast','actual_anchor_holding_s','actual_execution_complete','queries','metrics'):
                assert check[name] == path[name]
            assert path['passed'] and path['actual_execution_complete']
            index = lock['native_bin'] if branch == 'native' else int(branch[1:])
            assert path['actual_anchor_holding_s'] == bank.core.ACTIONS[index]
            coupling = path['coupling']
            assert coupling['uniform_mismatches'] == coupling['native_only'] == coupling['alternative_only'] == 0
            assert coupling['common_events'] == path['queries'] == row['paths']['native']['queries'] > 0
            query_counts.append(path['queries'])
    effects, arms = effect_tensor(rows, locks, bank)
    srows = {bank.unit_key(bank.evaluation_unit(c,f)):seal['all_S_units'][bank.selection.unit_key(bank.selection.selection_unit(c,f))]
        for c in range(4) for f in range(8)}
    old, _ = effect_tensor(srows, locks, bank)
    for c, lock in enumerate(locks):
        saved = seal['summary']['contrasts'][c]['arms'][lock['chosen_branch']]['mean_delta']
        np.testing.assert_allclose(old[c,:,0].mean(axis=0), [saved[k] for k in bank.METRICS], atol=1e-8, rtol=0)
    contexts = [bank.evaluation_unit(c,0) for c in range(4)]
    weights, rng = crossed_weights(contexts), np.random.default_rng(SEED)
    cohorts, full = {}, {}
    indices = [bank.METRICS.index(k) for k in CORE]
    for name, ids in COHORTS.items():
        chosen = effects[:, ids]
        counts = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=(DRAW_COUNT, 4))
        point, bounds = intervals(chosen, weights, counts)
        full[name] = {arm:as_stats(point, bounds, bank.METRICS, j) for j,arm in enumerate(arms)}
        primary = full[name]['locked_primary']
        axes = {key:{str(v):{m:float(chosen[[i for i,r in enumerate(contexts) if r[key] == v],:,0,bank.METRICS.index(m)].mean())
            for m in CORE} for v in sorted({r[key] for r in contexts})} for key in ('actor','demand_day')}
        cohorts[name] = dict(primary=primary, point_joint=joint_gate(primary['crossed_actor_day'], CONSTRAINTS, False),
            interval_joint={k:joint_gate(v, CONSTRAINTS, True) for k,v in primary.items()}, axes=axes,
            descriptive_all_arms={arm:{m:float(point[j,bank.METRICS.index(m)]) for m in CORE}
                for j,arm in enumerate(arms) if arm != 'locked_primary'}, mechanism=mechanism(rows,locks,bank,ids))
    detail = []
    for c, lock in enumerate(locks):
        new, previous = effects[c,:,0], old[c,:,0]
        detail.append(dict(context=c, actor=contexts[c]['actor'], demand_day=contexts[c]['demand_day'],
            contrast_id=lock['contrast_id'], native_alias=lock['chosen_branch'] == 'native',
            S_mean={m:float(previous[:,j].mean()) for j,m in enumerate(bank.METRICS)},
            E_mean={m:float(new[:,j].mean()) for j,m in enumerate(bank.METRICS)},
            E_minus_S={m:float(new[:,j].mean()-previous[:,j].mean()) for j,m in enumerate(bank.METRICS)},
            E_sd={m:float(new[:,j].std(ddof=1)) for j,m in enumerate(bank.METRICS)},
            E_mcse={m:float(new[:,j].std(ddof=1)/np.sqrt(8)) for j,m in enumerate(bank.METRICS)},
            passenger_gain_futures=int((new[:,0] < 0).sum()),
            joint_gain_futures=int(sum(new[f,0] < 0 and all(new[f,bank.METRICS.index(k)] <= 0 for k in CONSTRAINTS) for f in range(8))),
            halves={name:{m:float(new[list(ids),bank.METRICS.index(m)].mean()) for m in CORE}
                for name,ids in COHORTS.items() if name != 'full'},
            primary_futures=[] if lock['chosen_branch'] == 'native' else [dict(future=f,
                delta={m:float(new[f,bank.METRICS.index(m)]) for m in CORE},
                holding=rows[bank.unit_key(bank.evaluation_unit(c,f))]['paths'][lock['chosen_branch']]['holding'],
                supply=rows[bank.unit_key(bank.evaluation_unit(c,f))]['paths'][lock['chosen_branch']]['supply']) for f in range(8)]))
    noise = {name:signal_noise(old[:,:,selection][:,:,:,indices], effects[:,:,selection][:,:,:,indices], weights)
        for name,selection in (('locked_primary',[0]), ('all_legal_nonnative',list(range(2,len(arms)))))}
    for report in noise.values():
        report['metrics'] = CORE
    active_contexts = [i for i,r in enumerate(locks) if r['chosen_branch'] != 'native']
    gate = (all(cohorts['full']['interval_joint'].values()) and cohorts['A']['point_joint'] and cohorts['B']['point_joint'])
    summary = dict(source=a.source, status='v266_frozen_E_analysis_complete', protocol_source='4667fd4668',
        collector_source=COLLECTOR_SOURCE, S_seal_source=bank.S_SEAL_SOURCE, complete_units=32,
        physical_starts=224, failed_starts=0, queries_per_path_range=[min(query_counts),max(query_counts)],
        uniform_mismatches=0, actual_execution_complete=True, locked_contrasts=locks,
        cohorts=cohorts, contexts=detail, signal_noise=noise,
        frozen_service_gate_passed=gate, interval_types_both_required=True,
        half_gate='Point passenger improvement and all original constraint nonharm, as in V105; both half interval types also reported.',
        primary_only_native_contexts=sum(r['chosen_branch'] == 'native' for r in locks),
        primary_non_native_contexts=active_contexts,
        crossed_draws_with_zero_non_native_weight_fraction=float((weights[:,active_contexts].sum(axis=1) == 0).mean()),
        bootstrap=dict(draws=DRAW_COUNT,seed=SEED,actor_clusters=2,day_clusters=2,
            shared_across_metrics_arms_and_contrasts=True, future_resampling='Within each fixed context; all arm/metric pairs share counts.'),
        selection_recomputed=False, new_simulator_starts=0, training_updates=0, main_policy_changed=False,
        v238_changed=False, online_selector_defined=False, resource_specific_claim_admitted=False,
        limitations='Four consumed contexts, two archived actors and two demand days. Futures are new continuations, not new demand days. Native aliases are exact zero controls, not replicated gains. Other bins are descriptive. No distinct supply-blind allocation rule or sustained coordination policy is evaluated.')
    out = root/'protocol_v6_v266_frozen_evaluation/summary'
    write_new(out/'all_arm_statistics.json', full)
    write_new(out/'summary.json', summary)
    print('V266_COMPLETE '+json.dumps(dict(service_gate=gate, cohorts={k:dict(point_joint=v['point_joint'],
        interval_joint=v['interval_joint'], primary=v['primary']) for k,v in cohorts.items()})), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--results-root', type=Path, required=True)
    a = p.parse_args()
    source = a.results_root/'protocol_v6_v263_evaluation_bank/source_e414070e27/FreqDuet/freqduet/scripts'
    sys.path.insert(0,str(source))
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'), pattern='test_v266_evaluation.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.testsRun > 0 and result.wasSuccessful()
    analyze(a)


if __name__ == '__main__':
    main()
