#!/usr/bin/env python3
"""Seal the inbound-chain pilot; metadata and synthetic tests only, no replay."""

import argparse
import ast
import itertools
import json
from pathlib import Path
import subprocess
import sys


CONTEXTS = tuple(itertools.product((12037, 933083), (2061001, 2061004)))
ACTIONS = (0, 5, 10, 15, 20, 30, 45)
UPPER = (0, -30, 30)
CANDIDATES = tuple(itertools.product(UPPER, (0, 1)))
CONSTRAINTS = ('cv_squared', 'readiness_delay_s', 'public_launch_delay_total_s',
               'trips_unlaunched', 'trips_incomplete')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def future_seed(context, future):
    require(context in range(4) and future in range(32), 'unregistered future')
    return 2140000000 + 100 * context + future


def choose_supply(frame):
    """Only the pre-command, current physical snapshot is admitted here."""
    now, direction = frame['time_s'], frame['direction']
    if not 7200 <= now < 8100 or not frame['fresh_upper']:
        return []
    due = [p for p in frame['plans'] if not p['launched']
           and p['direction'] == direction and p['planned_s'] <= now]
    if not due or any(b['direction'] == direction for b in frame['ready']):
        return []
    inbound = sorted((b for b in frame['active'] if b['direction'] != direction),
                     key=lambda b: (b['remaining_distance_m'], b['bus_id'], b['trip_id']))[:2]
    return inbound if any(b['controlled_stop_ahead'] for b in inbound) else []


def reduction(anchor, query, seen):
    """One strictly smaller native bin per committed physical trip, or abstain."""
    if not anchor['time_s'] <= query['time_s'] < anchor['time_s'] + 900:
        return None
    identity = (query['bus_id'], query['trip_id'])
    pool = {(b['bus_id'], b['trip_id']) for b in anchor['inbound']}
    if identity not in pool or identity in seen or query['direction'] == anchor['direction']:
        return None
    if query['terminal_masked'] or not (query['forward_valid'] and query['follower_valid']):
        return None
    base, feasible = query['base_bin'], query['feasible_bins']
    require(base in feasible, 'bilateral incumbent must be feasible')
    smaller = [i for i in feasible if i < base]
    return max(smaller) if smaller else None


def joint_gate(stats):
    return stats['passenger_person_s']['ci95'][1] < 0 and all(
        stats[k]['ci95'][1] <= 0 for k in CONSTRAINTS)


def registration(source):
    return dict(stage='v214_inbound_supply_chain_preregistration', source=source,
        contexts=[list(c) for c in CONTEXTS], baseline='V106 bilateral025 on frozen B200 deployment',
        anchor='First FRESH native upper decision with actual query time in[7200,8100),before command: '
            'no ready vehicle in dispatch direction,at least one currently due unlaunched same-direction plan. '
            'Rank ALL opposite-direction active vehicles by remaining route distance,bus ID,trip ID;commit first two '
            '(or one if only one exists). At least one must have a native nonterminal controlled stop at/ahead. '
            'Use current last/next station,travel_distance and static route geometry only. If any context has no '
            'anchor,stop the pilot;no substitute context,day or hindsight anchor.',
        selection_inputs=['current query time', 'native fresh-upper flag', 'dispatch direction',
            'ready bus IDs/directions', 'active bus/trip/direction/current position',
            'static controlled-stop topology', 'current pending plans/queue',
            'query-time bilateral bin/feasible mask/forward and follower evidence'],
        lower='At each selected physical trip,first subsequent nonzero bilateral025 query with valid bilateral '
            'evidence,unmasked nonterminal action and a strictly smaller feasible native bin,in[anchor,anchor+900). '
            'Reduce ONE adjacent feasible bin,at most once per committed trip(max2). Match bus AND trip. '
            'No replacement vehicle,second attempt,added holding,negative holding or forced future identity. '
            'Lower0 records the same causal shadow opportunities without changing commands. All later control '
            'resumes native bilateral025;branch-dependent opportunities stay branch-dependent.',
        candidates=[dict(id=i, upper_shift_s=u, reduce_inbound=l) for i, (u, l) in enumerate(CANDIDATES)],
        upper='One native pre-EMA coefficient shift0/-30/+30,clipped to[-60,60]. Original SINGLE shared '
            'antisymmetric coefficient,EMA alpha.35,900s native rolling-zero-sum exact-headway planner and all '
            'execution gates unchanged. No independent directional coefficients,direct release,terminal wait '
            'override,layover cut,extra vehicle,service deletion or slack credit. Both directions are scored.',
        resource_ledger='Exactly12 fixed-pool vehicles. Capture anchor full ready/active/queue inventory;query '
            'states/time/issued and executed holding;every pending-plan update,actual launch,assigned physical '
            'vehicle and preceding inbound return. For every vehicle trip,elapsed=executed holding+other elapsed. '
            'Actual return/next assignment/realized follower labels are evaluation only,never selection inputs. '
            'Record anchor-selected supply trips even if they ultimately do not supply a changed slot.',
        future_namespace='2140000000+100*context+future;smoke2149000000+context',
        selection_futures=list(range(16)), evaluation_futures=list(range(16, 32)),
        fork='Fork only future innovations AFTER exact pre-command anchor/prefix match,using frozen V68 '
            'fork_future and V211 event-uniform coupling. Preserve current sampled window/day latent state. '
            'One common future per context across arms;smoke,selection,evaluation disjoint.',
        select='Six arms on16 selection futures/context. Pool joint=all6,upper-only=[0,2,4],lower-only=[0,1]. '
            'Choose minimum mean passenger_person_s subject to all constraint mean deltas<=0 vs0. '
            'Include0 as fallback,tie smaller ID. Freeze all three choices and best-single(min passenger mean, '
            'tie ID) BEFORE reading any evaluation labels. No fitting/state scorer/threshold search.',
        outcomes=dict(passenger_person_s='Frozen full-horizon waiting+onboard person-seconds including censored '
                'passengers;also journey_min using ALL generated demand,not served-passenger-only.',
            cv_squared='Frozen actual arrival-gap variance/mean-gap squared over BOTH directions and full service. '
                'Keep V211 measurement population;local/departure CV is diagnostic only.',
            readiness_delay_s='Original mean actual launch minus first execution-eligible clock. Also report '
                'total readiness delay,preceding return and actual launch clocks to expose eligibility effects.',
            public_launch_delay_total_s='SUM max(actual launch-ORIGINAL published launch,0),including all control '
                'waiting;also mean per launched trip,launched count,unlaunched and incomplete counts. '
                'Never replace original public clock with updated plan or score a missing launch as zero.'),
        gate='Primary contrasts:joint minus bilateral025,selection-best upper-only,and selection-best lower-only. '
            'For EACH contrast require passenger95%CI upper<0 and ALL constraint95%CI uppers<=0 under BOTH '
            'crossed actor/day and conditional-future bootstraps(5000 draws,seed214;paired futures resampled '
            'within each context;crossed actors/days resampled independently). Require the same strict passenger '
            'improvement/nonharm means in BOTH predeclared evaluation halves16--23/24--31. No metric tradeoff, '
            'relaxed margin or adaptive extension. Report selection-best-single too.',
        execution_gate='No absent opportunity may be counted as an active reduction. Retain all contexts/futures '
            'including zero opportunities,clipped upper noops,blocked dispatch and changed vehicle assignment; '
            'unsupported labels remain missing,not zero. Require complete paired outcomes,valid conservation '
            'and unchanged authority. Any uncleared arm blocks admission;retain its censored outcomes,do not '
            'impute launch delay for missing service. In each evaluation half,at least one joint execution must BOTH change '
            'the native upper coefficient/plan and execute a nonzero inbound reduction. To attribute supply-chain '
            'coordination,also demonstrate an altered selected-trip return followed by an earlier same-terminal '
            'actual launch of that physical vehicle. Failure means no coordination admission,not filtered data.',
        smoke='LATER authorization required. Test contracts,exact baseline archive endpoint,full observer-off/on '
            'noop action/timing/resource equivalence;repeat a joint arm on separate smoke future. Require a real '
            'nonzero lower reduction plus upper plan change within original masks and conserved ledger. '
            'Use context0 only,max6 smoke replays;four unbranched context references. Stop on mismatch/no support.',
        budget=dict(references=4, smoke_max=6, selection_max=384, evaluation_max=256,
                    evaluation_deduplicate=True, adaptive_extension=False),
        limitations='Four consumed actor/day prefixes with new future innovations,not new demand-day confirmation '
            'or an online deployable optimizer. Simulator fork retains hidden sampled-window/day state. '
            'Future selection can establish conditional opportunity only,not fixed-headway superiority or '
            'frequency-separation contribution. A passed pilot requires separate fresh-day policy validation.',
        current_authorization='Preregistration,metadata audit and synthetic tests ONLY;no references/smoke/replay.',
        new_rollouts=0, training_updates=0, interval_fit=False, main_policy_changed=False)


def read(path):
    return json.loads(path.read_text())


def audit(runtime, archives, parent, snapshot, out, source):
    require(not (out/'protocol.json').exists(), 'V214 protocol already sealed')
    receipt = read(snapshot)
    require(receipt['conflicts'] == [] and receipt['archive_checked'], 'scheduler namespace conflict/incomplete audit')
    conflicts, scanned = [], 0
    for stage in archives.glob('protocol_v6_*'):
        if not stage.is_dir() or stage == out:
            continue
        require('v214_' not in stage.name, 'pre-existing V214 root requires investigation')
        for path in stage.glob('*.json'):
            if path.name not in ('protocol.json', 'registration.json', 'manifest.json'):
                continue
            text = path.read_text()
            scanned += 1
            if any(token in text for token in ('2140000000', '2149000000', 'v214_inbound')):
                conflicts.append(str(path))
    require(scanned > 0 and not conflicts, 'archived protocol namespace conflict/missing inventory')
    required = {'env/bus.py': ('direction', 'travel_distance', 'last_station', 'next_station', 'on_route', 'trip_id'),
        'env/sim.py': ('bus_in_terminal', 'timetables', 'launch_bus', '_ready_vehicle_count',
                       '_freqduet_first_eligible_time', '_freqduet_scheduled_launch'),
        'runner_v3.py': ('_lower_terminal_action_masked', '_pending_lower_action_context',
                        '_upper_callback_v2', 'upper_decision_taken', 'timetable_action_ema_alpha')}
    for file, names in required.items():
        tree = ast.parse((runtime/file).read_text())
        found = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        found.update(n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        found.update(n.id for n in ast.walk(tree) if isinstance(n, ast.Name))
        found.update(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str))
        require(set(names) <= found, f'missing runtime observable interface:{file}:{set(names)-found}')
    old = read(parent/'protocol.json')
    require(old['contexts'] == [list(c) for c in CONTEXTS], 'parent context contract differs')
    protocol = registration(source)
    out.mkdir(parents=True, exist_ok=True)
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    summary = dict(status='v214_preregistration_sealed_no_replay', source=source,
        namespace_audit=dict(scheduler=receipt, archived_protocols_checked=scanned, conflicts=conflicts,
            scope='Known FreqDuet hot/archive scheduler receipts and top-level protocol metadata on node004'),
        runtime_interfaces_available=True, live_observation_and_execution_equivalence='pending later authorized smoke',
        contexts=protocol['contexts'], candidates=protocol['candidates'], budget=protocol['budget'],
        selection_futures=protocol['selection_futures'], evaluation_futures=protocol['evaluation_futures'],
        new_rollouts=0, training_updates=0, interval_fit=False, main_policy_changed=False)
    (out/'summary').mkdir()
    (out/'summary/status.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('V214_PREREGISTRATION_COMPLETE_NO_REPLAY', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime-root', 'archive-root', 'v211-root', 'scheduler-snapshot', 'out-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v214_inbound_preregistration.py', '-v'], check=True)
    audit(a.runtime_root, a.archive_root, a.v211_root, a.scheduler_snapshot, a.out_root, a.source)


if __name__ == '__main__':
    main()
