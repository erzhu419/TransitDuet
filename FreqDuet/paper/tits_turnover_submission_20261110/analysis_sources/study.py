"""Server-side archive inventory, frozen-policy supplement and evidence readout."""

import argparse
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NATIVE_CODE = Path('/home/zhengliang01/scheduleurm_work/FreqDuet-v6-engineering-v9-runtime-optimized/FreqDuet/freqduet')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, separators=(',', ':'), allow_nan=False,
                  default=lambda v: v.item())
        f.write('\n')


def bind(assets):
    from worker import bind as old_bind
    frozen = old_bind(assets)
    frozen.base.fixed.pilot.joint.optimized_runtime.__globals__['OPTIMIZED_ROOT'] = NATIVE_CODE
    return frozen


def preflight(args):
    for name in ('benchmark_summary.json', 'training_equivalence/summary.json'):
        target = args.root / 'runtime/runtime_optimization_20261014' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.assets / 'runtime/traffic/runtime_optimization_20261014' / name, target)
    frozen = bind(args.assets)
    pilot = frozen.base.fixed.pilot
    result = dict(completed=True, passed=True, physical_starts=0, training_updates=0,
        pilot_file=inspect.getfile(pilot), rollout_source=inspect.getsource(pilot.rollout),
        regime_inputs=read(HERE / 'protocol.json')['regimes'],
        archive_roots=[str(args.root.parent.parent / ('FreqDuet-v' + s)) for s in
                       ('400-independent-confirmation/run01', '402-budget-mechanism/run01', '403-joint-credit/run01')])
    joint = pilot.joint
    _, Runner, load_config, _ = joint.optimized_runtime(args.root / 'runtime/import_check')
    cfg = load_config(str(joint.MODEL_ROOT / 'configs_freqduet' / (joint.CURRENT + '.yaml')))
    result['config'] = {k: cfg[k] for k in ('data', 'env', 'generalization', 'demand') if k in cfg}
    result['runner_source'] = inspect.getsource(Runner.__init__)
    result['imported_sources'] = {k: str(Path(v.__file__).resolve()) for k, v in
        list(sys.modules.items()) if getattr(v, '__file__', None) and
        str(v.__file__).startswith('/home/zhengliang01/scheduleurm_work/')}
    from freshness_scan import audit
    days = [d for values in read(HERE / 'protocol.json')['days'].values() for d in values]
    old = read(args.root.parent.parent / 'FreqDuet-v401-budget-conditioned-upper/metadata03/summary/results.json')
    base = args.root.parent.parent / 'FreqDuet-v4-worktree/FreqDuet'
    freshness = audit([base / 'freqduet/results_freqduet', base / 'experiments'], days,
                      old['hpc']['resolved_readonly_artifacts'])
    assert not freshness['collisions'] and not freshness['read_errors'], freshness
    collisions = []
    from freshness_scan import hits
    for project in args.root.parent.parent.glob('FreqDuet-*'):
        if project == args.root.parent or not project.is_dir():
            continue
        for pattern in ('source*/protocol.json', 'source*/manifest.json'):
            for path in project.glob(pattern):
                found = hits(read(path), set(days))
                if found:
                    collisions.append(dict(path=str(path), days=found))
    assert not collisions, collisions
    result['freshness'] = dict(days=days, metadata_files=freshness['json_files_read'], collisions=[])
    write(args.root / 'preflight02/summary/results.json', result)
    print('V404_PREFLIGHT_COMPLETE', len(result['imported_sources']), flush=True)


def make_audit(frozen, credit, arm, regime):
    import numpy as np
    from policies import Manager, Controller
    from coupled_model import project_native
    from worker import helper

    class Native(Controller):
        def select_plan(self, model):
            r = model.evaluate(homogeneous=self.homogeneous)
            h = model.snapshot.native_action_s
            self.selection = dict(action_s=h, commands={}, reference=r, selected=r,
                fallback=r, evaluations=1, solver_proposals=1, full_cost=r['cost'],
                fallback_rejected={}, status='retain_native', optimality_claim=False)
            return self.selection

    class Audit(frozen.base.Audit):
        def attach(self, runner):
            super().attach(runner)
            self.saved_measurement = capture_at_measurement(self.env, self)
            self.bridge.detach()
            for key, value in regime.items():
                setattr(self.env, key, value)
            kind, parameter = ('static', float(arm.split('_')[1].replace('p', '.'))) if arm.startswith('static_') else (
                ('cap', float(arm.split('_')[1])) if arm.startswith('cap_') else ('pressure', .5))
            self.manager = Manager(self.env, self.credit, kind, parameter)
            self.manager.arm = {'fixed_model': 'fixed', 'fixed_bilateral': 'fixed',
                                'plan': 'plan', 'budget': 'budget'}.get(arm, 'joint')
            self.variant = self.manager.arm
            self.bridge = (Native if arm == 'fixed_bilateral' else Controller)(
                self.env, self.launches, self.credit, self.manager)
            from env.bus import Bus
            self.bridge.attach(Bus)

        def finish(self, runner, records, holds):
            super().finish(runner, records, holds)
            assert self.passenger_detail['generated'] == self.contract['passengers_generated']

        def detach(self):
            self.env._compute_measurement_vector = self.saved_measurement
            super().detach()

    return Audit('joint', credit)


def capture_at_measurement(env, observer):
    original = env._compute_measurement_vector
    def measure():
        result = original()
        observer.passenger_detail = passenger_detail(env)
        return result
    env._compute_measurement_vector = measure
    return original


def passenger_detail(env):
    import numpy as np
    grouped = {}
    all_wait, all_ride, all_journey = [], [], []
    generated = boarded = arrived = 0
    horizon = env.protocol.evaluation_end_time_s
    for station in env.stations:
        for p in station.total_passenger:
            generated += 1
            bt, at = p.boarding_time, p.arrive_time
            w = max(0., (horizon if bt is None else bt) - p.appear_time)
            ride = 0. if bt is None else max(0., (horizon if at is None else at) - bt)
            boarded += bt is not None
            arrived += at is not None
            all_wait.append(w); all_ride.append(ride); all_journey.append(w + ride)
            key = f'{int(station.station_id)}:{int(p.appear_time // 3600)}'
            g = grouped.setdefault(key, [0, 0., 0.])
            g[0] += 1; g[1] += w; g[2] += ride
    return dict(generated=generated, boarded=boarded, arrived=arrived,
        unserved=generated - boarded, incomplete=boarded - arrived,
        wait_p50_p90_p95_s=np.quantile(all_wait, [.5, .9, .95]).tolist(),
        journey_p50_p90_p95_s=np.quantile(all_journey, [.5, .9, .95]).tolist(),
        waiting_person_s=sum(all_wait), onboard_person_s=sum(all_ride), station_hour=grouped)


def collect(args, regime, day, arm, engineering=False):
    from worker import helper
    plan = read(HERE / 'protocol.json')
    out = args.root / ('engineering' if engineering else 'cases') / regime / str(day) / arm
    out.mkdir(parents=True, exist_ok=False)
    identity = dict(protocol_id=plan['protocol_id'], regime=regime, day=day, arm=arm,
        physical_starts=1, training_updates=0, main_policy_changed=False,
        automatic_retry=False, confirmation_expanded=False)
    write(out / 'started.json', identity)
    begin = time.perf_counter()
    try:
        frozen = bind(args.assets)
        credit = frozen.base.fixed.pilot.CreditObserver()
        audit = make_audit(frozen, credit, arm, {} if engineering else plan['regimes'][regime])
        runtime = args.root / 'runtime' / f'{regime}_{day}_{arm}'
        runtime.mkdir(exist_ok=False)
        row, uniforms, _, holds = frozen.base.fixed.pilot.rollout(runtime, 12011, day,
            'stochastic', 27011001, [0.] * 4, f'v404_{regime}_{day}_{arm}',
            probability_rule=audit, trace_observer=audit, credit_observer=credit)
        row.update(waiting_person_s=float(credit.stream.total[0]),
            onboard_person_s=float(credit.stream.total[1]), backlog_trip_s=float(credit.stream.total[4]),
            holding_person_s=audit.holding_person_s, **audit.dispatch, **audit.delays)
        frozen.base.previous.label_resources(audit.bridge, audit.dispatches, holds)
        trace = dict(holds=holds, dispatches=audit.dispatches, arrivals=audit.arrivals,
            departures=audit.departures, queries=audit.decisions, uniforms=uniforms,
            returns=credit.stream.returns, graph_queries=audit.bridge.queries,
            resource_predictions=audit.bridge.predictions, slow_events=audit.manager.events,
            publish_ticks=audit.publish_ticks, public_clocks=audit.public_clocks)
        result = dict(identity, completed=True, status='complete_development_path',
            results=row, contract=audit.contract, actor=12011, sampler=27011001,
            coverage=audit.coverage, execution=audit.execution, action_chain=audit.chain,
            graph=audit.bridge.summary(), slow=audit.manager.summary(),
            passenger_detail=audit.passenger_detail)
        probes = [q['quota_probe'] for q in audit.bridge.queries if 'quota_probe' in q]
        result['quota_probes'] = dict(attempted=len(probes), supported=sum(p['supported'] is True for p in probes),
            changed=sum(p['changed'] is True for p in probes), extra_evaluations=sum(p['evals'] for p in probes))
        result['physical_account'] = helper(args.assets, 'v326').physical_account(
            helper(args.assets, 'physical'), result, trace)
        assert result['physical_account']['passed'], result['physical_account']
        details = result['passenger_detail']
        result['clock_accounting'] = {key: dict(timestamp_sum=details[key],
            interval_sum=row[key], timestamp_minus_interval=details[key]-row[key])
            for key in ('waiting_person_s','onboard_person_s')}
        result['endpoints'] = helper(args.assets, 'v324').endpoints(result, 262,
            read(args.assets / 'K4/config.json')['reward_units'])
        early = [max(0., d['public_s'] - d['actual_s']) for d in audit.dispatches]
        result['operating_detail'] = dict(early_departures=sum(v > 0 for v in early),
            early_total_s=sum(early), early_max_s=max(early),
            callback_p50_p95_max_s=__import__('numpy').quantile(audit.callback_wall, [.5, .95, 1.]).tolist())
        result['wall_s'] = time.perf_counter() - begin
        frozen.dump(out / 'raw/path.pkl.gz', trace)
        write(out / 'summary/results.json', result)
        print('V404_CASE_COMPLETE', regime, day, arm, flush=True)
        return result, trace, out
    except Exception:
        write(out / 'summary/failure.json', dict(identity, error=traceback.format_exc()[-7000:]))
        raise


def observer_check(args):
    from types import SimpleNamespace as NS
    p=NS(appear_time=.5,boarding_time=2.,arrive_time=5.)
    station=NS(station_id=1,total_passenger=[p])
    marker=object()
    env=NS(stations=[station],protocol=NS(evaluation_end_time_s=10),
           _compute_measurement_vector=lambda:marker)
    observer=NS()
    saved=capture_at_measurement(env,observer)
    assert env._compute_measurement_vector() is marker
    station.total_passenger.clear()
    assert observer.passenger_detail['generated']==1
    assert observer.passenger_detail['waiting_person_s']==1.5
    assert observer.passenger_detail['onboard_person_s']==3.
    assert observer.passenger_detail['journey_p50_p90_p95_s']==[4.5]*3
    assert saved() is marker
    write(args.root/'observer_check/summary/results.json',dict(completed=True,passed=True,
        physical_starts=0,native_runner_constructions=0,checks=['snapshot before cleanup',
        'unchanged native score return','fractional timestamp arithmetic']))


def pressure_engineering(args):
    from worker import helper
    day=read(HERE/'protocol.json')['engineering_day']
    new,trace,_=collect(args,'engineering_repair',day,'pressure',True)
    old_root=args.root.parent.parent/'FreqDuet-v402-budget-mechanism/run01/engineering/pressure/d2410001'
    old=read(old_root/'summary/results.json');gate=helper(args.assets,'gate_default')
    old_trace=gate.raw(old_root/'raw/path.pkl.gz');errors={};roundoff=[]
    for key in ('results','contract','execution','coverage','action_chain','slow','quota_probes'):
        d=gate.difference(old[key],new[key],key,roundoff)
        if d is not None:errors[key]=d
    for key in ('holds','dispatches','arrivals','departures','queries','uniforms','returns',
                'slow_events','publish_ticks','public_clocks'):
        d=gate.difference(old_trace[key],trace[key],key,roundoff)
        if d is not None:errors[key]=d
    assert not errors,errors
    sources={name:str(Path(m.__file__).resolve()) for name,m in list(sys.modules.items())
        if getattr(m,'__file__',None) and str(m.__file__).startswith('/home/zhengliang01/scheduleurm_work/')
        and not Path(m.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())}
    write(args.root/'pressure_engineering/summary/runtime_sources.json',sources)
    write(args.root/'pressure_engineering/summary/results.json',dict(completed=True,passed=True,
        differences=errors,diagnostic_roundoff=len(roundoff),physical_starts=1,
        clock_accounting=new['clock_accounting']))


def package_runtime(args):
    import subprocess
    from bundle import build,seal
    assert read(args.root/'pressure_engineering/summary/results.json')['passed']
    f=bind(args.assets)
    f.base.fixed.pilot.joint.optimized_runtime(args.root/'runtime/import_check')
    dest=build(args,read(args.root/'pressure_engineering/summary/runtime_sources.json'))
    env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'}
    subprocess.run([sys.executable,'-I',str(dest/'run.py'),'import_check','--root',str(args.root)],
                   env=env,check=True)
    receipt=seal(dest)
    summary=args.root/'package_runtime/summary'
    summary.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(receipt['source_bundle'],summary/'freqduet_source.tar.gz')
    write(summary/'results.json',dict(completed=True,passed=True,**receipt,physical_starts=0))


def portable_case(args):
    import subprocess
    dest=args.root.parent/'portable_runtime'
    assert read(args.root/'package_runtime/summary/results.json')['passed']
    env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'}
    subprocess.run([sys.executable,'-I',str(dest/'run.py'),'case','--root',str(args.root),
        '--regime',args.regime,'--day',str(args.day),'--arm',args.arm],env=env,check=True)


def native_gate(args):
    plan=read(HERE/'protocol.json');shared=plan['shared_native_case']
    row=read(args.root/'cases'/shared['regime']/str(shared['day'])/shared['arm']/'summary/results.json')
    assert row['physical_account']['passed'] and row['graph']['actual_overrides']==0
    write(args.root/'native_gate/summary/results.json',dict(completed=True,passed=True,
        physical_starts=0,shared_case=shared,native_overrides=0,
        portable_traffic_completed_without_legacy_imports=True))


def engineering(args):
    import subprocess
    from bundle import build, seal
    from worker import helper
    day = read(HERE / 'protocol.json')['engineering_day']
    new, trace, _ = collect(args, 'engineering', day, 'pressure', True)
    old_root = args.root.parent.parent / 'FreqDuet-v402-budget-mechanism/run01/engineering/pressure/d2410001'
    old = read(old_root / 'summary/results.json')
    gate = helper(args.assets, 'gate_default')
    old_trace = gate.raw(old_root / 'raw/path.pkl.gz')
    errors, roundoff = {}, []
    for key in ('results', 'contract', 'execution', 'coverage', 'action_chain', 'slow', 'quota_probes'):
        d = gate.difference(old[key], new[key], key, roundoff)
        if d is not None:
            errors[key] = d
    for key in ('holds', 'dispatches', 'arrivals', 'departures', 'queries', 'uniforms',
                'returns', 'slow_events', 'publish_ticks', 'public_clocks'):
        d = gate.difference(old_trace[key], trace[key], key, roundoff)
        if d is not None:
            errors[key] = d
    assert not errors, errors
    portable = build(args)
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    subprocess.run([sys.executable, '-I', str(portable / 'run.py'), 'import_check',
                    '--root', str(args.root)], env=env, check=True)
    subprocess.run([sys.executable, '-I', str(portable / 'run.py'), 'native_engineering',
                    '--root', str(args.root)], env=env, check=True)
    native = read(args.root / 'engineering/engineering' / str(day) / 'fixed_bilateral/summary/results.json')
    assert native['graph']['actual_overrides'] == 0
    output = dict(completed=True, passed=True, exact_pressure_reproduction=True,
        diagnostic_roundoff=len(roundoff), differences=errors, native_overrides=0,
        physical_starts=2, training_updates=0, portable_runtime=seal(portable),
        portable_traffic_completed_without_legacy_imports=True)
    write(args.root / 'engineering/summary/results.json', output)


def aggregate(args):
    import numpy as np
    from worker import helper
    plan = read(HERE / 'protocol.json')
    data, details = {}, {}
    metrics = ('journey_s_per_passenger', 'waiting_s_per_passenger', 'onboard_s_per_passenger',
               'public_delay_s_per_day', 'readiness_s_per_trip', 'cv_squared', 'holding_s_per_trip')
    gate = helper(args.assets, 'gate_default')
    physical = helper(args.assets, 'physical')
    checks = []
    for regime in plan['regimes']:
        for day in plan['days'][regime]:
            reference = None
            for arm in plan['arms']:
                out = args.root / 'cases' / regime / str(day) / arm
                row = read(out / 'summary/results.json')
                assert row['physical_account']['passed']
                trace = gate.raw(out / 'raw/path.pkl.gz')
                if reference is None:
                    reference = trace
                paired = physical.crn_pair(reference['uniforms'], trace['uniforms'])
                assert paired['exact'] and paired['common_events'] == 5200
                checks.append([regime, day, arm, True])
                data[regime, day, arm] = row['endpoints']
                details[regime, day, arm] = dict(passengers={k: v for k, v in row['passenger_detail'].items()
                    if k != 'station_hour'}, operating=row['operating_detail'], slow=row['slow'],
                    supported=row['graph']['supported_queries'], overrides=row['graph']['actual_overrides'],
                    wall_s=row['wall_s'])
    effects, means, readouts = {}, {}, {}
    for regime in plan['regimes']:
        days = plan['days'][regime]
        rng = np.random.default_rng(plan['bootstrap_seed'])
        draws = rng.integers(0, len(days), size=(plan['bootstrap_draws'], len(days)))
        def values(arm, metric):
            return np.array([data[regime, d, arm][metric] for d in days])
        comparisons = {
            'joint_vs_fixed_model': (('pressure', 1.), ('fixed_model', -1.)),
            'plan_at_budget': (('pressure', 1.), ('budget', -1.)),
            'plan_without_budget': (('plan', 1.), ('fixed_model', -1.)),
            'budget_at_plan': (('pressure', 1.), ('plan', -1.)),
            'budget_without_plan': (('budget', 1.), ('fixed_model', -1.)),
            'interaction': (('pressure', 1.), ('budget', -1.), ('plan', -1.), ('fixed_model', 1.)),
            'joint_vs_bilateral': (('pressure', 1.), ('fixed_bilateral', -1.)),
            'pressure_vs_static': (('pressure', 1.), ('static_7p5', -(1-plan['static_high_probability'])),
                                   ('static_10', -plan['static_high_probability'])),
            'pressure_vs_cap': (('pressure', 1.), ('cap_10', -(1-plan['cap_high_probability'])),
                                ('cap_15', -plan['cap_high_probability']))}
        effects[regime] = {}
        for name, components in comparisons.items():
            effects[regime][name] = {}
            for metric in metrics:
                v = sum(w * values(arm, metric) for arm, w in components)
                effects[regime][name][metric] = dict(mean=float(v.mean()),
                    descriptive_ci95=np.quantile(v[draws].mean(1), [.025, .975]).tolist(),
                    by_day=dict(zip(map(str, days), v.tolist())))
        means[regime] = {arm: {m: float(values(arm, m).mean()) for m in metrics} for arm in plan['arms']}
        readouts[regime] = {arm: {str(d): details[regime, d, arm] for d in days} for arm in plan['arms']}
    output = dict(completed=True, physical_starts=146, training_updates=0, main_policy_changed=False,
        statistical_unit='four independent demand days per fixed regime; descriptive paired intervals',
        means=means, effects=effects, details=readouts, physical_pairing_checks=len(checks),
        historical_gates_unchanged=True, independent_route_transfer=False)
    write(args.root / 'aggregate/summary/results.json', output)
    print('V404_AGGREGATE_COMPLETE', len(checks), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('command', choices=('preflight', 'engineering', 'case', 'aggregate',
        'observer_check','pressure_engineering','package_runtime','portable_case','native_gate'))
    p.add_argument('--regime')
    p.add_argument('--day', type=int)
    p.add_argument('--arm')
    a = p.parse_args()
    globals()[a.command](a)
