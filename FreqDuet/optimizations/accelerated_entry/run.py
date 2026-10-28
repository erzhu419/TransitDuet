"""New frozen-controller evaluation paths, without any archived-trajectory input."""

import argparse
import gzip
import json
from pathlib import Path
import pickle
import time
import traceback

import numpy as np

from controllers import CONTROLLERS
from freqduet_native_event_graph import native_query
import run_freqduet_v267_bilateral_factorial as fixed
import run_freqduet_v276_native_continuation as previous


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('preflight', 'run'))
    result.add_argument('--root', type=Path, required=True)
    result.add_argument('--actor', type=int, required=True)
    result.add_argument('--day', type=int, required=True)
    result.add_argument('--sampler', type=int, required=True)
    result.add_argument('--controller', choices=tuple(CONTROLLERS), required=True)
    return result


def preflight(args):
    fixed.pilot.joint.optimized_runtime(args.root)
    joint = fixed.pilot.joint
    checkpoint_root = joint.TRAIN_ROOT/'logs'/f'{joint.CURRENT}_seed{args.actor}'/'checkpoints'
    files = [checkpoint_root/f'{part}_ep{joint.BASE_EP}.pt' for part in ('runner', 'lower', 'upper')]
    if not all(path.is_file() for path in files):
        raise FileNotFoundError(f'Existing frozen actor checkpoint is incomplete: {checkpoint_root}')
    return dict(actor=args.actor, day=args.day, sampler=args.sampler, controller=args.controller,
        checkpoint_files=[str(path) for path in files], checkpoint_ep=joint.BASE_EP,
        reference_trajectory_required=False, training_updates=0, traffic_starts=0)


class Audit(fixed.BenchmarkAudit):
    def __init__(self, actor, controller, credit):
        super().__init__(actor, 'fixed_bilateral025')
        self.controller, self.credit = controller, credit
        self.bridge = None
        self.decisions, self.callback_wall = [], []

    def attach(self, runner):
        super().attach(runner)
        from env.bus import Bus
        self.bridge = CONTROLLERS[self.controller](self.env, self.launches, self.credit)
        self.bridge.attach(Bus)
        self.started = time.perf_counter()

    def __call__(self, policy, states, features, theta):
        begin = time.perf_counter()
        law = super().__call__(policy, states, features, theta)
        bus = self.runner._bus_for_agent(self.runner._pilot_agent)
        legal = tuple(int(fixed.pilot.step.ACTIONS[i]) for i in np.flatnonzero(
            policy.feasible_action_mask(states)[0].numpy()))
        native = int(fixed.pilot.step.ACTIONS[int(law.argmax(dim=1).item())])
        query = native_query(self.env, bus, legal, native, self.credit.stream.pending_commands)
        action = self.bridge.observe(bus, legal, native)['shadow_action_s']
        query['executed_request_s'] = int(action)
        self.decisions.append(query)
        self.callback_wall.append(time.perf_counter()-begin)
        if len(self.decisions) % 100 == 0:
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='accelerated_frozen_evaluation',
                current=float(self.env.current_time), total=float(self.env.protocol.evaluation_end_time_s),
                unit='sim_seconds', queries=len(self.decisions), elapsed_s=time.perf_counter()-self.started)), flush=True)
        return previous.selected_law(law, int(action), legal)

    def detach(self):
        if self.bridge is not None:
            self.bridge.detach()
        super().detach()

    def finish(self, runner, records, holds):
        super().finish(runner, records, holds)
        self.chain = previous.executed_chain(self.decisions, holds)
        assert self.execution['done_reason'] == 'service_cleared'
        assert self.execution['unexecuted_s'] == 0


def run_one(args):
    readiness = preflight(args)
    args.root.mkdir(parents=True, exist_ok=False)
    write(args.root/'preflight.json', readiness)
    write(args.root/'started/path.json', dict(actor=args.actor, day=args.day, sampler=args.sampler,
        controller=args.controller, automatic_retry=False, training_updates=0))
    begin = time.perf_counter()
    credit = fixed.pilot.CreditObserver()
    audit = Audit(args.actor, args.controller, credit)
    try:
        row, uniforms, _, holds = fixed.pilot.rollout(args.root, args.actor, args.day, 'stochastic',
            args.sampler, [0.]*4, 'accelerated_path', probability_rule=audit,
            trace_observer=audit, credit_observer=credit)
        row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
            backlog_trip_s=float(credit.stream.total[4]), holding_person_s=audit.holding_person_s,
            **audit.dispatch, **audit.delays)
        previous.label_resources(audit.bridge, audit.dispatches, holds)
        trace = dict(holds=holds, dispatches=audit.dispatches, arrivals=audit.arrivals,
            departures=audit.departures, queries=audit.decisions, uniforms=uniforms,
            graph_queries=audit.bridge.queries, resource_predictions=audit.bridge.predictions)
        destination = args.root/'traces/path.pkl.gz'
        destination.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(destination, 'xb') as stream:
            pickle.dump(trace, stream, pickle.HIGHEST_PROTOCOL)
        result = dict(status='complete_frozen_evaluation', actor=args.actor, day=args.day,
            sampler=args.sampler, controller=args.controller, results=row, contract=audit.contract,
            coverage=audit.coverage, execution=audit.execution, action_chain=audit.chain,
            graph=audit.bridge.summary(), wall_s=time.perf_counter()-begin,
            callback_latency_s=dict(mean=float(np.mean(audit.callback_wall)),
                p95=float(np.quantile(audit.callback_wall, .95)), max=max(audit.callback_wall)),
            reference_trajectory_required=False, training_updates=0, main_policy_changed=False)
        write(args.root/'summary/results.json', result)
        print('ACCELERATED_FROZEN_EVALUATION_COMPLETE', flush=True)
        return result
    except Exception as error:
        write(args.root/'summary/failure.json', dict(error=repr(error),
            traceback=traceback.format_exc()[-5000:], automatic_retry=False,
            training_updates=0, main_policy_changed=False))
        raise


if __name__ == '__main__':
    args = parser().parse_args()
    if args.command == 'preflight':
        print(json.dumps(preflight(args), indent=2), flush=True)
    else:
        run_one(args)
