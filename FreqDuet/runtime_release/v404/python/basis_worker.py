"""V350 shared-basis wiring with the frozen V326 collection/training recipe."""

import argparse
import copy
import gzip
import json
from pathlib import Path
import pickle
import random
import time
import traceback

import numpy as np
import torch

import runner as base
import law
import coupled_model
import basis_integration as integration
from basis_integration import ACTION_VALUES, STATE_DIM, LearnedAudit, ObservedUpperTrainer, RewardUnits


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False,
                         separators=(',', ':'), default=lambda value: value.item())
    if path.name == 'results.json':
        assert len(encoded.encode()) <= 48*1024, 'Summary exceeds the small-fetch bound'
    with path.open('x') as stream:
        stream.write(encoded+'\n')


def read_config(path):
    config = json.loads(path.read_text())
    assert config['action_values'] == list(ACTION_VALUES) == [2, 3]
    assert STATE_DIM == len(config['state_divisors']) == 180
    assert Path(integration.__file__).resolve().parent == Path(__file__).resolve().parent
    days = [day for batch in config['train_batches'] for day in batch]
    assert len(days) == len(set(days))
    assert not set(days) & set(config['evaluation_days'])
    assert all(len(batch) == 4 for batch in config['train_batches'])
    assert len(config['learning_seeds']) == len(set(config['learning_seeds']))
    assert config['behavior_epsilon'] == .1 and config['evaluation_deterministic'] is True
    # In this project the fixed native actor is embedded in the frozen Audit.
    assert config['fast_actor'] == base.ACTOR and config['fast_sampler'] == base.SAMPLER
    expected = Path(config['base_source']).resolve()
    for module in (base, law, coupled_model):
        assert Path(module.__file__).resolve().parent == expected, module.__file__
    return config


def require_preflight(config, engineering_k4=False):
    receipt = json.loads(Path(config['preflight_result']).read_text())
    assert receipt['completed'] and receipt['engineering_passed'] and receipt['k4_exact']
    assert receipt['protocol_id'] == 'v350_live_basis_wiring'
    assert all(receipt[key] == 0 for key in ('physical_starts', 'training_updates',
                                            'model_evaluations', 'native_calls'))
    if not engineering_k4:
        gate = json.loads(Path(config['rollout_gate_result']).read_text())
        assert gate['completed']
        assert gate['physical_starts'] == 1 and gate['training_updates'] == 0
        admission = json.loads(Path(config['operational_admission_result']).read_text())
        assert admission['completed'] and admission['engineering_passed']
        assert admission['admit_matched_K0_training'] and admission['operational_reproduction']
        assert admission['original_gate_result'] == config['rollout_gate_result']
        assert admission['physical_starts'] == 0 and admission['training_updates'] == 0


def bind_runtime(config, results_root):
    """Bind asset locations in this process; frozen source/configs stay intact."""
    assets = json.loads(Path(config['runtime_assets_result']).read_text())
    traffic = json.loads(Path(config['traffic_assets_result']).read_text())
    assert assets['completed'] and assets['phase'] == 'training'
    assert traffic['completed'] and traffic['engineering_passed']
    bindings = assets['runtime_bindings']
    assert Path(results_root).resolve() == Path(bindings['traffic_results_root']).resolve()
    joint = base.fixed.pilot.joint
    joint.MODEL_ROOT = Path(bindings['model_root'])
    joint.TRAIN_ROOT = Path(bindings['native_train_root'])
    joint.optimized_runtime.__globals__['OPTIMIZED_ROOT'] = Path(bindings['optimized_code_root'])
    import runner_v3
    runner_v3.SCRIPT_DIR = Path(bindings['runner_script_dir'])
    return bindings


def action_library(trainer):
    expected = torch.as_tensor([[2.], [3.]],device=trainer.device)
    for values in (trainer.discrete_actions,trainer.policy_net.action_candidates,
                   trainer.q_net.action_candidates,trainer.target_q_net.action_candidates):
        assert values.shape == (2, 1) and torch.equal(values,expected)


def transition_actions(transitions):
    previous = None
    for index,transition in enumerate(transitions):
        action = np.asarray(transition['action'])
        assert action.shape == (1,) and float(action[0]) in ACTION_VALUES
        state,next_state = np.asarray(transition['state']),np.asarray(transition['next_state'])
        assert state.shape == next_state.shape == (STATE_DIM,)
        expected = np.zeros(4,dtype=np.float32)
        if previous is not None:
            expected[previous] = 1.
        np.testing.assert_array_equal(state[12:16],expected)
        if not transition['done']:
            following = np.zeros(4,dtype=np.float32)
            following[int(action[0])] = 1.
            np.testing.assert_array_equal(next_state[12:16],following)
            np.testing.assert_array_equal(next_state,transitions[index+1]['state'])
        else:
            assert index == len(transitions)-1
            np.testing.assert_array_equal(next_state,np.zeros(STATE_DIM,dtype=np.float32))
        previous = int(action[0])


def action_semantics(manager):
    choices,transitions = manager.choices,manager.transitions
    publications = [e for e in manager.events if e['kind'] == 'publish']
    assert len(choices) == len(transitions) == len(publications) > 0
    transition_actions(transitions)
    for choice,event,transition in zip(choices,publications,transitions):
        action = choice['action']
        assert action in ACTION_VALUES and choice['mode'] == integration.MODES[action]
        assert event['learned_action'] == action == int(transition['action'][0])
        assert event['time_s'] == choice['time_s'] == transition['start_s']
        assert 'permission' not in event and 'plan_frequency' not in event
        assert not event['plans'] or action == 3
        pi,mu = np.asarray(choice['pi']),np.asarray(choice['mu'])
        assert pi.shape == mu.shape == (2,) and np.isfinite(pi).all() and np.isfinite(mu).all()
        assert (pi >= 0).all() and (mu >= 0).all()
        np.testing.assert_allclose([pi.sum(),mu.sum()],[1.,1.],atol=1e-6,rtol=0)
        if manager.deterministic:
            assert action == ACTION_VALUES[int(pi.argmax())]
            expected = np.zeros(2)
            expected[ACTION_VALUES.index(action)] = 1.
        else:
            expected = (1.-manager.behavior_epsilon)*pi+manager.behavior_epsilon/2.
        np.testing.assert_allclose(mu,expected,atol=1e-7,rtol=0)
    assert manager.previous_action == choices[-1]['action'] and manager.arm == choices[-1]['mode']
    return dict(passed=True,action_values=list(ACTION_VALUES),policy_heads=2,records=len(choices),
        actor_critic_execution_labels_equal=True,previous_action_slots=[12,13,14,15],
        previous_action_actual_canonical=True,permission_override=False)


def make_trainer(config, seed):
    assert seed in config['learning_seeds']
    streams = np.random.SeedSequence(seed).spawn(3)
    initial, policy, replay = [int(s.generate_state(1)[0]) for s in streams]
    random.seed(initial)
    np.random.seed(initial)
    torch.manual_seed(initial)
    settings = dict(config['trainer_settings'])
    settings.update(policy_sample_seed=policy, replay_seed=replay)
    trainer = ObservedUpperTrainer(**settings)
    action_library(trainer)
    return trainer


def load_policy(checkpoint, config, seed, engineering_k4=False):
    receipt = json.loads((checkpoint/'summary/results.json').read_text())
    assert receipt['status'] in ('complete_upper_initialize', 'complete_upper_batch')
    policy_path = checkpoint/'policy.pt'
    envelope = torch.load(policy_path, map_location='cpu', weights_only=False)
    expected_config = config
    if engineering_k4:
        expected_config = json.loads(Path(config['reference_learning_config']).read_text())
        assert seed == 314011 and envelope['completed_batches'] == 8
        assert checkpoint.resolve() == Path(config['reference_checkpoint']).resolve()
        for key in ('learning_seeds', 'candidate_seed', 'train_batches', 'evaluation_days',
                    'evaluation_deterministic', 'behavior_epsilon', 'reward_units',
                    'state_divisors', 'trainer_settings', 'updates_per_batch', 'batch_size',
                    'fast_actor', 'fast_sampler', 'action_values'):
            assert config[key] == expected_config[key], key
    assert envelope['protocol'] == expected_config and envelope['learner_seed'] == seed
    assert receipt['completed_batches'] == envelope['completed_batches']
    assert receipt['successful_updates'] == envelope['successful_updates']
    trainer = make_trainer(config, seed)
    trainer.policy_net.load_state_dict(envelope['policy'])
    action_library(trainer)
    return trainer, envelope


def load_training_checkpoint(checkpoint, config, seed):
    receipt = json.loads((checkpoint/'summary/results.json').read_text())
    assert receipt['status'] in ('complete_upper_initialize','complete_upper_batch')
    saved = torch.load(checkpoint/'training.pt',map_location='cpu',weights_only=False)
    assert saved['protocol'] == config and saved['learner_seed'] == seed
    assert receipt['completed_batches'] == saved['completed_batches']
    assert receipt['successful_updates'] == saved['successful_updates']
    trainer = make_trainer(config,seed)
    trainer.load_training_state_dict(saved['trainer_state'])
    trainer.successful_updates = saved['successful_updates']
    action_library(trainer)
    assert all(np.asarray(row[1]).shape == (1,) and float(row[1][0]) in ACTION_VALUES
               for row in saved['trainer_state']['replay_buffer']['buffer'])
    return trainer,saved


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'xb') as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def run_day(args):
    config = read_config(args.config)
    engineering_k4 = bool(getattr(args, 'engineering_k4', False))
    require_preflight(config, engineering_k4)
    bind_runtime(config, args.results)
    if engineering_k4:
        assert args.split == 'development' and args.day == 2077002 and args.seed == 314011
    else:
        assert config['upper_fourier_k'] == 0
    trainer, checkpoint = load_policy(args.checkpoint, config, args.seed, engineering_k4)
    assert checkpoint['successful_updates'] == checkpoint['completed_batches']*config['updates_per_batch']
    if args.split == 'train':
        assert args.batch == checkpoint['completed_batches']
        assert args.day in config['train_batches'][args.batch]
        assert args.day not in checkpoint['consumed_days']
        deterministic = False
        behavior_epsilon = float(config['behavior_epsilon'])
    else:
        assert checkpoint['completed_batches'] == len(config['train_batches'])
        assert args.day in config['evaluation_days'] and args.batch is None
        deterministic = bool(config['evaluation_deterministic'])
        behavior_epsilon = 0.
    # Each path has its own sampler; all four workers use identical policy weights.
    sample_seed = int(np.random.SeedSequence([args.seed, args.day]).generate_state(
        1, dtype=np.uint64)[0])
    generator = torch.Generator(device=trainer.device)
    generator.manual_seed(sample_seed)
    trainer.policy_net.set_sampling_state(generator.get_state())
    before = copy.deepcopy(trainer.policy_net.state_dict())
    assert not len(trainer.replay_buffer) and trainer.successful_updates == 0
    args.out.mkdir(parents=True, exist_ok=False)
    identity = dict(protocol_id=config['protocol_id'], learner_seed=args.seed,
        day=args.day, split=args.split, batch=args.batch,
        checkpoint=str(args.checkpoint.resolve()), policy_sampling_seed=sample_seed,
        upper_deterministic=deterministic, behavior_epsilon=behavior_epsilon,
        upper_fourier_k=4 if engineering_k4 else 0, engineering_k4=engineering_k4,
        source=str(Path(__file__).resolve().parent),
        base_source=str(Path(base.__file__).resolve().parent))
    write(args.out/'started.json', dict(identity, automatic_retry=False,
        physical_start_cap=1, training_updates=0))
    label = f'{args.split}_s{args.seed}_b{args.batch}_d{args.day}'
    if engineering_k4:
        label = 'engineering_k4_'+label
    runtime = args.results/(config['runtime_prefix']+'_'+label)
    begin, starts = time.perf_counter(), 0
    credit = base.fixed.pilot.CreditObserver()
    audit = LearnedAudit(credit, trainer, RewardUnits(**config['reward_units']),
                         config['state_divisors'], deterministic, behavior_epsilon,
                         upper_fourier_k=4 if engineering_k4 else 0)
    try:
        runtime.mkdir(parents=True, exist_ok=False)
        starts = 1
        row, uniforms, _, holds = base.fixed.pilot.rollout(runtime,
            config['fast_actor'], args.day, 'stochastic', config['fast_sampler'],
            [0.]*4, label, probability_rule=audit, trace_observer=audit,
            credit_observer=credit)
        assert trainer.successful_updates == 0 and not len(trainer.replay_buffer)
        for name, value in before.items():
            torch.testing.assert_close(trainer.policy_net.state_dict()[name], value,
                                       atol=0., rtol=0.)
        semantics = action_semantics(audit.manager)
        row.update(waiting_person_s=float(credit.stream.total[0]),
            onboard_person_s=float(credit.stream.total[1]),
            backlog_trip_s=float(credit.stream.total[4]),
            holding_person_s=audit.holding_person_s, **audit.dispatch, **audit.delays)
        base.previous.label_resources(audit.bridge, audit.dispatches, holds)
        dump(args.out/'raw/path.pkl.gz', dict(holds=holds, dispatches=audit.dispatches,
            arrivals=audit.arrivals, departures=audit.departures, queries=audit.decisions,
            uniforms=uniforms, returns=credit.stream.returns, graph_queries=audit.bridge.queries,
            resource_predictions=audit.bridge.predictions, slow_events=audit.manager.events,
            publish_ticks=audit.publish_ticks, public_clocks=audit.public_clocks,
            upper_choices=audit.manager.choices))
        dump(args.out/'raw/upper_transitions.pkl.gz', dict(identity=identity,
            protocol=config, upper=audit.manager.upper_receipt,
            transitions=audit.manager.transitions,action_semantics=semantics))
        probes = [q['quota_probe'] for q in audit.bridge.queries if 'quota_probe' in q]
        result = dict(identity, status='complete_upper_day', results=row,
            contract=audit.contract, coverage=audit.coverage, execution=audit.execution,
            action_chain=audit.chain, graph=audit.bridge.summary(), slow=audit.manager.summary(),
            upper=audit.manager.upper_receipt,action_semantics=semantics,
            quota_probes=dict(attempted=len(probes),
                supported=sum(p['supported'] is True for p in probes),
                changed=sum(p['changed'] is True for p in probes),
                extra_evaluations=sum(p['evals'] for p in probes)),
            raw_transitions=str((args.out/'raw/upper_transitions.pkl.gz').resolve()),
            raw_trace=str((args.out/'raw/path.pkl.gz').resolve()),
            wall_s=time.perf_counter()-begin, physical_starts=starts,
            training_updates=0, native_actors_unchanged=True,
            independent_upper_unchanged=True, main_policy_changed=False,
            confirmation_expanded=False, automatic_retry=False)
        write(args.out/'summary/results.json', result)
        print('UPPER_DAY_COMPLETE '+json.dumps(identity), flush=True)
        return result
    except Exception:
        write(args.out/'summary/failure.json', dict(identity,
            error=traceback.format_exc()[-7000:], physical_starts=starts,
            training_updates=0, automatic_retry=False, main_policy_changed=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--split', choices=('train', 'development'), required=True)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--day', type=int, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--engineering-k4', action='store_true')
    run_day(parser.parse_args())

