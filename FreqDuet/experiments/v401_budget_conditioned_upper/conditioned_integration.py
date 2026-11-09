"""V401 explicit/implicit budget state with the frozen two-action recipe.

The V350 physical state prefix, reward and native V314 execution remain frozen.
Policy head indices 0/1 represent canonical executed actions 2/3 throughout
the actor, critic, replay and previous-action state. No permission remapping.
"""

from dataclasses import dataclass

import numpy as np
import torch

from runner import Audit as V314Audit
from conditioned_law import SlowManager, START_S, PERIOD_S
from provider import UpperSummaryProvider
from freqduet_fleet_observation import capture_fleet
from freqduet_system_learner import ArrivalMoments
from upper.resac_upper import RESACUpperTrainer, UpperReplayBuffer

MODES = ('fixed', 'plan', 'budget', 'joint')
ACTION_VALUES = (2, 3)
ACTION_CANDIDATES = [[2], [3]]
STATE_DIM = 181  # The unchanged 180 physical features, then one budget channel.
COMPONENTS = ('waiting_person_s', 'onboard_person_s', 'public_delay_s', 'arrival_cv_squared')


def choose_upper_action(trainer, state, deterministic, behavior_epsilon):
    """One draw from the actual training mixture; greedy evaluation draws none."""
    epsilon = float(behavior_epsilon)
    assert 0. <= epsilon <= 1. and (not deterministic or epsilon == 0.)
    policy = trainer.policy_net
    with torch.no_grad():
        pi = policy.dist_info(torch.as_tensor(state, dtype=torch.float32,
            device=trainer.device)[None])[0][0]
        assert pi.shape == (len(ACTION_VALUES),)
        if deterministic:
            index = int(pi.argmax().item())
            mu = torch.zeros_like(pi)
            mu[index] = 1.
        else:
            assert policy._sample_generator is not None
            mu = (1.-epsilon)*pi+epsilon/len(ACTION_VALUES)
            index = int(torch.multinomial(mu, 1, generator=policy._sample_generator).item())
        action = int(policy.action_candidates[index, 0].item())
        assert action == ACTION_VALUES[index]
    return action, pi.cpu().tolist(), mu.cpu().tolist()


@dataclass(frozen=True)
class RewardUnits:
    passenger_scale: float
    public_delay_scale: float
    cv_scale: float
    public_delay_weight: float
    cv_weight: float

    def value(self, parts):
        return ((parts[0]+parts[1])/self.passenger_scale
                + self.public_delay_weight*parts[2]/self.public_delay_scale
                + self.cv_weight*parts[3]/self.cv_scale)


def public_debt(now_s, public, dispatches):
    """Elapsed public obligations, independent of mutable scheduled releases."""
    actual = {r['trip']:r['actual_s'] for r in dispatches}
    return sum(max(0., actual.get(trip, now_s)-clock) for trip, clock in public.items())


def encode_state(manager, now_s, debt, previous_action, divisors):
    """Current APC/AVL, published plans and completed history only; no queues."""
    assert previous_action is None or previous_action in ACTION_VALUES
    env = manager.env
    captured = capture_fleet(env)
    frequency = manager.upper_summary
    fleet = captured['fleet']
    moments = manager.moments
    values = [now_s, captured['horizon_s']-now_s, sum(r['load'] for r in fleet),
        float(frequency['freq_updates'] > 0 and
              frequency['freq_low_demand']+frequency['freq_low_forecast'] > 0),
        *[frequency[k] for k in ('freq_low_demand', 'freq_low_slope',
                                 'freq_low_forecast', 'freq_high_energy')],
        moments.count, moments.total, moments.square_total, debt,
        *[float(previous_action == action) for action in range(4)]]
    for direction in (False, True):
        previous = manager.last_actual[direction]
        # Both directional streams have launched before the 7200 s first action.
        assert previous is not None
        values.extend((sum(not r['on_route'] and r['direction'] == direction for r in fleet),
            sum(r['on_route'] and r['direction'] != direction for r in fleet),
            previous['actual_s']-now_s, previous['release_s']-now_s))
        plans = sorted((p for p in captured['plans'] if not p['launched'] and
            p['direction'] == direction), key=lambda p:manager.public[p['trip_id']])
        for index in range(2):
            if index < len(plans):
                plan = plans[index]
                values.extend((1., manager.public[plan['trip_id']]-now_s,
                    plan['published_release_s']-now_s, plan['target_headway_s']))
            else:
                values.extend((0., 0., 0., 0.))
        history = manager._history(direction, now_s)
        count = sum(r['return_s'] < now_s and r['history_eligible']
                    for r in manager.completed[direction])
        values.extend((count, *([history[k] for k in ('nonholding_s',
            'command_per_stop_s', 'residence_per_stop_s')] if history else [0., 0., 0.])))
    pending = {(c.bus_id, c.trip_id):c for c in manager.credit.stream.pending_commands}
    physical = {int(bus.bus_id):bus for bus in env.bus_all}
    assert sorted(physical) == list(range(12))
    for row in sorted(fleet, key=lambda r:r['bus_id']):
        cycle = row['bus_id'], row['active_trip_id']
        command = pending.get(cycle)
        vehicle = physical[row['bus_id']]
        values.extend((float(row['on_route']), float(row['direction']),
            row['progress_m']/captured['route_lengths_m'][int(row['direction'])],
            row['load']/row['capacity'],
            float(vehicle._freqduet_dispatch_target_headway) if row['on_route'] else 0.,
            float(cycle in manager.remaining), manager.remaining.get(cycle, 0.),
            float(command is not None), 0. if command is None else command.action_s,
            row['remaining_service_s'], row['remaining_command_hold_s']))
    values.append(manager.retain_fraction if manager.budget_observation == 'explicit' else 0.)
    result = np.asarray(values, dtype=np.float32)
    assert result.shape == divisors.shape == (STATE_DIM,)
    assert np.isfinite(result).all() and np.isfinite(divisors).all() and (divisors > 0).all()
    return result/divisors


class LearnedManager(SlowManager):
    """One persistent manager; a sampled mode only changes new publication."""

    def __init__(self, env, credit, audit, trainer, units, divisors, deterministic,
                 behavior_epsilon, upper_provider=None, upper_fourier_k=4, *,
                 retain_fraction, budget_observation):
        super().__init__('fixed', env, credit, retain_fraction)
        assert budget_observation in ('explicit', 'implicit')
        self.budget_observation = budget_observation
        self.upper_provider = (UpperSummaryProvider.from_environment(env, upper_fourier_k)
                               if upper_provider is None else upper_provider)
        self.upper_summary = None
        self.audit, self.trainer, self.units = audit, trainer, units
        self.divisors = np.asarray(divisors, dtype=np.float32)
        self.deterministic = bool(deterministic)
        self.behavior_epsilon = float(behavior_epsilon)
        self.moments, self.previous_action = ArrivalMoments(), None
        self.interval, self.prefix, self.final_parts = None, None, None
        self.transitions, self.choices = [], []
        self.replay_appended = False

    def parts(self, now_s):
        self.moments.observe(self.env.headway_events.headways_s)
        total = self.credit.stream.total
        return np.asarray([total[0], total[1], public_debt(now_s, self.public,
            self.audit.dispatches), self.moments.cv_squared], dtype=np.float64)

    def _close(self, now_s, state, parts, done):
        if self.interval is None:
            return
        start, previous_state, action, initial = self.interval
        assert action in ACTION_VALUES
        delta = parts-initial
        assert now_s > start
        self.transitions.append(dict(state=previous_state, action=np.asarray([action], dtype=np.float32),
            reward=-float(self.units.value(delta)), next_state=state, done=bool(done),
            duration_steps=(now_s-start)/PERIOD_S, start_s=start, end_s=now_s,
            budget_retention=self.retain_fraction, budget_observation=self.budget_observation,
            components=dict(zip(COMPONENTS, delta.tolist()))))

    def publish(self, env, bridge):
        now = float(env.current_time)
        cycle = int((now-START_S)//PERIOD_S)
        # Deduplicate before sampling/credit; env.step can run repeatedly at one tick.
        if now < START_S or cycle == self.last_cycle:
            return
        assert self.credit.stream.clock_s == now
        self.upper_summary = self.upper_provider.refresh(
            now, env.frequency_tracker.total_updates, bridge.prefix.flow_events)
        if self.public is None:
            self.public = dict(self.audit.public_clocks)
        parts = self.parts(now)
        state = encode_state(self, now, parts[2], self.previous_action, self.divisors)
        self._close(now, state, parts, False)
        action, pi, mu = choose_upper_action(self.trainer, state,
                                            self.deterministic, self.behavior_epsilon)
        if self.prefix is None:
            self.prefix = parts.copy()
        self.interval = now, state, action, parts.copy()
        self.previous_action, self.arm = action, MODES[action]
        # No quota restoration, no plan cancellation, no manager recreation.
        super().publish(env, bridge)
        event = self.events[-1]
        assert event['kind'] == 'publish' and event['time_s'] == now
        event['learned_action'] = action
        self.choices.append(dict(time_s=now, action=action, mode=MODES[action], pi=pi, mu=mu,
            behavior_epsilon=self.behavior_epsilon,
            planned_changes=len(event['plans']), budget_issues=len(event['budgets'])))

    def finish_upper(self):
        stream = self.credit.stream
        assert stream.closed and stream.clock_s == self.env.protocol.evaluation_end_time_s
        assert self.interval is not None and self.final_parts is None
        parts = self.parts(stream.clock_s)
        self._close(stream.clock_s, np.zeros(STATE_DIM, dtype=np.float32), parts, True)
        self.final_parts = parts
        assigned = self.prefix + np.sum([[t['components'][k] for k in COMPONENTS]
                                         for t in self.transitions], axis=0)
        np.testing.assert_allclose(assigned, parts, rtol=0., atol=1e-7)
        assert len(self.transitions) == len(self.choices)
        assert sum(t['done'] for t in self.transitions) == 1
        np.testing.assert_allclose(START_S+sum(t['duration_steps']*PERIOD_S for t in self.transitions),
                                   stream.clock_s, rtol=0., atol=1e-7)
        np.testing.assert_allclose(parts[2], self.audit.delays['public_launch_delay_total_s'],
                                   rtol=0., atol=1e-7)
        self.upper_receipt = dict(actions=len(self.choices), transitions=len(self.transitions),
            budget_retention=self.retain_fraction, budget_observation=self.budget_observation,
            behavior_epsilon=self.behavior_epsilon, deterministic=self.deterministic,
            action_values=list(ACTION_VALUES),
            terminal_transitions=1, prefix=dict(zip(COMPONENTS, self.prefix.tolist())),
            endpoint=dict(zip(COMPONENTS, parts.tolist())),
            reward_sum=sum(t['reward'] for t in self.transitions),
            expected_reward=-float(self.units.value(parts-self.prefix)),
            mode_counts={mode:sum(r['mode'] == mode for r in self.choices) for mode in MODES},
            learning_selected_plan_publications=sum(r['action'] == 3 and r['planned_changes'] > 0
                                                    for r in self.choices))

    def append_completed_day_to_replay(self):
        assert self.final_parts is not None and not self.replay_appended
        for row in self.transitions:
            self.trainer.replay_buffer.push(*(row[k] for k in
                ('state', 'action', 'reward', 'next_state', 'done', 'duration_steps')))
        self.replay_appended = True
        return len(self.transitions)


class LearnedAudit(V314Audit):
    def __init__(self, credit, trainer, units, divisors, deterministic, behavior_epsilon,
                 upper_fourier_k=4, *, retain_fraction, budget_observation):
        # This label only selects the inherited dynamic-plan finish contract.
        super().__init__('joint', credit)
        self.upper_fourier_k = upper_fourier_k
        self.learned_trainer, self.units = trainer, units
        self.divisors, self.deterministic = divisors, deterministic
        self.behavior_epsilon = float(behavior_epsilon)
        self.retain_fraction = float(retain_fraction)
        self.budget_observation = budget_observation

    def attach(self, runner):
        super().attach(runner)
        self.manager = LearnedManager(self.env, self.credit, self, self.learned_trainer,
                                     self.units, self.divisors, self.deterministic, self.behavior_epsilon,
                                     upper_fourier_k=self.upper_fourier_k,
                                     retain_fraction=self.retain_fraction,
                                     budget_observation=self.budget_observation)
        self.bridge.manager = self.manager
        self.variant = 'learned'

    def finish(self, runner, records, holds):
        super().finish(runner, records, holds)
        # pilot.rollout invokes this only after CreditObserver finished the tail.
        self.manager.finish_upper()


class TargetAuditReplay(UpperReplayBuffer):
    """Inspect the actual sampled batch before the frozen trainer clamps targets."""

    def __init__(self, owner, capacity, seed):
        super().__init__(capacity, seed)
        self.owner = owner

    def sample(self, batch_size):
        batch = super().sample(batch_size)
        self.owner.inspect_target(batch)
        return batch


class ObservedUpperTrainer(RESACUpperTrainer):
    def __init__(self, **settings):
        # Numeric entropy/optimizer choices are intentionally mandatory inputs.
        for key in ('initial_alpha', 'minimum_alpha', 'maximum_alpha', 'policy_sample_seed',
                    'replay_seed', 'replay_capacity', 'hidden_dim', 'ensemble_size', 'beta',
                    'beta_ood', 'weight_reg', 'weight_reg_mode', 'lr', 'soft_tau',
                    'critic_aggregation', 'device'):
            assert key in settings
        super().__init__(state_dim=STATE_DIM, action_dim=1, action_candidates=ACTION_CANDIDATES,
            discrete_critic='indexed', gamma=1., auto_entropy=False,
            temperature_contract='bounded_log_parameter_v4', **settings)
        assert self.alpha == settings['initial_alpha']
        self.replay_buffer = TargetAuditReplay(self, settings['replay_capacity'], settings['replay_seed'])
        self.successful_updates, self.target_receipt = 0, None
        self.target_failure = None

    def inspect_target(self, batch):
        _, actions, rewards, next_states, dones, duration = batch
        assert np.isin(actions, ACTION_VALUES).all(), 'Replay action is outside the actual budget/joint library'
        with torch.no_grad():
            state = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
            probs, logs, _ = self.policy_net.dist_info(state)
            q = self._aggregate_target_q(self._discrete_q_values(self.target_q_net, state))
            future = (probs*(q-self.alpha*logs)).sum(dim=-1)
            reward = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).squeeze(-1)
            done = torch.as_tensor(dones, dtype=torch.float32, device=self.device).squeeze(-1)
            # gamma=1 exactly; inspect the same pre-update target, without another replay draw.
            target = reward+(1.-done)*future
            finite = torch.isfinite(target)
            nonfinite_count = int((~finite).sum())
            outside_clip = int((finite & (target.abs() > 50.)).sum())
            self.target_receipt = dict(finite=nonfinite_count == 0, nonfinite_count=nonfinite_count,
                raw_target_min=float(target.min()) if not nonfinite_count else None,
                raw_target_max=float(target.max()) if not nonfinite_count else None,
                outside_clip=outside_clip, samples=len(target),
                effective_alpha=float(self.alpha))
            if nonfinite_count or outside_clip:
                if self.target_failure is None:
                    self.target_failure = dict(self.target_receipt)
                raise RuntimeError('upper target rejected before optimizer update: '
                    f'nonfinite={nonfinite_count}, outside_clip={outside_clip}')

    def update(self, batch_size):
        metrics = super().update(batch_size)
        if metrics:
            self.successful_updates += 1
            metrics.update(self.target_receipt)
            metrics['successful_upper_updates'] = self.successful_updates
        return metrics
