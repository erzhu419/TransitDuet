"""Finite-day categorical SAC with once-only passenger credit and a CV constraint."""

from dataclasses import dataclass

import numpy as np
import torch


PASSENGER_HOURS_SCALE = 1000.0
ENTROPY_ALPHA = 1e-5


class ArrivalMoments:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.square_total = 0.0

    def observe(self, headways):
        values = np.asarray(headways[self.count:], dtype=float)
        self.count += len(values)
        self.total += float(values.sum())
        self.square_total += float(values @ values)

    @property
    def cv_squared(self):
        if not self.count:
            return 0.0
        mean = self.total / self.count
        variance = max(self.square_total / self.count - mean * mean, 0.0)
        return variance / max(mean, 1.0) ** 2


@dataclass(frozen=True)
class SystemStateEncoder:
    local_dim: int
    buses: int
    trips: int
    stations: int
    horizon_s: float

    @property
    def dimension(self):
        return self.local_dim + 11 + 6 * self.buses

    def encode(self, state, moments):
        if len(state.local_state) != self.local_dim or not 0 <= state.bus_id < self.buses:
            raise ValueError("decision is outside the registered local state or fleet")
        count_scale = self.trips * self.stations
        header = [state.time_s / self.horizon_s, state.bus_id / self.buses,
                  state.trip_id / self.trips, state.station_id / self.stations,
                  float(state.direction), state.target_headway_s / 600.0,
                  state.ready_origin / self.buses, state.ready_opposite / self.buses,
                  moments.count / count_scale, moments.total / (count_scale * 600.0),
                  moments.square_total / (count_scale * 600.0 ** 2)]
        pending = np.zeros((self.buses, 6), dtype=np.float32)
        for order, command in enumerate(state.pending_commands, start=1):
            if not 0 <= command.bus_id < self.buses or pending[command.bus_id, 0]:
                raise ValueError("pending command slot is outside the fleet or repeated")
            pending[command.bus_id] = [1, command.time_s / self.horizon_s,
                                      command.trip_id / self.trips,
                                      command.station_id / self.stations,
                                      command.action_s / 45.0, order / self.buses]
        result = np.concatenate((state.local_state, header, pending.ravel())).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("non-finite observable system state")
        return result


class ExtendedInputProjection(torch.nn.Module):
    """Keep the original GEMM shape while learning an additive observable projection."""

    def __init__(self, layer, local_dim):
        super().__init__()
        # Retain the Parameter objects already registered with the optimizer.
        self.weight, self.bias = layer.weight, layer.bias
        self.local_dim = local_dim

    def forward(self, state):
        local = torch.nn.functional.linear(state[..., :self.local_dim].contiguous(),
                                           self.weight[:, :self.local_dim].contiguous(), self.bias)
        extra = torch.nn.functional.linear(state[..., self.local_dim:].contiguous(),
                                           self.weight[:, self.local_dim:].contiguous())
        return local + extra


def warm_start_actor(old, new, local_dim):
    """Preserve the old categorical policy; new observable inputs start at zero weight."""
    weights = new.state_dict()
    for name, value in old.state_dict().items():
        if name == "fc1.weight":
            weights[name].zero_()
            weights[name][:, :local_dim].copy_(value)
        else:
            weights[name].copy_(value)
    new.load_state_dict(weights)
    new.fc1 = ExtendedInputProjection(new.fc1, local_dim)


def finite_day_target(immediate, done, continuation):
    return immediate + (1.0 - done) * continuation


class SystemSAC:
    """Reuse RE-SAC networks/aggregation, with finite-day targets and an episodic dual."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.updates = 0

    def dual_step(self, realized_cv_squared, budget):
        trainer = self.trainer
        loss = -trainer.log_lambda.exp() * float(realized_cv_squared - budget)
        trainer.lambda_optimizer.zero_grad()
        loss.backward()
        trainer.lambda_optimizer.step()
        trainer.log_lambda.data.clamp_(min=-5.0, max=1.5)
        return float(trainer.lambda_param)

    def update(self, replay, batch_size):
        t = self.trainer
        state, action, reward, cost, next_state, done, _ = replay.sample(batch_size)
        s, a, r, c, ns, d = [torch.as_tensor(v, dtype=torch.float32, device=t.device)
                             for v in (state, action, reward, cost, next_state, done)]
        with torch.no_grad():
            probs, logs, _ = t.policy_net.dist_info(ns)
            next_q = t._aggregate_target_q(t._discrete_q_values(t.target_q_net, ns))
            continuation = (probs * (next_q - ENTROPY_ALPHA * logs)).sum(-1)
            target_q = finite_day_target(r.flatten(), d.flatten(), continuation)
            next_cost = (probs * t._discrete_cost_values(t.target_cost_q_net, ns)).sum(-1)
            target_cost = finite_day_target(c.flatten(), d.flatten(), next_cost)
        predicted = t.q_net(s, a)
        mse = (predicted - target_q[None, :]).square().mean()
        q_loss = (mse + t.beta_ood * predicted.std(dim=0).mean()
                  + t.weight_reg * t.q_net.compute_l1_norm(mode=t.weight_reg_mode).mean())
        cost_loss = (t.cost_q_net(s, a).flatten() - target_cost).square().mean()
        for network, optimizer, loss in ((t.q_net, t.q_optimizer, q_loss),
                                          (t.cost_q_net, t.cost_q_optimizer, cost_loss)):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 50.0)
            optimizer.step()
        # Critic weights are fixed during the actor step; action values still
        # affect the categorical expectation through the actor probabilities.
        with torch.no_grad():
            q, _, _ = t._policy_q_value(t._discrete_q_values(t.q_net, s))
            cq = t._discrete_cost_values(t.cost_q_net, s)
        probs, logs, _ = t.policy_net.dist_info(s)
        actor_loss = (probs * (ENTROPY_ALPHA * logs - q + float(t.lambda_param) * cq)).sum(-1).mean()
        t.policy_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(t.policy_net.parameters(), 50.0)
        t.policy_optimizer.step()
        with torch.no_grad():
            for target, source in ((t.target_q_net, t.q_net), (t.target_cost_q_net, t.cost_q_net)):
                for tp, sp in zip(target.parameters(), source.parameters()):
                    tp.lerp_(sp, t.soft_tau)
        self.updates += 1
        metrics = dict(q_loss=float(q_loss), cost_q_loss=float(cost_loss),
                       actor_loss=float(actor_loss), q_target_abs_max=float(target_q.abs().max()),
                       cost_target_min=float(target_cost.min()),
                       lambda_value=float(t.lambda_param), updates=self.updates)
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError("non-finite finite-day learning update")
        return metrics


def ingest_day(replay, stream, states, potentials, final_cv_squared, day_id):
    if len(states) != len(stream.transitions) or len(potentials) != len(states):
        raise ValueError("system replay states do not align with physical decisions")
    reward_sum, cost_sum = 0.0, 0.0
    for index, transition in enumerate(stream.transitions):
        terminal = transition.done
        successor = np.zeros_like(states[index]) if terminal else states[index + 1]
        after = final_cv_squared if terminal else potentials[index + 1]
        cost = after - potentials[index]
        reward = transition.reward / PASSENGER_HOURS_SCALE
        replay.push(states[index], np.array([transition.action_s], dtype=np.float32),
                    reward, cost, successor, terminal, day_id)
        reward_sum += reward
        cost_sum += cost
    expected_reward = -(stream.total[0] + stream.total[1] - stream.prefix[0] - stream.prefix[1])
    expected_reward /= 3600.0 * PASSENGER_HOURS_SCALE
    np.testing.assert_allclose(reward_sum, expected_reward, rtol=0, atol=1e-10)
    np.testing.assert_allclose(cost_sum + potentials[0], final_cv_squared, rtol=0, atol=1e-10)
    return dict(replay_reward_sum=reward_sum, cv_increment_sum=cost_sum,
                cv_prefix=potentials[0], cv_squared=final_cv_squared,
                replay_added=len(states), day_terminals=sum(t.done for t in stream.transitions))
