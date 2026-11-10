"""Independent slow-layer harmonic summary from the existing causal APC stream.

The native frequency tracker remains the fast-layer source. Its completed-bin
count supplies the clock only; no native tracker or environment state is edited.
"""

import math

import numpy as np

from frequency.intensity_estimator import CausalHarmonicBandState, fit_harmonic_prior


SIGNAL_KEYS = ('freq_low_demand', 'freq_low_slope', 'freq_low_forecast',
               'freq_high_energy', 'freq_updates')


def historical_rates(stations, interval_s, period_s, start_hour=6, end_hour=19):
    """Same global hourly OD accumulation as the frozen native prior builder."""
    rates = np.zeros(int(round(period_s / interval_s)), dtype=np.float64)
    for station in stations:
        if station.od is None:
            continue
        for index in range(len(rates)):
            hour = max(start_hour, min(end_hour,
                start_hour + int(index * interval_s // 3600)))
            values = station.od.get(f'{hour:02}:00:00', {})
            if isinstance(values, dict):
                total = 0.
                for value in values.values():
                    value = float(value)
                    if value > 0.:
                        total += value
                rates[index] += total / 60.
    return rates


def field(event, name):
    return event[name] if isinstance(event, dict) else getattr(event, name)


class UpperSummaryProvider:
    """One episode's independent global filter, refreshed once per slow tick."""

    @classmethod
    def from_environment(cls, env, k, ridge=.01):
        native = env.frequency_tracker
        assert env.frequency_observation_source == 'apc_boardings'
        saved = dict(vars(native), global_state=vars(native.global_state))
        rates = historical_rates(env.stations, float(native.bin_interval_s),
            float(native.global_state.period_s),
            int(getattr(env, 'service_start_hour', 6)),
            int(getattr(env, 'service_end_hour', 19)))
        terminal = {(int(s.station_id), bool(s.direction)) for s in env.stations
                    if int(s.station_type) == 0}
        return cls.from_saved(saved, k, rates, terminal, ridge)

    @classmethod
    def from_saved(cls, saved, k, rates, terminal_keys, ridge=.01):
        assert k in (0, 4) and saved['method'] == 'harmonic'
        assert saved['forecast_mode'] == 'causal'
        assert not (saved['promotion_enabled'] and saved['promotion_adapt_low'])
        global_saved = saved['global_state']
        interval = float(saved['bin_interval_s'])
        assert interval == float(global_saved['update_interval_s']) == 60.
        assert saved['bin_steps'] * saved['update_interval_s'] == interval
        period = float(global_saved['period_s'])
        original_prior = fit_harmonic_prior(rates, interval, period, 4, ridge)
        np.testing.assert_allclose(original_prior, global_saved['initial_theta'],
            rtol=1e-12, atol=1e-12,
            err_msg='upper_provider_historical_source_does_not_match_native_prior')
        prior = (np.asarray(global_saved['initial_theta']).copy() if k == 4 else
                 fit_harmonic_prior(rates, interval, period, 0, ridge))
        provider = cls()
        provider.k, provider.interval_s = int(k), interval
        provider.terminal_keys = set(terminal_keys)
        provider.global_demand_norm = float(saved['global_demand_norm'])
        provider.slope_norm = float(saved['slope_norm'])
        provider.forecast_steps = float(saved['forecast_steps'])
        provider.state = CausalHarmonicBandState(
            update_interval_s=interval, period_s=period, fourier_k=k,
            forgetting_factor=global_saved['forgetting_factor'],
            prior_var=global_saved['prior_var'],
            energy_alpha=global_saved['energy_alpha'],
            residual_alpha=global_saved['residual_alpha'],
            middle_alpha=global_saved['middle_alpha'], prior_theta=prior)
        provider.reset()
        return provider

    def reset(self):
        self.state.reset()
        self.cursor, self.updates = 0, 0
        self.pending = {}
        self.last_arrival_s, self.last_query_s = -1., -1.
        self.last_bus_clock = set()
        self.summary = self._summary()

    def _summary(self):
        state = self.state
        return dict(
            freq_low_demand=float(state.low / self.global_demand_norm),
            freq_low_slope=float(state.low_slope / self.slope_norm),
            freq_low_forecast=float(max(0., float(state.forecast(self.forecast_steps))) /
                                    self.global_demand_norm),
            freq_high_energy=float(math.sqrt(max(state.high_energy, 0.)) /
                                   self.global_demand_norm),
            freq_updates=int(self.updates))

    def refresh(self, now_s, native_updates, flow_events):
        """Consume new APC events, then only bins already closed by native code.

        Native bins contain arrivals in (end-60, end], including any time-zero
        event in the first bin. Events from the currently open bin are retained
        locally until the native tracker closes it.
        """
        now_s, native_updates = float(now_s), int(native_updates)
        assert self.last_query_s <= now_s
        assert self.updates <= native_updates
        assert native_updates * self.interval_s < now_s or now_s == 0.
        assert self.cursor <= len(flow_events)
        for index in range(self.cursor, len(flow_events)):
            event = flow_events[index]
            clock = float(field(event, 'arrival_s'))
            assert int(field(event, 'order')) == index
            assert self.last_arrival_s <= clock <= now_s
            assert float(field(event, 'available_s')) == clock
            key = int(field(event, 'bus')), clock
            if clock != self.last_arrival_s:
                self.last_bus_clock.clear()
            assert key not in self.last_bus_clock
            self.last_bus_clock.add(key)
            self.last_arrival_s = clock
            station = int(field(event, 'station')), bool(field(event, 'direction'))
            if station in self.terminal_keys:
                continue
            boarded = float(field(event, 'boarded'))
            assert boarded >= 0. and boarded == int(boarded)
            slot = max(0, int(math.ceil(clock / self.interval_s)) - 1)
            assert slot >= self.updates, 'APC arrived after its bin was consumed'
            self.pending[slot] = self.pending.get(slot, 0.) + boarded
        self.cursor = len(flow_events)
        while self.updates < native_updates:
            count = self.pending.pop(self.updates, 0.)
            self.state.update(count * 60. / self.interval_s, step=self.updates)
            self.updates += 1
        assert self.state.n == self.updates
        self.last_query_s = now_s
        self.summary = self._summary()
        assert all(np.isfinite(value) for value in self.summary.values())
        return self.summary
