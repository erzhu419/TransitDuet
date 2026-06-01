import json
import time
import numpy as np
import pandas as pd
import copy
import os
import sys
import pygame

# Ensure the project root (parent of this env package) is on sys.path so that
# `import env.*` works even when running env/sim.py directly.
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.timetable import Timetable
from env.bus import Bus
from env.route import Route
from env.station import Station
from env.visualize import visualize
from frequency import DemandEventLogger, DemandFrequencyTracker, fit_harmonic_prior
from gym.spaces.box import Box
from gym.spaces import MultiDiscrete


class env_bus(object):

    def __init__(self, path, debug=False, render=False, route_sigma=1.5):
        self.path = path
        self.route_sigma = route_sigma
        sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'r') as f:
            args = json.load(f)
        self.args = args
        self.effective_trip_num = 264
        
        self.time_step = args["time_step"]
        self.passenger_update_freq = args["passenger_state_update_freq"]
        # read data, multi-index used here
        self.od = pd.read_excel(os.path.join(path, "data/passenger_OD.xlsx"), index_col=[1, 0])
        self.station_set = pd.read_excel(os.path.join(path, "data/stop_news.xlsx"))
        self.routes_set = pd.read_excel(os.path.join(path, "data/route_news.xlsx"))
        # Ensure hourly columns use datetime.time objects so downstream lookups work
        time_cols = self.routes_set.columns[5:]
        rename_map = {}
        for col in time_cols:
            if isinstance(col, str):
                try:
                    rename_map[col] = pd.to_datetime(col).time()
                except ValueError:
                    continue
        if rename_map:
            self.routes_set = self.routes_set.rename(columns=rename_map)
        self.timetable_set = pd.read_excel(os.path.join(path, "data/time_table.xlsx"))
        # Truncate the original timetable by first 50 trips to reduce the calculation pressure
        self.timetable_set = self.timetable_set.sort_values(by=['launch_time', 'direction'])[:self.effective_trip_num].reset_index(drop=True)
        # add index for timetable
        self.timetable_set['launch_turn'] = range(self.timetable_set.shape[0])
        self.max_agent_num = 25

        self.visualizer = visualize(self)
        # Allow disabling automatic plotting when simulation ends
        self.enable_plot = True

        # Set effective station and time period
        self.effective_station_name = sorted(set([self.od.index[i][0] for i in range(self.od.shape[0])]))
        self.effective_period = sorted(list(set([self.od.index[i][1] for i in range(self.od.shape[0])])))

        self.action_space = Box(0, 60, shape=(1,))

        if debug:
            self.summary_data = pd.DataFrame(columns=['bus_id', 'station_id', 'trip_id', 'abs_dis', 'forward_headway',
                                                  'backward_headway', 'headway_diff', 'time'])
            self.summary_reward = pd.DataFrame(columns=['bus_id', 'station_id', 'trip_id', 'forward_headway',
                                                    'backward_headway', 'reward', 'time'])

        self.stations = self.set_stations()
        self.routes = self.set_routes()
        self.timetables = self.set_timetables()

        self._base_state_dim = 8 + len(self.routes)//2  # +1 for headway_dev
        self.state_dim = self._base_state_dim
        self.upper_state_dim = 11

        # FreqDuet: optional causal frequency-separated demand features.
        self.frequency_enabled = False
        self.frequency_upper_enabled = False
        self.frequency_lower_enabled = False
        self.frequency_replace_upper_demand = True
        self.frequency_tracker = None
        self.frequency_logger = None
        self.frequency_logging_enabled = False
        self.frequency_logger_cfg = {}
        self.lower_context_enabled = False
        self.lower_context_dim = 0
        self.lower_context_queue_norm = 50.0
        self.lower_context_features = []

        # TransitDuet: upper policy callback, cost tracking
        self._upper_policy_callback = None  # Set by runner
        self._peak_concurrent = 0

    @property
    def bus_in_terminal(self):
        return [bus for bus in self.bus_all if not bus.on_route]

    # @property
    # def bus_on_route(self):
    #     return [bus for bus in self.bus_all if bus.on_route]

    def set_timetables(self):
        return [Timetable(self.timetable_set['launch_time'][i], self.timetable_set['launch_turn'][i], self.timetable_set['direction'][i]) for i in range(self.timetable_set.shape[0])]

    def set_routes(self):
        return [
            Route(
                self.routes_set['route_id'][i],
                self.routes_set['start_stop'][i],
                self.routes_set['end_stop'][i],
                self.routes_set['distance'][i],
                self.routes_set['V_max'][i],
                self.routes_set.iloc[i, 5:],
                sigma=self.route_sigma
            )
            for i in range(self.routes_set.shape[0])
        ]

    def set_stations(self):
        station_concat = pd.concat([self.station_set, self.station_set[::-1][1:]]).reset_index()
        total_station = []
        for idx, station in station_concat.iterrows():
            # station type is 0 if Terminal else 1
            station_type = 1 if station['stop_name'] not in ['Terminal_up', 'Terminal_down'] else 0

            direction = False if idx >= station_concat.shape[0] / 2 else True
            od = None
            if station['stop_name'] in self.effective_station_name:
                od = self.od.loc[station['stop_name'], station['stop_name']:] if direction else self.od.loc[station['stop_name'], :station['stop_name']]
                # To reduce the OD value in False direction stations in ['X13','X14','X15'] because too many passengers stuck cause the overwhelming
                if station['stop_name'] in ['X13','X14','X15'] and not direction:
                    od *= 0.4

                od.index = od.index.map(str)
                od = od.to_dict(orient='index')

            total_station.append(Station(station_type, station['stop_id'], station['stop_name'], direction, od))

        return total_station

    # return default state and reward
    def reset(self):

        self.current_time = 0

        # initialize station, routes and timetables
        self.stations = self.set_stations()
        self.routes = self.set_routes()
        self.timetables = self.set_timetables()

        # Episode-level demand stochasticity:
        # - Per-hour multiplier: demand intensity varies by hour.
        # - Peak-hour lookup shift: held-out rush-pattern tests can set this
        #   explicitly, otherwise demand-noise runs use the historical default.
        # Stored on env so station_update can use it.
        demand_noise = getattr(self, 'demand_noise', 0.0)
        if demand_noise > 0:
            # Per-hour demand multipliers (14 hours: 6:00-19:00)
            self._demand_multipliers = {
                h: np.clip(np.random.normal(1.0, demand_noise), 0.3, 2.0)
                for h in range(6, 20)
            }
        else:
            self._demand_multipliers = None

        peak_shift_choices = getattr(self, 'peak_shift_choices', None)
        if peak_shift_choices is not None:
            try:
                choices = np.asarray([int(x) for x in peak_shift_choices],
                                     dtype=int)
            except Exception:
                choices = np.asarray([], dtype=int)
            if choices.size:
                probs = None
                peak_shift_probs = getattr(self, 'peak_shift_probs', None)
                if peak_shift_probs is not None:
                    try:
                        probs_arr = np.asarray(
                            [float(x) for x in peak_shift_probs], dtype=float)
                        probs_arr = np.maximum(probs_arr, 0.0)
                        total = float(probs_arr.sum())
                        if probs_arr.size == choices.size and total > 0:
                            probs = probs_arr / total
                    except Exception:
                        probs = None
                self._peak_shift = int(np.random.choice(choices, p=probs))
            else:
                self._peak_shift = 0
        elif demand_noise > 0:
            # Historical generalization default: shift peak demand pattern by
            # one hour sometimes, while usually keeping the original profile.
            self._peak_shift = int(np.random.choice([-1, 0, 0, 0, 1]))
        else:
            self._peak_shift = 0
        self._demand_scale = max(
            0.0, float(getattr(self, 'demand_scale', 1.0)))
        self._od_multipliers = self._sample_od_multipliers()

        if self.frequency_tracker is not None:
            self.frequency_tracker.reset()
        if self.frequency_logger is not None:
            self.frequency_logger.start_episode(
                int(getattr(self, '_freqduet_episode', 0)))

        # initial list of bus on route
        self.bus_id = 0
        self.bus_all = []
        self.route_state = []

        # self.state is combine with route_state, which contains the route.speed_limit of each route, station_state, which
        # contains the station.waiting_passengers of each station and bus_state, which is bus.obs for each bus.
        self.state = {key: [] for key in range(self.max_agent_num)}
        self.reward = {key: 0 for key in range(self.max_agent_num)}
        self.cost = {key: 0.0 for key in range(self.max_agent_num)}
        self.done = False
        self._peak_concurrent = 0
        self._cached_measurement = None
        self._dispatch_rewards = {}
        self._last_dispatch_trip = {}
        # Track last dispatch time per direction for headway enforcement
        self._last_dispatch_time = {True: -9999, False: -9999}  # direction → time

        self.action_dict = {key: None for key in list(range(self.max_agent_num))}

    def _sample_od_multipliers(self):
        """Episode-level OD pair demand multipliers for generalization tests."""
        od_noise = float(getattr(self, 'od_noise', 0.0))
        if od_noise <= 0:
            return None
        clip = getattr(self, 'od_noise_clip', [0.3, 2.0])
        try:
            lo, hi = float(clip[0]), float(clip[1])
        except Exception:
            lo, hi = 0.3, 2.0
        if hi < lo:
            lo, hi = hi, lo
        multipliers = {}
        mean = -0.5 * od_noise * od_noise
        for station in self.stations:
            if station.od is None:
                continue
            for period_od in station.od.values():
                if not isinstance(period_od, dict):
                    continue
                for destination_name, demand in period_od.items():
                    if float(demand) <= 0:
                        continue
                    key = (
                        int(station.station_id),
                        bool(station.direction),
                        str(destination_name),
                    )
                    if key not in multipliers:
                        sample = np.random.lognormal(mean=mean, sigma=od_noise)
                        multipliers[key] = float(np.clip(sample, lo, hi))
        return multipliers

    def initialize_state(self, render=False):
        def count_non_empty_sublist(lst):
            return sum(1 for sublist in lst if sublist)

        while count_non_empty_sublist(list(self.state.values())) == 0:
            self.state, self.reward, self.cost, _ = self.step(self.action_dict, render=render)

        return self.state, self.reward, self.done

    def launch_bus(self, trip):
        # Trip set(self.timetable) contain both direction trips. So we have to make sure the direction and launch time
        # is satisfied before the trip launched.
        # If there is no more appropriate bus in terminal, create a new bus, then add it to all_bus list.
        dispatch_target_headway = float(getattr(trip, 'target_headway', 360.0))
        actual_launch = getattr(trip, '_actual_launch_time', trip.launch_time)
        if len(list(filter(lambda i: i.direction == trip.direction, self.bus_in_terminal))) == 0:
            # cause bus.next_station， current_route and effective station & routes is defined by @property, so no initialize here
            bus = Bus(self.bus_id, trip.launch_turn, actual_launch, trip.direction, self.routes, self.stations)
            self.bus_all.append(bus)
            self.bus_id += 1
        else:
            # if there is bus in terminal and also the direction is satisfied, then we reuse the bus to relaunch one of
            # them, which has the earliest arrived time to terminal.
            bus = sorted(list(filter(lambda i: i.direction == trip.direction, self.bus_in_terminal)), key=lambda bus: bus.back_to_terminal_time)[0]
            bus.reset_bus(trip.launch_turn, actual_launch)
            # in drive() function, we set bus.on_route = False when it finished a trip. Here we set it to True because
            # the iteration in drive(), we just update the state of those bus which on routes
            bus.on_route = True
        bus._freqduet_dispatch_target_headway = dispatch_target_headway

    def step(self, action, debug=False, render=False, episode = 0):
        # Enumerate trips in timetables, if current_time<=launch_time of the trip, then launch it.
        # E.X. timetables = [6:00/launched, 6:05, 6:10], current time is 6:05, then iteration will judge from first trip [6:00]
        # But [6:00] is launched, so next is [6:05]
        for i, trip in enumerate(self.timetables):
            eligible_launch = getattr(
                trip, '_freqduet_scheduled_launch', trip.launch_time)
            if eligible_launch <= self.current_time and not trip.launched:
                if self._upper_policy_callback is not None:
                    # Call upper policy ONCE per trip (when it first becomes eligible)
                    if not hasattr(trip, '_upper_queried') or not trip._upper_queried:
                        s_upper = self._build_upper_state(trip)
                        result = self._upper_policy_callback(s_upper, trip)
                        trip.target_headway = float(result)
                        trip._upper_queried = True
                        self._compute_dispatch_proxy_reward(trip)

                    # v2g mode: if trip has _delta_t, use direct time offset
                    # (no headway enforcement cascade)
                    if getattr(trip, '_freqduet_terminal_dispatch', False):
                        effective_launch = getattr(
                            trip, '_freqduet_scheduled_launch', trip.launch_time)
                        if self.current_time < effective_launch:
                            continue
                    elif hasattr(trip, '_delta_t'):
                        effective_launch = trip._original_launch + trip._delta_t
                        if self.current_time < effective_launch:
                            continue  # not yet time to launch (δ_t delay)
                    else:
                        # v1 mode: headway enforcement
                        actual_gap = self.current_time - self._last_dispatch_time[trip.direction]
                        if actual_gap < trip.target_headway:
                            continue

                # Soft fleet constraint: buffer of 3 over target allows overshoot
                # but hard cap prevents unbounded fleet growth
                n_fleet = getattr(self, '_n_fleet_target', 25)
                buffer = getattr(self, '_fleet_buffer', 3)
                concurrent = sum(1 for bus in self.bus_all if bus.on_route)
                if concurrent >= n_fleet + buffer:
                    continue  # hard cap — only prevents catastrophic overshoot

                # Launch
                trip.launched = True
                trip._actual_launch_time = self.current_time  # v2g: record real launch
                self.launch_bus(trip)
                self._last_dispatch_time[trip.direction] = self.current_time
        # route
        route_state = []
        # update route speed limit by freq
        if self.current_time % self.args['route_state_update_freq'] == 0:
            for route in self.routes:
                route.route_update(self.current_time, self.effective_period)
                route_state.append(route.speed_limit)
            self.route_state = route_state
        # update waiting passengers of every station every second
        # station_state = []
        if self.current_time % self.passenger_update_freq == 0:
            freq_arrivals = {}
            freq_od_arrivals = {}
            collect_od_frequency = (
                self.frequency_enabled
                and self.frequency_tracker is not None
                and getattr(self.frequency_tracker, 'od_features_enabled', False)
            )
            for station in self.stations:
                if collect_od_frequency:
                    new_count, od_counts = station.station_update(
                        self.current_time, self.stations, self.passenger_update_freq,
                        demand_multipliers=self._demand_multipliers,
                        demand_scale=self._demand_scale,
                        od_multipliers=self._od_multipliers,
                        peak_shift=self._peak_shift,
                        return_details=True)
                else:
                    new_count = station.station_update(
                        self.current_time, self.stations, self.passenger_update_freq,
                        demand_multipliers=self._demand_multipliers,
                        demand_scale=self._demand_scale,
                        od_multipliers=self._od_multipliers,
                        peak_shift=self._peak_shift,
                        return_details=False)
                    od_counts = {}
                if self.frequency_enabled and new_count:
                    key = (int(station.station_id), bool(station.direction))
                    freq_arrivals[key] = freq_arrivals.get(key, 0) + int(new_count)
                    if collect_od_frequency:
                        for od_key, od_count in od_counts.items():
                            freq_od_arrivals[od_key] = (
                                freq_od_arrivals.get(od_key, 0) + int(od_count))
            if self.frequency_enabled and self.frequency_tracker is not None:
                prev_freq_updates = int(self.frequency_tracker.total_updates)
                self.frequency_tracker.update(freq_arrivals, freq_od_arrivals)
                bin_applied = int(self.frequency_tracker.total_updates) > prev_freq_updates
                if self.frequency_logger is not None:
                    self.frequency_logger.log_step(
                        self.current_time,
                        freq_arrivals,
                        self.stations,
                        self.bus_all,
                        self.frequency_tracker,
                        bin_applied=bin_applied)
            # station_state.append(len(station.waiting_passengers))
        # update bus state
        for bus in self.bus_all:
            if bus.on_route:
                bus.reward = None
                bus.obs = []
                bus.cost = None
                target_hw = self._get_target_headway_for_bus(bus)
                bus.drive(self.current_time, action[bus.bus_id], self.bus_all,
                          debug=debug, target_headway=target_hw,
                          frequency_tracker=self.frequency_tracker,
                          lower_frequency_enabled=self.frequency_lower_enabled,
                          lower_context_enabled=self.lower_context_enabled,
                          lower_context_queue_norm=self.lower_context_queue_norm,
                          lower_context_features=self.lower_context_features)

        self.state_bus_list = state_bus_list = list(filter(lambda x: len(x.obs) != 0, self.bus_all))
        self.reward_list = reward_list = list(filter(lambda x: x.reward is not None, self.bus_all))

        if len(state_bus_list) != 0:
            # state_bus_list = sorted(state_bus_list, key=lambda x: x.bus_id)
            for i in range(len(state_bus_list)):
                # print('return state is ', state_bus_list[i].obs, ' for bus: ', state_bus_list[i].bus_id, 'at time:', self.current_time)
                # if len(self.state[state_bus_list[i].bus_id]) < 2:
                self.state[state_bus_list[i].bus_id].append(state_bus_list[i].obs)
                # if state_bus_list[i].last_station.station_id not in [0,1,21,22]:
                #     print(1)
                # else:
                #     self.state[state_bus_list[i].bus_id][0] = self.state[state_bus_list[i].bus_id][1]
                #     self.state[state_bus_list[i].bus_id][1] = state_bus_list[i].obs
                # if state_bus_list[i].bus_id == 0:
                #     print(state_bus_list[i].obs[-1], 'bus_id: ', state_bus_list[i].obs[0], ', station_id: ', state_bus_list[i].obs[1], ', trip_id: ', state_bus_list[i].obs[2])
                #     print('return state is ', state_bus_list[i].obs, ' for bus: ', state_bus_list[i].bus_id,
                #           'at time: ', self.current_time)
                # if len(self.state[state_bus_list[i].bus_id]) > 2:
                #     print(1)
                # if debug:
                #     new_data = [state_bus_list[i].obs[0], state_bus_list[i].obs[1], state_bus_list[i].obs[2],
                #                 state_bus_list[i].obs[4]*1000, state_bus_list[i].obs[6] * 60, state_bus_list[i].obs[7]*60,
                #                 state_bus_list[i].obs[6] * 60 - state_bus_list[i].obs[7] * 60, self.current_time]
                #     self.summary_data.loc[len(self.summary_data)] = new_data
        if len(reward_list) != 0:
            # reward_list = sorted(reward_list, key=lambda x: x.bus_id)
            for i in range(len(reward_list)):
                # if reward_list[i].bus_id == 0:
                #     print('return reward is: ', reward_list[i].reward, ' for bus: ', reward_list[i].bus_id, ' at time:', self.current_time)
                # if (reward_list[i].last_station.station_id != 22 and reward_list[i].direction != 0) and \
                #         (reward_list[i].last_station.station_id != 1 and reward_list[i].direction != 1):
                # if len(self.reward[reward_list[i].bus_id]) > 1:
                #     print(2)
                self.reward[reward_list[i].bus_id] = reward_list[i].reward

        # TransitDuet: collect cost
        self.cost_list = [bus for bus in self.bus_all if bus.cost is not None]
        for bus in self.cost_list:
            self.cost[bus.bus_id] = bus.cost

        # Track peak concurrent vehicles for measurement_vector
        concurrent = sum(1 for bus in self.bus_all if bus.on_route)
        self._peak_concurrent = max(self._peak_concurrent, concurrent)


        self.current_time += self.time_step
        if sum([trip.launched for trip in self.timetables]) == len(self.timetables) and sum([bus.on_route for bus in self.bus_all]) == 0:
            self.done = True
            # Cache measurement_vector BEFORE clearing data
            self._cached_measurement = self._compute_measurement_vector()
            if self.frequency_logger is not None:
                self.frequency_logger.flush()
            if not debug:
                for bus in self.bus_all:
                    bus.trajectory.clear()
                    bus.trajectory_dict.clear()
                    del bus.trajectory
                    del bus.trajectory_dict
                for station in self.stations:
                    station.waiting_passengers = np.array([])
                    station.total_passenger.clear()
        else:
            self.done = False

        if self.done and debug:
            self.summary_data = self.summary_data.sort_values(['bus_id', 'time'])

            output_dir = os.path.join(self.path, 'pic')
            os.makedirs(output_dir, exist_ok=True)
            if self.enable_plot:
                self.visualizer.plot(episode)

            self.summary_data.to_csv(os.path.join(output_dir, 'summary_data.csv'))
            self.summary_reward = self.summary_reward.sort_values(['bus_id', 'time'])
            self.summary_reward.to_csv(os.path.join(self.path, 'pic', 'summary_reward.csv'))

        if render and self.current_time % 1 == 0:
            self.visualizer.render()
            time.sleep(0.05)  # Add a delay to slow down the rendering

        return self.state, self.reward, self.cost, self.done

    # ---- TransitDuet helper methods ----

    def configure_frequency_features(self, cfg=None):
        """Enable/disable FreqDuet demand-frequency state augmentation."""
        cfg = dict(cfg or {})
        self.frequency_enabled = bool(cfg.get('enable', False))
        self.frequency_upper_enabled = bool(cfg.get('upper_features', True))
        self.frequency_lower_enabled = bool(cfg.get('lower_features', True))
        self.frequency_replace_upper_demand = bool(
            cfg.get('replace_upper_demand_with_low', True))
        self.frequency_logger_cfg = dict(cfg.get('logging', {}) or {})
        self.frequency_logging_enabled = (
            self.frequency_enabled
            and bool(self.frequency_logger_cfg.get('enable', False)))
        self.frequency_logger = None
        lower_context_cfg = cfg.get('lower_context', {}) or {}
        self.lower_context_enabled = bool(lower_context_cfg.get('enable', False))
        default_context = [
            'load', 'capacity', 'queue', 'speed_residual',
            'shock_age', 'schedule_slack']
        requested_context = lower_context_cfg.get('features', default_context)
        allowed_context = set(default_context)
        self.lower_context_features = [
            str(x) for x in requested_context if str(x) in allowed_context
        ] if self.lower_context_enabled else []
        self.lower_context_dim = len(self.lower_context_features)
        self.lower_context_queue_norm = max(
            float(lower_context_cfg.get('queue_norm', 50.0)), 1e-6)

        if self.frequency_enabled:
            method = str(cfg.get('method', '')).lower()
            if method in {'harmonic', 'dynamic_harmonic', 'harmonic_rls',
                          'dynamic_harmonic_nb'}:
                if cfg.get('use_historical_prior', True) and not isinstance(
                        cfg.get('harmonic_prior'), dict):
                    cfg['harmonic_prior'] = self._build_harmonic_prior(cfg)
            self.frequency_tracker = DemandFrequencyTracker.from_config(
                cfg, update_interval_s=self.passenger_update_freq)
            self.state_dim = self._base_state_dim + (
                self.frequency_tracker.lower_feature_dim
                if self.frequency_lower_enabled else 0) + self.lower_context_dim
            upper_extra = 0
            if self.frequency_upper_enabled:
                upper_extra = (
                    self.frequency_tracker.upper_feature_dim - 1
                    if self.frequency_replace_upper_demand
                    else self.frequency_tracker.upper_feature_dim)
            self.upper_state_dim = 11 + upper_extra
        else:
            self.frequency_tracker = None
            self.state_dim = self._base_state_dim + self.lower_context_dim
            self.upper_state_dim = 11
        log_output_dir = self.frequency_logger_cfg.get('output_dir', None)
        if log_output_dir:
            self.configure_frequency_logging(log_output_dir)

    def configure_frequency_logging(self, output_dir=None):
        """Create the optional passive demand/frequency trace logger."""
        self.frequency_logger = None
        if not self.frequency_logging_enabled:
            return
        if output_dir is None:
            output_dir = self.frequency_logger_cfg.get('output_dir', None)
        if output_dir is None:
            output_dir = os.path.join(self.path, 'pic')
        self.frequency_logger = DemandEventLogger(
            output_dir=output_dir,
            aggregate_filename=self.frequency_logger_cfg.get(
                'aggregate_filename', 'demand_trace.csv'),
            station_filename=self.frequency_logger_cfg.get(
                'station_filename', 'demand_station_trace.csv'),
            station_rows=self.frequency_logger_cfg.get('station_rows', True),
            bin_only=self.frequency_logger_cfg.get('bin_only', True),
            include_empty_stations=self.frequency_logger_cfg.get(
                'include_empty_stations', False))

    def _build_harmonic_prior(self, cfg):
        """Fit harmonic priors from the historical OD table used by the env.

        The simulator's passenger generator already treats the OD spreadsheet as
        the historical hourly intensity. This prior gives the harmonic online
        estimator a service-day baseline; RLS then adapts to per-episode demand
        noise and stochastic arrivals.
        """
        bin_interval_s = float(cfg.get('bin_sec', self.passenger_update_freq))
        period_s = float(cfg.get('harmonic_period_s', 14 * 3600.0))
        fourier_k = int(cfg.get('fourier_K', cfg.get('fourier_k', 4)))
        ridge = float(cfg.get('harmonic_ridge', 1e-2))
        n_bins = max(1, int(round(period_s / max(bin_interval_s, 1e-6))))

        global_rates = np.zeros(n_bins, dtype=np.float64)
        local_rates = {}
        od_rates = {}
        station_lookup = {
            (s.station_name, bool(s.direction)): int(s.station_id)
            for s in self.stations
        }

        for station in self.stations:
            if station.od is None:
                continue
            local_key = (int(station.station_id), bool(station.direction))
            local_arr = local_rates.setdefault(
                local_key, np.zeros(n_bins, dtype=np.float64))
            for i in range(n_bins):
                hour = 6 + int((i * bin_interval_s) // 3600)
                hour = max(6, min(19, hour))
                period_key = f"{hour:02}:00:00"
                period_od = station.od.get(period_key, {})
                if not isinstance(period_od, dict):
                    continue
                total_hourly = 0.0
                for dest_name, demand_hourly in period_od.items():
                    demand_hourly = float(demand_hourly)
                    if demand_hourly <= 0:
                        continue
                    total_hourly += demand_hourly
                    dest_id = station_lookup.get(
                        (str(dest_name), bool(station.direction)))
                    if dest_id is None:
                        continue
                    od_key = (
                        int(station.station_id), int(dest_id),
                        bool(station.direction))
                    od_arr = od_rates.setdefault(
                        od_key, np.zeros(n_bins, dtype=np.float64))
                    od_arr[i] += demand_hourly / 60.0
                local_arr[i] += total_hourly / 60.0
                global_rates[i] += total_hourly / 60.0

        return {
            'global': fit_harmonic_prior(
                global_rates, bin_interval_s, period_s, fourier_k, ridge),
            'local': {
                k: fit_harmonic_prior(v, bin_interval_s, period_s, fourier_k, ridge)
                for k, v in local_rates.items()
            },
            'od': {
                k: fit_harmonic_prior(v, bin_interval_s, period_s, fourier_k, ridge)
                for k, v in od_rates.items()
            },
        }

    def _frequency_adjusted_upper_demand(self, fallback_demand_norm):
        """Return (demand_norm, extra_features) for the upper state."""
        if not (self.frequency_enabled and self.frequency_upper_enabled
                and self.frequency_tracker is not None):
            return fallback_demand_norm, []
        feats = self.frequency_tracker.upper_features()
        if self.frequency_replace_upper_demand:
            return float(feats[0]), [float(x) for x in feats[1:]]
        return fallback_demand_norm, [float(x) for x in feats]

    def frequency_summary(self):
        if self.frequency_tracker is None:
            return {
                'freq_low_demand': 0.0,
                'freq_low_slope': 0.0,
                'freq_low_forecast': 0.0,
                'freq_high_energy': 0.0,
                'freq_middle': 0.0,
                'freq_middle_energy': 0.0,
                'freq_od_entropy': 0.0,
                'freq_od_high_energy': 0.0,
                'freq_od_active': 0,
                'freq_updates': 0,
                'freq_promotion_flag': 0.0,
                'freq_promotion_strength': 0.0,
                'freq_promotion_age': 0.0,
                'freq_promotion_score': 0.0,
                'freq_promotion_absorptions': 0,
                'freq_promotion_absorbed': 0.0,
            }
        return self.frequency_tracker.summary()

    def _compute_dispatch_proxy_reward(self, current_trip):
        """
        Compute per-dispatch proxy reward for the PREVIOUS same-direction dispatch.
        Measures what happened between the last dispatch and this one:
          - headway regularity of on-route buses (negative CV)
          - fleet utilization penalty (over N_fleet)
          - passenger wait proxy (waiting passengers at stations)

        Stored in self._dispatch_rewards[trip_id] for runner to collect.
        """
        if not hasattr(self, '_dispatch_rewards'):
            self._dispatch_rewards = {}
        if not hasattr(self, '_last_dispatch_trip'):
            self._last_dispatch_trip = {}

        direction = current_trip.direction
        prev_trip_id = self._last_dispatch_trip.get(direction, None)
        self._last_dispatch_trip[direction] = current_trip.launch_turn

        if prev_trip_id is None:
            return  # first dispatch in this direction, no reward yet

        # Headway regularity: negative CV of active buses' forward headways
        active_buses = [b for b in self.bus_all
                        if b.on_route and b.direction == direction
                        and b.forward_headway > 0]
        if len(active_buses) >= 2:
            hws = np.array([b.forward_headway for b in active_buses])
            hw_cv = hws.std() / max(hws.mean(), 1.0)
            r_regularity = -min(hw_cv, 2.0)  # in [-2, 0]
        else:
            r_regularity = 0.0

        # Fleet penalty: penalize exceeding N_fleet
        concurrent = sum(1 for b in self.bus_all if b.on_route)
        n_fleet = getattr(self, '_n_fleet_target', 12)
        r_fleet = -max(0.0, concurrent - n_fleet) / n_fleet  # in [-1, 0] roughly

        # Waiting passengers proxy (normalized)
        total_waiting = sum(len(s.waiting_passengers) for s in self.stations)
        r_wait = -min(total_waiting / 500.0, 2.0)  # in [-2, 0]

        # Combined proxy reward (normalized to roughly [-1, 0])
        proxy_reward = 0.5 * r_regularity + 0.3 * r_fleet + 0.2 * r_wait

        self._dispatch_rewards[prev_trip_id] = float(proxy_reward)

    def _get_target_headway_for_bus(self, bus):
        """Look up the target_headway from the timetable that launched this bus."""
        if hasattr(bus, '_freqduet_dispatch_target_headway'):
            return bus._freqduet_dispatch_target_headway
        for tt in self.timetables:
            if tt.launch_turn == bus.trip_id:
                return tt.target_headway
        return 360.0

    def _build_upper_state(self, trip):
        """
        Build upper policy state vector at each dispatch event.
        Returns: np.array of shape (5,)
            [hour_norm, demand_norm, fleet_norm, prev_headway_norm, unhealthy_rate]
        """
        hour = 6 + self.current_time // 3600

        # Aggregate current-period passenger demand across all stations
        effective_period_str = f"{min(int(hour), 19):02}:00:00"
        total_demand = 0.0
        for s in self.stations:
            if s.od is not None:
                period_data = s.od.get(effective_period_str, {})
                if isinstance(period_data, dict):
                    total_demand += sum(period_data.values())

        fleet_on_route = sum(1 for bus in self.bus_all if bus.on_route)

        # Actual headway from previous same-direction trip
        same_dir_launched = [tt for tt in self.timetables
                             if tt.launched and tt.direction == trip.direction]
        if len(same_dir_launched) >= 2:
            last_two = sorted(same_dir_launched, key=lambda t: t.launch_time)[-2:]
            prev_actual_headway = last_two[1].launch_time - last_two[0].launch_time
        else:
            prev_actual_headway = 360.0

        # Fraction of on-route buses currently holding (signal of fleet pressure)
        from env.bus import BusState
        holding_count = sum(1 for bus in self.bus_all
                            if bus.on_route and bus.state == BusState.HOLDING)
        holding_ratio = holding_count / max(fleet_on_route, 1)

        return np.array([
            hour / 24.0,
            total_demand / 1000.0,
            fleet_on_route / self.max_agent_num,
            prev_actual_headway / 600.0,
            holding_ratio,
        ], dtype=np.float32)

    def _compute_measurement_vector(self):
        """
        Compute measurement z(π) from live data. Must be called BEFORE cleanup.
        Returns: np.array [avg_wait_min, peak_fleet, headway_cv]
        """
        # z[0]: average passenger wait time (minutes)
        total_wait, pax_count = 0.0, 0
        for s in self.stations:
            for p in s.total_passenger:
                if hasattr(p, 'boarding_time') and p.boarding_time is not None:
                    total_wait += (p.boarding_time - p.appear_time)
                    pax_count += 1
        avg_wait = (total_wait / max(pax_count, 1)) / 60.0

        # z[1]: peak concurrent fleet size
        peak_fleet = self._peak_concurrent

        # z[2]: headway coefficient of variation (replaces bunching_rate)
        headways = [bus.forward_headway for bus in self.bus_all
                    if hasattr(bus, 'forward_headway') and bus.forward_headway > 0]
        if len(headways) >= 2:
            hw_arr = np.array(headways)
            headway_cv = float(hw_arr.std() / max(hw_arr.mean(), 1.0))
        else:
            headway_cv = 0.0

        return np.array([avg_wait, peak_fleet, headway_cv])

    @property
    def measurement_vector(self):
        """
        ApproPO long-term measurement z(π), computed at episode end.
        Returns: np.array [avg_wait_min, peak_fleet, headway_cv]
        """
        if hasattr(self, '_cached_measurement') and self._cached_measurement is not None:
            return self._cached_measurement
        return self._compute_measurement_vector()

    # ---- v2: per-trip holding feedback support ----

    def get_completed_trip_holdings(self):
        """
        Collect applied holding actions for all buses that have completed trips
        (off-route) since last call.

        Returns: dict {trip_id: [action_1, action_2, ...]}
        """
        result = {}
        for bus in self.bus_all:
            if not bus.on_route and hasattr(bus, 'applied_actions') and bus.applied_actions:
                # Bus completed its trip; collect actions keyed by trip_id
                result[bus.trip_id] = list(bus.applied_actions)
        return result

    def get_direction_holding_stats(self, direction, n_recent=5):
        """
        Get holding statistics from recent completed trips in a given direction.

        Returns: dict with rolling_mean, rolling_std (of per-trip mean holdings)
        """
        completed = []
        for bus in self.bus_all:
            if not bus.on_route and bus.direction == direction:
                if hasattr(bus, 'applied_actions') and bus.applied_actions:
                    completed.append({
                        'trip_id': bus.trip_id,
                        'mean': float(np.mean(bus.applied_actions)),
                        'n_stops': len(bus.applied_actions),
                    })
        # Sort by trip_id (proxy for time order), take last n_recent
        completed.sort(key=lambda x: x['trip_id'])
        recent = completed[-n_recent:] if len(completed) > n_recent else completed

        if not recent:
            return {'rolling_mean': 0.0, 'rolling_std': 0.0, 'n_trips': 0}
        means = [c['mean'] for c in recent]
        return {
            'rolling_mean': float(np.mean(means)),
            'rolling_std': float(np.std(means)) if len(means) > 1 else 0.0,
            'n_trips': len(recent),
        }

    def _build_upper_state_v2(self, trip):
        """
        Build enriched upper state for v2 coupling.

        Returns: np.array of shape (upper_state_dim_v2,)
            [hour_norm, demand_norm, fleet_norm, gap_to_prev_norm,
             holding_ratio, holding_mean_same_dir, holding_std_same_dir,
             holding_mean_other_dir, scheduled_headway_norm, direction]
        """
        hour = 6 + self.current_time // 3600

        # Demand
        effective_period_str = f"{min(int(hour), 19):02}:00:00"
        total_demand = 0.0
        for s in self.stations:
            if s.od is not None:
                period_data = s.od.get(effective_period_str, {})
                if isinstance(period_data, dict):
                    total_demand += sum(period_data.values())

        fleet_on_route = sum(1 for bus in self.bus_all if bus.on_route)

        # Actual headway gap to previous dispatch in same direction
        actual_gap = self.current_time - self._last_dispatch_time.get(trip.direction, -9999)
        if actual_gap > 9000:
            actual_gap = 360.0  # first trip

        # Holding ratio (fraction of active buses currently holding)
        from env.bus import BusState
        holding_count = sum(1 for bus in self.bus_all
                            if bus.on_route and bus.state == BusState.HOLDING)
        holding_ratio = holding_count / max(fleet_on_route, 1)

        # v2 KEY: holding statistics from recent trips (lower → upper feedback)
        same_dir_stats = self.get_direction_holding_stats(trip.direction, n_recent=5)
        other_dir_stats = self.get_direction_holding_stats(not trip.direction, n_recent=5)

        # Scheduled headway from original timetable
        scheduled_hw = trip.target_headway if hasattr(trip, 'target_headway') else 360.0

        # v2k: include current fleet budget in state (for Pareto-aware policy)
        n_fleet_norm = getattr(self, '_n_fleet_target', 12) / 20.0

        demand_norm = total_demand / 1000.0
        demand_norm, freq_extra = self._frequency_adjusted_upper_demand(demand_norm)

        state = [
            hour / 24.0,                                   # [0] time of day
            demand_norm,                                   # [1] raw or low-frequency demand
            fleet_on_route / self.max_agent_num,           # [2] fleet utilization
            actual_gap / 600.0,                            # [3] gap to prev dispatch
            holding_ratio,                                 # [4] fraction of fleet holding
            same_dir_stats['rolling_mean'] / 60.0,         # [5] mean holding, same dir
            same_dir_stats['rolling_std'] / 60.0,          # [6] std holding, same dir
            other_dir_stats['rolling_mean'] / 60.0,        # [7] mean holding, other dir
            scheduled_hw / 600.0,                          # [8] base scheduled headway
            float(trip.direction),                         # [9] direction (0 or 1)
            n_fleet_norm,                                  # [10] v2k: fleet budget
        ]
        state.extend(freq_extra)
        return np.array(state, dtype=np.float32)

    def set_timetable_from_planner(self, headway_params):
        """
        Upper layer writes target_headway into timetable.

        Args:
            headway_params: dict {'peak': float, 'off_peak': float, 'transition': float}
                            OR list[float] of per-trip target headways
        """
        if isinstance(headway_params, dict):
            for tt in self.timetables:
                hour = 6 + tt.launch_time // 3600
                if 7 <= hour <= 9 or 17 <= hour <= 19:
                    tt.target_headway = headway_params['peak']
                elif 9 < hour < 17:
                    tt.target_headway = headway_params['off_peak']
                else:
                    tt.target_headway = headway_params['transition']
        else:
            for tt, hw in zip(self.timetables, headway_params):
                tt.target_headway = float(hw)


if __name__ == '__main__':
    debug = True
    render = False
    num_runs = 1

    env_dir = Path(__file__).resolve().parent
    env = env_bus(str(env_dir), debug=debug)
    env.enable_plot = True
    actions = {key: 0. for key in list(range(env.max_agent_num))}

    all_events = []
    cumulative_time = 0

    for run_idx in range(1, num_runs + 1):
        env.reset()
        while not env.done:
            state, reward, cost, done = env.step(action=actions, debug=debug,
                                           render=render, episode=run_idx)

        events = env.visualizer.extract_bunching_events()
        cumulative_time += env.current_time
        all_events.extend(events)

    # Only quit pygame if it was initialized
    if pygame.get_init():
        pygame.quit()

    if all_events:
        df = pd.DataFrame(all_events).sort_values(['time'])
        output_dir = os.path.join(env.path, 'pic')
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, f'all_bunching_records_{num_runs}.csv'), index=False)
        # env.visualizer.plot_bunching_events(all_events, exp=str(num_runs))

    print('Total simulation time:', cumulative_time)
