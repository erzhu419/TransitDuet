from enum import Enum, auto
import numbers
import numpy as np


def _remove_indices_preserve_order(values, removed_indices):
    """Return an array without selected positions while retaining FIFO order."""
    array = np.asarray(values)
    if array.size == 0 or not removed_indices:
        return array.copy()
    keep = np.ones(len(array), dtype=bool)
    indices = np.asarray(removed_indices, dtype=np.int64)
    indices = indices[(indices >= 0) & (indices < len(array))]
    keep[indices] = False
    return array[keep]


class BusState(Enum):
    HOLDING = auto()
    WAITING_ACTION = auto()
    DWELLING = auto()
    TRAVEL = auto()


class Bus(object):
    def __init__(self, bus_id, trip_id, launch_time, direction, routes, stations):
        self.bus_id = bus_id
        self.trip_id = trip_id
        self.trip_id_list = [trip_id]
        self.launch_time = launch_time
        self.direction = direction

        self.routes_list = routes
        self.stations_list = stations
        self.in_station = True
        self.passengers = np.array([]) # list of passengers on bus
        self.capacity = 50 # upper bound of passengers on bus
        self.current_speed = 0. # current speed of bus

        self.trip_turn = len(self.trip_id_list)
        station_split = (len(self.stations_list) + 1) // 2
        self.effective_station = self.stations_list[:station_split] if self.direction else self.stations_list[station_split - 1:] # 从所有站点中抽取有效站点
        self.last_station = self.effective_station[0] # 初始化首站
        self.next_station = self.effective_station[1] # 初始化次站
        self.last_station_dis = 0. # 上一站到当前站的距离
        self.route_index = {(route.start_stop, route.end_stop): route for route in routes} # GPT优化方案1 构建索引字典
        self.next_station_dis = self.current_route.distance # 当前站到下一站的距离
        self.absolute_distance = 0. if self.direction else len(self.stations_list) // 2 * 500 # 在上行时，绝对距离从0开始，下行时从11500开始
        self.trajectory = [] # 轨迹记录
        self.trajectory_dict = {} # 轨迹字典
        for station in self.effective_station:
            self.trajectory_dict[station.station_name] = []

        self.obs = [] # 状态值
        self.forward_bus = None # 前车对象
        self.backward_bus = None  # 后车对象
        self.forward_headway = 360. # 前车车头时距
        self.backward_headway = 360. # 后车车头时距
        self.reward = None # 奖励值
        self.cost = None  # Lagrangian cost (headway deviation²)
        self._target_headway = 360.0  # set by sim via drive()
        self._frequency_tracker = None
        self._headway_recorder = None
        self.forward_headway_source = "target_default"
        self.forward_predecessor_trip_id = None
        self.pre_action_forward_headway = None
        self.pre_action_forward_headway_source = "unavailable"
        self._lower_state_input_schema = "legacy_headway_deviation"
        self._lower_observation_contract = "latent_oracle_legacy"
        self._headway_reward_mode = "symmetric_legacy"
        self._lower_frequency_enabled = False
        self._lower_context_enabled = False
        self._lower_context_queue_norm = 50.0
        self._lower_context_features = []
        self._lower_context_gate_value = 1.0

        self.alight_num = 0. # 下车人数
        self.board_num = 0. # 上车人数
        self.last_board_wait_sum_s = 0.0
        self.last_board_lf_wait_sum_s = 0.0
        self.last_board_hf_wait_sum_s = 0.0
        self.last_board_lf_mass = 0.0
        self.last_board_hf_mass = 0.0
        self.last_board_count = 0
        self.last_board_station_id = int(self.next_station.station_id)
        self.last_board_time = None
        self.last_service_dwell_s = 0.0
        self.last_left_behind_count = 0
        self.last_action_s = 0.0
        self.last_action_time = None
        self.last_action_station_id = int(self.last_station.station_id)
        self.back_to_terminal_time = None
        self.last_completed_trip_id = None
        self.last_completed_direction = None
        self.last_completed_reward = None
        self.last_completed_cost = None
        self.last_completed_station_id = None
        self.last_completed_target_headway = None
        self.last_completed_board_wait_sum_s = 0.0
        self.last_completed_board_lf_wait_sum_s = 0.0
        self.last_completed_board_hf_wait_sum_s = 0.0
        self.last_completed_board_lf_mass = 0.0
        self.last_completed_board_hf_mass = 0.0
        self.last_completed_board_count = 0

        self.acceleration = 3 # 加速度
        self.deceleration = 5 # 刹车加速度

        self.state = BusState.HOLDING  # 初始状态：在站内上下客
        self.on_route = True # 是否在路上，如果在路上，为True，否则为False，用于判断是否到达终点站

        self.holding_time = 0. # 停站时间，用于上下乘客
        self.dwelling_time = 0. # 驻站时间，用于执行动作，停车等待

        self.headway_dif = []
        self.applied_actions = []  # v2: track holding actions per trip for feedback
        self.applied_action_loads = []
        self.episode_hold_vehicle_seconds = 0.0
        self.episode_hold_person_seconds = 0.0
        self.episode_commanded_hold_vehicle_seconds = 0.0
        self.episode_commanded_hold_person_seconds = 0.0
        self._active_holding_load = 0
        self.holding_action_trace_mode = "positive_only"
        self.unobserved_action_mode = "legacy_stale"

        # record of stop intervals [station_name, start_time, end_time]
        self.stop_records = []
        self._stop_start_time = None
        self._stop_station = None

    @property
    def occupancy(self):
        return str(len(self.passengers)) + '/' + str(self.capacity)

    # decide if the negative or positive of step_length, when direction == 1, step_length > 0, vise versa
    @property
    def direction_int(self):
        return 1 if self.direction else -1

    # effective_route is effective routes for every bus, same as effective_station
    @property
    def effective_route(self):
        return self.routes_list[:round(len(self.routes_list) / 2)] if self.direction else self.routes_list[round(len(self.routes_list) / 2):]

    # searching for next_station when last_station changed
    @property
    def travel_distance(self):
        return self.absolute_distance if self.direction else sum([route.distance for route in self.effective_route]) - self.absolute_distance

    def next_station_func(self):
        return self.effective_station[self.last_station.station_id + self.direction_int] if self.direction else self.effective_station[-(self.last_station.station_id + self.direction_int + 1)]

    @property
    def station_after_the_next(self):
        # return the station after the next station
        return self.effective_station[self.last_station.station_id + 2 * self.direction_int] if self.direction else self.effective_station[-(self.last_station.station_id + 2 * self.direction_int + 1)]

    @property
    def station_before_the_last(self):
        # return the station before the last station
        return self.effective_station[self.last_station.station_id - 2 * self.direction_int] if self.direction else self.effective_station[-(self.last_station.station_id - 2 * self.direction_int + 1)]
    # searching for current_route when last_station and next_station changed
    # @property
    # def current_route(self):
    #     return list(filter(lambda i: i.start_stop == self.last_station.station_name and i.end_stop == self.next_station.station_name, self.effective_route))[0]

    # GPT优化方案1 构建索引字典
    @property
    def current_route(self):
        # 从字典中查找对应路段
        key = (self.last_station.station_name, self.next_station.station_name)
        return self.route_index[key]

    # When bus is arrived in a station, passengers have to alight and boarding.
    def exchange_passengers(self, current_time, debug):
        # Because we cannot mutate the list inter iteration. Record the index of every passenger we want to remove from
        # original passengers list then remove them with the pre-record index
        index_of_passenger_on_bus = []
        index_of_passenger_in_station = []
        self.last_board_wait_sum_s = 0.0
        self.last_board_lf_wait_sum_s = 0.0
        self.last_board_hf_wait_sum_s = 0.0
        self.last_board_lf_mass = 0.0
        self.last_board_hf_mass = 0.0
        self.last_board_count = 0
        self.last_board_station_id = int(self.next_station.station_id)
        self.last_board_time = current_time
        # passengers alight from bus(self)
        for i, passenger in enumerate(self.passengers):
            if passenger.destination_station.station_name == self.next_station.station_name:
                passenger.arrived = True
                passenger.arrive_time = current_time
                self.alight_num += 1
                index_of_passenger_on_bus.append(i)
        # remove passengers from bus
        self.passengers = _remove_indices_preserve_order(
            self.passengers, index_of_passenger_on_bus)
        # passengers boarding from station(self.next_station)
        for i, passenger in enumerate(self.next_station.waiting_passengers):
            if len(self.passengers) < self.capacity:
                passenger.boarded = True
                passenger.boarding_time = current_time
                passenger.travel_bus = self
                wait_s = max(0.0, float(current_time - passenger.appear_time))
                low_share = float(getattr(
                    passenger, 'frequency_low_share', 1.0))
                high_share = float(getattr(
                    passenger, 'frequency_high_share', 0.0))
                if abs(low_share + high_share - 1.0) > 1e-9:
                    raise AssertionError(
                        'passenger LF/HF shares no longer conserve unit mass')
                self.last_board_wait_sum_s += wait_s
                self.last_board_lf_wait_sum_s += low_share * wait_s
                self.last_board_hf_wait_sum_s += high_share * wait_s
                self.last_board_lf_mass += low_share
                self.last_board_hf_mass += high_share
                self.last_board_count += 1
                self.passengers = np.append(self.passengers, passenger)
                self.board_num += 1
                index_of_passenger_in_station.append(i)

        self.next_station.waiting_passengers = _remove_indices_preserve_order(
            self.next_station.waiting_passengers,
            index_of_passenger_in_station,
        )

        self.holding_time = max(self.alight_num, (self.board_num * 2.)) + 4.
        self.last_service_dwell_s = float(self.holding_time)
        self.last_left_behind_count = int(
            len(self.next_station.waiting_passengers))
        # print('Bus id: ',self.bus_id, ', stop id: ', self.last_station.station_id," ,holding time: ", self.holding_time)
        # if self.bus_id == 2 and debug:
        #     print('Bus: ', self.bus_id, ' at station: ', self.next_station.station_id ,' ,current time: ', current_time,' ,holding time: ', self.holding_time)
        self.alight_num = 0.
        self.board_num = 0.

    def bus_update(self):
        # update the bus state
        self.last_station = self.next_station
        self.next_station = self.next_station_func()
        self.last_station_dis = 0
        self.next_station_dis = self.current_route.distance

    def drive(self, current_time, action, bus_all, debug, target_headway=360.0,
              frequency_tracker=None, lower_frequency_enabled=False,
              lower_context_enabled=False, lower_context_queue_norm=50.0,
              lower_context_features=None, lower_context_gate_value=1.0,
              headway_recorder=None,
              lower_state_input_schema="legacy_headway_deviation",
              lower_observation_contract="latent_oracle_legacy",
              headway_reward_mode="symmetric_legacy"):
        self._target_headway = target_headway
        self._frequency_tracker = frequency_tracker
        self._headway_recorder = headway_recorder
        self._lower_state_input_schema = str(lower_state_input_schema)
        self._lower_observation_contract = str(lower_observation_contract)
        self._headway_reward_mode = str(headway_reward_mode)
        self._lower_frequency_enabled = lower_frequency_enabled
        self._lower_context_enabled = lower_context_enabled
        self._lower_context_queue_norm = max(float(lower_context_queue_norm), 1e-6)
        self._lower_context_features = list(lower_context_features or [])
        self._lower_context_gate_value = float(np.clip(
            lower_context_gate_value, 0.0, 1.0))
        # absolute_distance & last_station_dis is divided by 1000 as kilometers rather than meters. forward_headway & backward_headway
        # is divided by 60 minutes rather than seconds. passengers on bus, boarding passengers and alighting passengers are divided by self.capacity
        # step_length = 0, which means how long a bus moves in a time step, calculated by speeding up and original velocity.

        if self.state == BusState.TRAVEL:
            if self.next_station_dis <= self.current_speed:
                self.exchange_passengers(current_time, debug)  # self.holding_time is set in this function

                # self.trajectory.append([self.next_station.station_name, current_time, self.absolute_distance, self.direction, self.trip_id])
                # self.trajectory_dict[self.next_station.station_name].append([
                #     self.next_station.station_name,
                #     current_time + self.holding_time + 0.01,
                #     self.absolute_distance,
                #     self.direction,
                #     self.trip_id
                # ])

                self.arrive_station(current_time, bus_all, debug)
                self.state = BusState.HOLDING
                self.in_station = True
            else:
                self._advance_on_route()
        elif self.state == BusState.HOLDING:
            self._process_holding(current_time, bus_all, debug)
        elif self.state == BusState.WAITING_ACTION:
            self._start_dwelling(action, current_time)
        elif self.state == BusState.DWELLING:
            self._process_dwelling(current_time)
        else:
            # Recover gracefully if state was not initialised as expected
            self.state = BusState.TRAVEL
            self._advance_on_route()

    def _advance_on_route(self):
        if self.current_route.speed_limit >= self.current_speed:
            if self.current_route.speed_limit - self.current_speed > self.acceleration:
                step_length = (self.current_speed + self.acceleration / 2) * self.direction_int
                self.current_speed += self.acceleration
            else:
                step_length = (self.current_speed + self.current_route.speed_limit) * 0.5 * self.direction_int
                self.current_speed = self.current_route.speed_limit
        else:
            if self.current_speed - self.current_route.speed_limit > self.deceleration:
                step_length = (self.current_speed - self.deceleration / 2) * self.direction_int
                self.current_speed -= self.deceleration
            else:
                step_length = (self.current_speed + self.current_route.speed_limit) * 0.5 * self.direction_int
                self.current_speed = self.current_route.speed_limit

        self.last_station_dis += abs(step_length)
        self.next_station_dis -= abs(step_length)
        self.absolute_distance += step_length

    def _process_holding(self, current_time, bus_all, debug):
        if self.holding_time <= 1:
            self.holding_time = 0
            self._prepare_for_action(current_time, bus_all, debug)
        else:
            self.holding_time -= 1

    def _prepare_for_action(self, current_time, bus_all, debug):
        self.forward_bus = list(filter(lambda x: self.trip_id - 2 in x.trip_id_list, bus_all))
        self.backward_bus = list(filter(lambda x: self.trip_id + 2 in x.trip_id_list, bus_all))
        self._update_pre_action_forward_headway(current_time)

        controlled_stop = self.next_station in self.effective_station[2:]
        exact_forward_valid = (
            getattr(self, 'forward_headway_source', 'target_default')
            == 'arrival_event')
        if getattr(
                self, '_lower_state_input_schema',
                'legacy_headway_deviation') == 'causal_forward_v4':
            action_requested = controlled_stop and exact_forward_valid
            if controlled_stop and not exact_forward_valid:
                self._invalid_headway_mask_time = float(current_time)
        else:
            action_requested = (
                controlled_stop
                and (len(self.forward_bus) != 0 or len(self.backward_bus) != 0)
            )
        if action_requested:
            target_hw = self._target_headway
            headway_dev = (self.forward_headway - target_hw) / max(target_hw, 1.0)
            target_or_deviation = (
                float(target_hw)
                if self._lower_state_input_schema in {
                    "explicit_target_v2", "causal_forward_v4"}
                else float(headway_dev)
            )

            if self._lower_state_input_schema == 'causal_forward_v4':
                follower_or_validity = float(exact_forward_valid)
                service_or_downstream = float(self.last_service_dwell_s)
            else:
                follower_or_validity = float(self.backward_headway)
                service_or_downstream = (
                    len(self.next_station.waiting_passengers) * 1.5
                    + self.current_route.distance / self.current_route.speed_limit
                )

            self.obs = [
                self.bus_id,
                self.last_station.station_id,
                current_time // 3600,
                self.direction,
                self.forward_headway,
                follower_or_validity,
                service_or_downstream,
                target_or_deviation,
            ]
            all_route = self.routes_list[:len(self.routes_list) // 2] if self.direction else self.routes_list[len(self.routes_list) // 2:]
            speed_list = [all_route[i].speed_limit for i in range(len(all_route))]
            self.obs.extend(speed_list)

            if self._lower_context_enabled:
                target_hw_safe = max(float(target_hw), 1.0)
                load_norm = len(self.passengers) / max(float(self.capacity), 1.0)
                cap_remain_norm = max(0.0, 1.0 - load_norm)
                if self._lower_observation_contract == 'deployable_apc_avl_v4':
                    queue_norm = (
                        self.last_left_behind_count
                        / self._lower_context_queue_norm)
                else:
                    queue_norm = (
                        len(self.next_station.waiting_passengers)
                        / self._lower_context_queue_norm)
                route_speed_mean = max(float(np.mean(speed_list)), 1e-6)
                speed_residual = (
                    float(self.current_route.speed_limit) - route_speed_mean
                ) / route_speed_mean
                shock_age = 0.0
                if (self._frequency_tracker is not None
                        and hasattr(self._frequency_tracker,
                                    "local_promotion_summary")):
                    station_id = getattr(
                        self.last_station, 'station_id', self.next_station.station_id)
                    shock_age = float(
                        self._frequency_tracker.local_promotion_summary(
                            station_id, self.direction).get("age", 0.0))
                schedule_slack = (target_hw - self.forward_headway) / target_hw_safe
                fwd_norm = self.forward_headway / target_hw_safe
                bwd_norm = self.backward_headway / target_hw_safe
                headway_balance = (
                    self.backward_headway - self.forward_headway
                ) / target_hw_safe
                # First-order proxy for whether extra holding improves local
                # headway regularity: holding increases forward headway and
                # decreases backward headway.
                hold_value_proxy = (
                    abs(self.forward_headway - target_hw)
                    + abs(self.backward_headway - target_hw)
                    - abs((self.forward_headway + 15.0) - target_hw)
                    - abs((self.backward_headway - 15.0) - target_hw)
                ) / target_hw_safe
                route_len = max(
                    float(sum(route.distance for route in self.effective_route)),
                    1.0)
                route_progress = float(self.travel_distance) / route_len
                station_phase = float(self.last_station.station_id) / max(
                    len(self.effective_station) - 1, 1)
                prev_launch_gap = 1.0
                next_launch_gap = 1.0
                if len(self.forward_bus) != 0:
                    prev_launch_gap = (
                        float(self.launch_time)
                        - float(getattr(self.forward_bus[0], 'launch_time',
                                        self.launch_time - target_hw))
                    ) / target_hw_safe
                if len(self.backward_bus) != 0:
                    next_launch_gap = (
                        float(getattr(self.backward_bus[0], 'launch_time',
                                      self.launch_time + target_hw))
                        - float(self.launch_time)
                    ) / target_hw_safe
                day_phase = (
                    (float(current_time) % (14.0 * 3600.0))
                    / (14.0 * 3600.0))
                time_angle = 2.0 * np.pi * day_phase
                prev_queue = 0.0
                next_queue = 0.0
                current_idx = self.effective_station.index(self.last_station)
                prev_idx = current_idx - 1
                next_idx = current_idx + 1
                if 0 <= prev_idx < len(self.effective_station):
                    prev_queue = (
                        len(self.effective_station[prev_idx].waiting_passengers)
                        / self._lower_context_queue_norm)
                if 0 <= next_idx < len(self.effective_station):
                    next_queue = (
                        len(self.effective_station[next_idx].waiting_passengers)
                        / self._lower_context_queue_norm)
                context_values = {
                    'load': float(np.clip(load_norm, 0.0, 2.0)),
                    'capacity': float(np.clip(cap_remain_norm, 0.0, 1.0)),
                    'queue': float(np.clip(queue_norm, 0.0, 2.0)),
                    'speed_residual': float(np.clip(speed_residual, -2.0, 2.0)),
                    'shock_age': float(np.clip(shock_age, 0.0, 1.0)),
                    'schedule_slack': float(np.clip(schedule_slack, -2.0, 2.0)),
                    'fwd_headway_norm': float(np.clip(fwd_norm, 0.0, 3.0)),
                    'bwd_headway_norm': float(np.clip(bwd_norm, 0.0, 3.0)),
                    'headway_balance': float(np.clip(headway_balance, -3.0, 3.0)),
                    'hold_value_proxy': float(np.clip(hold_value_proxy, -1.0, 1.0)),
                    'route_progress': float(np.clip(route_progress, 0.0, 1.0)),
                    'station_phase': float(np.clip(station_phase, 0.0, 1.0)),
                    'prev_launch_gap': float(np.clip(prev_launch_gap, 0.0, 3.0)),
                    'next_launch_gap': float(np.clip(next_launch_gap, 0.0, 3.0)),
                    'time_sin': float(np.sin(time_angle)),
                    'time_cos': float(np.cos(time_angle)),
                    'prev_queue': float(np.clip(prev_queue, 0.0, 2.0)),
                    'next_queue': float(np.clip(next_queue, 0.0, 2.0)),
                }
                gate = self._lower_context_gate_value
                self.obs.extend([
                    gate * context_values[name]
                    for name in self._lower_context_features
                    if name in context_values
                ])

            if self._lower_frequency_enabled and self._frequency_tracker is not None:
                station_id = getattr(self.last_station, 'station_id', self.next_station.station_id)
                self.obs.extend(
                    self._frequency_tracker.lower_features(
                        station_id, self.direction).tolist()
                )

            self.reward, self.cost = self._headway_reward_cost(target_hw)

        if action_requested or self.unobserved_action_mode == "legacy_stale":
            self.state = BusState.WAITING_ACTION
        else:
            # No policy observation was emitted, so there is no action to
            # execute. Reusing the previous station's action would create a
            # physical hold with no corresponding replay transition.
            self.dwelling_time = 0.0
            self.state = BusState.DWELLING

    def _start_dwelling(self, action, current_time=None):
        dwell_time = self._normalize_action(action)
        if dwell_time is not None:
            dwell_time = max(0.0, dwell_time)

        should_record = (
            dwell_time is not None
            and not (self.trip_id in [0, 1] and action is None)
            and (
                dwell_time > 0.0
                or self.holding_action_trace_mode == "all_decisions"
            )
        )
        if should_record:
            self.applied_actions.append(float(dwell_time))
            load = int(len(getattr(self, 'passengers', ())))
            if not hasattr(self, 'applied_action_loads'):
                self.applied_action_loads = []
            self.applied_action_loads.append(load)
            self.episode_commanded_hold_vehicle_seconds = float(getattr(
                self, 'episode_commanded_hold_vehicle_seconds', 0.0)) + float(
                    dwell_time)
            self.episode_commanded_hold_person_seconds = float(getattr(
                self, 'episode_commanded_hold_person_seconds', 0.0)) + (
                    float(dwell_time) * load)
            self._active_holding_load = load

        if (self.trip_id in [0, 1] and action is None) or dwell_time is None or dwell_time == 0:
            self.dwelling_time = 0
            if dwell_time == 0:
                self.last_action_s = 0.0
                self.last_action_time = current_time
                self.last_action_station_id = int(self.last_station.station_id)
        else:
            self.dwelling_time = dwell_time
            self.last_action_s = float(dwell_time)
            self.last_action_time = current_time
            self.last_action_station_id = int(self.last_station.station_id)

        self.state = BusState.DWELLING

    def _headway_reward_cost(self, target_headway):
        """Return the common station-arrival reward and constraint signal."""
        target_hw = max(float(target_headway), 1.0)

        def headway_reward(headway):
            return -min(abs(float(headway) - target_hw) / target_hw, 1.0)

        if getattr(self, '_headway_reward_mode', 'symmetric_legacy') == (
                'forward_event_only'):
            if getattr(self, 'forward_headway_source', None) != 'arrival_event':
                return 0.0, 0.0
            reward = headway_reward(self.forward_headway)
            headway_dev = (
                (float(self.forward_headway) - target_hw) / target_hw)
            return float(reward), float(min(headway_dev ** 2, 1.0))

        has_forward = bool(self.forward_bus)
        has_backward = bool(self.backward_bus)
        forward_reward = (
            headway_reward(self.forward_headway) if has_forward else None)
        backward_reward = (
            headway_reward(self.backward_headway) if has_backward else None)
        if forward_reward is not None and backward_reward is not None:
            fwd_dev = abs(float(self.forward_headway) - target_hw)
            bwd_dev = abs(float(self.backward_headway) - target_hw)
            weight = fwd_dev / (fwd_dev + bwd_dev + 1e-6)
            similarity_bonus = -min(
                abs(float(self.forward_headway) - float(self.backward_headway))
                / target_hw,
                1.0,
            ) * 0.3
            reward = (
                forward_reward * weight
                + backward_reward * (1.0 - weight)
                + similarity_bonus
            )
        elif forward_reward is not None:
            reward = forward_reward
        elif backward_reward is not None:
            reward = backward_reward
        else:
            reward = -1.0

        headway_dev = (float(self.forward_headway) - target_hw) / target_hw
        return float(reward), float(min(headway_dev ** 2, 1.0))

    def _process_dwelling(self, current_time):
        if self.dwelling_time is not None and self.dwelling_time > 0.0:
            executed_s = min(1.0, float(self.dwelling_time))
            self.episode_hold_vehicle_seconds = float(getattr(
                self, 'episode_hold_vehicle_seconds', 0.0)) + executed_s
            self.episode_hold_person_seconds = float(getattr(
                self, 'episode_hold_person_seconds', 0.0)) + (
                    executed_s * int(getattr(self, '_active_holding_load', 0)))
        if self.dwelling_time is None or self.dwelling_time <= 1:
            self.in_station = False
            if self._stop_start_time is not None:
                self.stop_records.append([
                    self._stop_station,
                    self._stop_start_time,
                    current_time
                ])
                self._stop_start_time = None
                self._stop_station = None
            recorder = getattr(self, '_headway_recorder', None)
            if recorder is not None and hasattr(recorder, 'record_departure'):
                recorder.record_departure(
                    int(self.last_station.station_id),
                    bool(self.direction),
                    float(current_time),
                    int(self.trip_id),
                )
            self.dwelling_time = 0
            self._active_holding_load = 0
            self.state = BusState.TRAVEL
        else:
            self.dwelling_time -= 1

    def _normalize_action(self, action):
        if action is None:
            return None
        if isinstance(action, numbers.Number):
            return float(action)
        if isinstance(action, np.ndarray):
            if action.size == 0:
                return None
            return float(action.reshape(-1)[0])
        if isinstance(action, (list, tuple)):
            if not action:
                return None
            return self._normalize_action(action[0])
        if hasattr(action, 'item'):
            try:
                return float(action.item())
            except (TypeError, ValueError):
                return None
        try:
            return float(action)
        except (TypeError, ValueError):
            return None

    def _recorded_forward_headway(self, current_time):
        """Return the causal same-stop arrival headway when one is available."""
        recorder = getattr(self, '_headway_recorder', None)
        if recorder is None:
            return None
        if hasattr(recorder, 'previous_arrival_event'):
            event = recorder.previous_arrival_event(
                int(self.next_station.station_id), bool(self.direction))
            if event is None:
                return None
            self.forward_predecessor_trip_id = int(event['trip_id'])
            previous = float(event['time_s'])
        elif hasattr(recorder, 'previous_arrival_time'):
            previous = recorder.previous_arrival_time(
                int(self.next_station.station_id), bool(self.direction))
        else:
            return None
        if previous is None:
            return None
        return max(0.0, float(current_time) - float(previous))

    def _update_pre_action_forward_headway(self, current_time):
        """Match the immediate predecessor's causal departure at this stop."""
        self.pre_action_forward_headway = None
        self.pre_action_forward_headway_source = "unavailable"
        recorder = getattr(self, '_headway_recorder', None)
        predecessor = getattr(self, 'forward_predecessor_trip_id', None)
        if (recorder is None or predecessor is None
                or not hasattr(recorder, 'previous_departure_event')):
            return
        event = recorder.previous_departure_event(
            int(self.last_station.station_id), bool(self.direction))
        if event is None or int(event['trip_id']) != int(predecessor):
            self.pre_action_forward_headway_source = "predecessor_not_departed"
            return
        gap_s = float(current_time) - float(event['time_s'])
        if not np.isfinite(gap_s) or gap_s < 0.0:
            return
        self.pre_action_forward_headway = gap_s
        self.pre_action_forward_headway_source = "matched_departure_event"

    def arrive_station(self, current_time, bus_all, debug):
        # Because we have to use the self.holding_time later, so we exchange passenger first when arrived a station
        # self.exchange_passengers(current_time) # self.holding_time is set in this function
        # Update forward_bus backward_bus and relative reward when a bus is arrived a station(except terminal)

        # record the start time and station when the bus stops
        self.current_speed = 0
        self._stop_start_time = current_time
        self._stop_station = self.next_station.station_name

        self.forward_bus = list(filter(
            lambda x: self.trip_id - 2 in x.trip_id_list, bus_all))
        recorded_headway = self._recorded_forward_headway(current_time)
        if recorded_headway is not None:
            # The recorder has not seen this bus yet, so this is the exact
            # same-stop headway to the preceding arrival with no look-ahead.
            self.forward_headway = recorded_headway
            self.forward_headway_source = "arrival_event"
        elif len(self.forward_bus) != 0:
            forward = self.forward_bus[0]
            if not forward.on_route:
                forward_travel_distance = (
                    len(self.stations_list) // 2 * 500
                    + forward.travel_distance)
            else:
                forward_travel_distance = forward.travel_distance
            elapsed_s = max(
                float(current_time) + float(self.holding_time)
                - float(self.launch_time),
                1.0,
            )
            average_speed = max(float(self.travel_distance), 0.0) / elapsed_s
            if average_speed > 1e-6:
                distance_gap = max(
                    0.0,
                    float(forward_travel_distance) - float(self.travel_distance),
                )
                self.forward_headway = distance_gap / average_speed
                self.forward_headway_source = "spatial_fallback"
            else:
                self.forward_headway = float(self._target_headway)
                self.forward_headway_source = "target_default"
        else:
            self.forward_headway = float(self._target_headway)
            self.forward_headway_source = "target_default"

        self.backward_bus = list(filter(lambda x: self.trip_id + 2 in x.trip_id_list, bus_all))
        self.backward_headway = self.backward_bus[0].forward_headway if len(self.backward_bus) != 0 else 360
        # self.backward_headway = 360
        # when the bus arrives at a station, drive() will switch the state to HOLDING so this logic only executes once
        self.absolute_distance += self.next_station_dis * self.direction_int
        # station_type == 0, means the next_station is terminal, then put this bus to terminal_bus rather than on_route
        # then change the direction of the bus.
        if self.next_station.station_type == 0 and self.on_route:
            self.last_completed_trip_id = int(self.trip_id)
            self.last_completed_direction = bool(self.direction)
            self.last_completed_reward, self.last_completed_cost = (
                self._headway_reward_cost(self._target_headway)
            )
            self.last_completed_station_id = int(self.next_station.station_id)
            self.last_completed_target_headway = float(self._target_headway)
            self.last_completed_board_wait_sum_s = float(
                self.last_board_wait_sum_s)
            self.last_completed_board_lf_wait_sum_s = float(
                self.last_board_lf_wait_sum_s)
            self.last_completed_board_hf_wait_sum_s = float(
                self.last_board_hf_wait_sum_s)
            self.last_completed_board_lf_mass = float(self.last_board_lf_mass)
            self.last_completed_board_hf_mass = float(self.last_board_hf_mass)
            self.last_completed_board_count = int(self.last_board_count)
            self.on_route = False
            self.back_to_terminal_time = current_time
            self.last_station = self.effective_station[-1]
            self.direction = int(not self.direction)
            station_split = (len(self.stations_list) + 1) // 2
            self.effective_station = self.stations_list[:station_split] if self.direction else self.stations_list[station_split - 1:]
            self.next_station = self.next_station_func()
        else:
            # if next_station is normal station, update last_station to its next_station, reset the relative distance of bus
            # if len(self.forward_bus) != 0:
            #     print('original_reward_place:', self.reward)
            station_id = self.last_station.station_id + 1 if self.direction else self.last_station.station_id - 1
            self.headway_dif.append([self.forward_headway - self.backward_headway, station_id])
            self.bus_update()

    # When a bus is re-launched from terminal, we have to reset the bus like a new bus we created, which means
    # we have to reset many attribute of the bus, then we add the trip_id to the trip history list. absolute_distance is 0
    # if it begins from terminal up, rather than 11500 if it begins from terminal down.

    def reset_bus(self, trip_num, launch_time):
        self.trip_id = trip_num
        self.trip_id_list.append(trip_num)
        self.launch_time = launch_time
        self.last_station = self.effective_station[0]

        self.forward_headway = 360
        self.backward_headway = 360
        self.forward_headway_source = "target_default"
        self.forward_predecessor_trip_id = None
        self.pre_action_forward_headway = None
        self.pre_action_forward_headway_source = "unavailable"

        self.last_station_dis = 0.
        self.next_station_dis = self.current_route.distance
        self.absolute_distance = 0. if self.direction else len(self.stations_list) // 2 * 500

        self.passengers = np.array([])
        self.current_speed = 0.
        self.holding_time = 0.
        self.back_to_terminal_time = None
        self.last_completed_trip_id = None
        self.last_completed_direction = None
        self.last_completed_reward = None
        self.last_completed_cost = None
        self.last_completed_station_id = None
        self.last_completed_target_headway = None
        self.last_completed_board_wait_sum_s = 0.0
        self.last_completed_board_lf_wait_sum_s = 0.0
        self.last_completed_board_hf_wait_sum_s = 0.0
        self.last_completed_board_lf_mass = 0.0
        self.last_completed_board_hf_mass = 0.0
        self.last_completed_board_count = 0
        self.board_num = 0.
        self.alight_num = 0.
        self.last_board_wait_sum_s = 0.0
        self.last_board_lf_wait_sum_s = 0.0
        self.last_board_hf_wait_sum_s = 0.0
        self.last_board_lf_mass = 0.0
        self.last_board_hf_mass = 0.0
        self.last_board_count = 0
        self.last_board_station_id = int(self.next_station.station_id)
        self.last_board_time = None
        self.last_service_dwell_s = 0.0
        self.last_left_behind_count = 0
        self.last_action_s = 0.0
        self.last_action_time = None
        self.last_action_station_id = int(self.last_station.station_id)
        self.applied_actions = []  # v2: reset per-trip action tracking
        self.applied_action_loads = []
        self._active_holding_load = 0
        self.in_station = False
        self.forward_bus = None
        self.backward_bus = None
        self.reward = None
        self.obs = []
        self.cost = None

        self.state = BusState.TRAVEL
        self.on_route = True
        self.trip_turn = len(self.trip_id_list)
