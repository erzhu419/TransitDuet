from env.passenger import Passenger
import numpy as np


class Station(object):
    def __init__(self, station_type, station_id, station_name, direction, od):
        # if the station is terminal or not terminal,
        self.station_type = station_type
        # the id of stations
        self.station_id = station_id
        self.station_name = station_name
        # waiting passengers in this station
        self.waiting_passengers = np.array([])
        self.total_passenger = []
        self._pending_passengers = []
        # the direction is True if upstream, else False
        self.direction = direction
        # od is the passengers demand of every hour
        self.od = od

    # def station_update(self, current_time, stations):
    #     # 自己写的
    #     # if self.od is not None:
    #     #     # effective_period_str = effective_period[current_time//3600].strftime("%H:%M:%S")
    #     #     effective_period_str = '0'+str(6+current_time//3600)+':00:00' if 6+current_time//3600 < 10 else str(6+current_time//3600)+':00:00'
    #     #     period_od = self.od[effective_period_str]
    #     #     for destination_name, demand in period_od.items():
    #     #     # for destination_name in effective_station_name:
    #     #     # 对如果period_od[destination_name] == 0,则不计算泊松分布，因为太慢，且太多
    #     #         destination_demand_num = 0 if demand == 0 else np.random.poisson(demand/3600)
    #     #         for _ in range(destination_demand_num):
    #     #             destination = list(filter(lambda x: x.station_name == destination_name and x.direction == self.direction, stations))[0]
    #     #             passenger = Passenger(current_time, self, destination)
    #     #             self.waiting_passengers = np.append(self.waiting_passengers, passenger)
    #     #             self.total_passenger.append(passenger)
    #     #     sorted(self.waiting_passengers, key=lambda i: i.appear_time)
    #
    #     if self.od is not None: # GPT优化的，减少不必要的操作
    #
    #         effective_period_str = f"{6 + current_time // 3600:02}:00:00"
    #         period_od = self.od[effective_period_str]
    #
    #         for destination_name, demand in period_od.items():
    #             if demand > 0:  # 直接过滤掉不需要计算的需求
    #                 destination_demand_num = np.random.poisson(demand / 3600)
    #                 if destination_demand_num > 0:
    #                     destination = next(x for x in stations if x.station_name == destination_name and x.direction == self.direction)
    #                     new_passengers = [Passenger(current_time, self, destination) for _ in range(destination_demand_num)]
    #                     self.waiting_passengers = np.append(self.waiting_passengers, new_passengers)
    #                     self.total_passenger.extend(new_passengers)

    def schedule_passenger_window(
            self, current_time, stations, passenger_update_interval=1,
            demand_multipliers=None, demand_scale=1.0,
            od_multipliers=None, peak_shift=0,
            service_start_hour=6, service_end_hour=19,
            scenario_tape=None):
        """Sample a Poisson window but keep future arrivals unobservable."""
        interval_s = float(passenger_update_interval)
        if interval_s <= 0.0:
            raise ValueError("passenger_update_interval must be positive")
        if self.od is None:
            return 0

        service_start_hour = int(service_start_hour)
        service_end_hour = int(service_end_hour)
        hour = service_start_hour + int(float(current_time) // 3600)
        hour = max(service_start_hour, min(service_end_hour, hour))
        lookup_hour = max(
            service_start_hour,
            min(service_end_hour, hour + int(peak_shift)),
        )
        period_od = self.od.get(f"{lookup_hour:02}:00:00", {})
        demand_mult = 1.0
        if demand_multipliers is not None and hour in demand_multipliers:
            demand_mult = demand_multipliers[hour]
        demand_mult *= max(0.0, float(demand_scale))

        scheduled = 0
        for destination_name, demand in period_od.items():
            if demand <= 0:
                continue
            od_mult = 1.0
            if od_multipliers is not None:
                od_key = (
                    int(self.station_id),
                    bool(self.direction),
                    str(destination_name),
                )
                od_mult = float(od_multipliers.get(od_key, 1.0))
            arrival_rate = (
                float(demand) * demand_mult * od_mult / 3600.0 * interval_s)
            if scenario_tape is None:
                count = int(np.random.poisson(arrival_rate))
            else:
                count = scenario_tape.poisson(
                    arrival_rate,
                    "passenger_arrival_window",
                    int(self.station_id),
                    bool(self.direction),
                    str(destination_name),
                    float(current_time),
                    interval_s,
                )
            if count <= 0:
                continue
            destination = next(
                station for station in stations
                if station.station_name == destination_name
                and station.direction == self.direction
            )
            for index in range(count):
                if scenario_tape is None:
                    offset_s = float(np.random.uniform(0.0, interval_s))
                else:
                    offset_s = scenario_tape.uniform(
                        0.0,
                        interval_s,
                        "passenger_arrival_offset",
                        int(self.station_id),
                        bool(self.direction),
                        str(destination_name),
                        float(current_time),
                        int(index),
                    )
                # A draw at exactly zero must not become visible before the
                # first simulation second in the sampled window has elapsed.
                offset_s = max(offset_s, float(np.nextafter(0.0, 1.0)))
                passenger = Passenger(
                    float(current_time) + offset_s, self, destination)
                self._pending_passengers.append(passenger)
            scheduled += int(count)
        if scheduled:
            self._pending_passengers.sort(key=lambda item: item.appear_time)
        return scheduled

    def release_passengers(self, current_time, return_details=False):
        """Expose only passengers whose sampled arrival time has elapsed."""
        split = 0
        now = float(current_time)
        for passenger in self._pending_passengers:
            if float(passenger.appear_time) > now + 1e-9:
                break
            split += 1
        released = self._pending_passengers[:split]
        if split:
            self._pending_passengers = self._pending_passengers[split:]
            self.waiting_passengers = np.append(
                self.waiting_passengers, released)
            self.total_passenger.extend(released)
        od_counts = {}
        if return_details:
            for passenger in released:
                key = (
                    int(self.station_id),
                    int(passenger.destination_station.station_id),
                    bool(self.direction),
                )
                od_counts[key] = od_counts.get(key, 0) + 1
            return len(released), od_counts
        return len(released)

    def station_update(self, current_time, stations, passenger_update_interval=1,
                        demand_multipliers=None, demand_scale=1.0,
                        od_multipliers=None, peak_shift=0,
                        service_start_hour=6, service_end_hour=19,
                        return_details=False, scenario_tape=None):
        """Compatibility wrapper for scheduling one window and releasing due rows."""
        self.schedule_passenger_window(
            current_time,
            stations,
            passenger_update_interval=passenger_update_interval,
            demand_multipliers=demand_multipliers,
            demand_scale=demand_scale,
            od_multipliers=od_multipliers,
            peak_shift=peak_shift,
            service_start_hour=service_start_hour,
            service_end_hour=service_end_hour,
            scenario_tape=scenario_tape,
        )
        return self.release_passengers(
            current_time, return_details=return_details)
