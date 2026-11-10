import numpy as np

class Route(object):
    def __init__(self, route_id, start_stop, end_stop, route_length, max_speed, route_speed_history, sigma=1.5):
        self.route = []
        self.maximum_velocity = 0
        self.variant_velocity = 0

        self.sigma = sigma
        self.route_id = route_id
        self.route_max_speed = max_speed
        self.speed_history = route_speed_history
        self.speed_limit = 15

        self.start_stop = start_stop
        self.end_stop = end_stop
        self.distance = route_length

    def route_update(self, current_time, effective_period, scenario_tape=None):
        current_hour = effective_period[min(current_time//3600, len(effective_period) -1)]
        mean_log_speed = float(self.speed_history.loc[current_hour])
        if scenario_tape is None:
            sampled_log_speed = float(np.random.normal(mean_log_speed, self.sigma))
        else:
            sampled_log_speed = scenario_tape.normal_stream(
                mean_log_speed,
                self.sigma,
                "route_speed",
                int(self.route_id),
                str(self.start_stop),
                str(self.end_stop),
            )
        v = np.clip(sampled_log_speed, 2, 15)
        self.speed_limit = min(self.route_max_speed, max(int(v), 0))
