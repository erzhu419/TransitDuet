class Passenger(object):
    def __init__(self, t, boarding_station, destination_station):
        self.appear_time = t
        self.boarding_time = None
        self.arrive_time = None

        self.appear_station = boarding_station
        self.destination_station = destination_station

        self.travel_bus = None

        self.boarded = False
        self.arrived = False

        # Frozen at the passenger's generation event. Later filter updates or
        # promotion decisions must not relabel historical credit.
        self.frequency_low_share = 1.0
        self.frequency_high_share = 0.0
        self.frequency_label_time = float(t)
        self.frequency_label_source = "default_low"

    def set_frequency_shares(self, low_share, high_share, source):
        low = float(low_share)
        high = float(high_share)
        if low < 0.0 or high < 0.0:
            raise ValueError("passenger frequency shares must be non-negative")
        total = low + high
        if abs(total - 1.0) > 1e-9:
            raise ValueError("passenger LF/HF shares must sum to one")
        self.frequency_low_share = low
        self.frequency_high_share = high
        self.frequency_label_source = str(source)

    @property
    def travel_time(self):
        return self.arrive_time - self.boarding_time if self.arrived else -1

    @property
    def waiting_time(self):
        return self.boarding_time - self.appear_time if self.boarded else -1
