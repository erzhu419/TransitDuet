class Timetable(object):

    def __init__(self, launch_time, launch_turn, direction, target_headway=360.0):
        self.launch_time = launch_time
        self.direction = direction
        self.launch_turn = launch_turn
        self.launched = False
        # Written by upper policy before dispatch; default 360s (6 min)
        self.target_headway = target_headway
