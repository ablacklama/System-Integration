from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController

"""
This file contains a stub of the Controller class. You can use this class to implement vehicle control. 
For example, the control method can take twist data as input and return throttle, brake, and steering values. 
Within this class, you can import and use the provided pid.py and lowpass.py if needed for acceleration.
And yaw_controller.py for steering. 

Note that it is not required for you to use these, and you are free to write and import other controllers.
"""

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    # def __init__(self, *args, **kwargs):
    def __init__(self, **ros_param):
        # Init PID TODO: tweaking
        kp = 3.
        ki = 0.005
        kd = 0.1

        self.pid = PID(kp, ki, kd)

        # Init yaw controller min_speed TODO: need test min_speed param
        min_speed = 0.1
        self.yaw_controller = YawController(min_speed, **ros_param)

        pass

    # def control(self, target_linear_velocity, target_angular_velocity,
    #                   cur_linear_velocity, dbw_status, **kwargs):
    def control(self, target_linear_velocity, target_angular_velocity,
                      cur_linear_velocity, dbw_status):
        # Check input info is ready
        if dbw_status is None or False:
            return 0, 0, 0

        # TODO: get throttle and brake
        acc = 0.5

        acc = (target_linear_velocity - cur_linear_velocity) / 0.02

        # get steer angle
        steer = self.yaw_controller.get_steering(target_linear_velocity, target_angular_velocity,
                                                 cur_linear_velocity)

        # Return throttle, brake, steer
        return (acc, 0., steer) if acc > 0. else (0., acc, steer)