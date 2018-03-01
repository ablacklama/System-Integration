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
        # Init PID
        kp = 3.
        ki = 0.005
        kd = 0.1
        self.pid = PID(kp, ki, kd)

        # Init yaw controller min_speed need test
        min_speed = 0.1
        self.yaw_controller = YawController(min_speed, **ros_param)

        pass

    def control(self, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        return 1., 0., 0.
