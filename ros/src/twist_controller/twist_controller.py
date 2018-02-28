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
    def __init__(self, *args, **kwargs):
        # TODO: Implement
        pass

    def control(self, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        return 1., 0., 0.
