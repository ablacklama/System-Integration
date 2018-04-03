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
    def __init__(self, **ros_param):

        self.max_velocity = 0.0
        kp = 0.5
        ki = 0.
        kd = 0.
        self.brake_deadband = ros_param['brake_deadband']
        self.steer_ratio = ros_param['steer_ratio']

        self.yaw_controller = YawController(**ros_param)
        self.pid = PID(kp, ki, kd, ros_param['decel_limit'],
                        ros_param['accel_limit'])

        self.low_pass_filter = LowPassFilter(0.05, 0.02)

    def control(self, target_linear_velocity, target_angular_velocity,
                      cur_linear_velocity, dbw_status):
        # Check input info is ready
        if not dbw_status:
            return 0., 0., 0.
        else:
            # Handle throttle.
            # Ref discussions
            throttle = 0.
            throttle_err = self.pid.step(
                    target_linear_velocity-cur_linear_velocity,
                    0.02)
            if throttle_err > 0.:
                throttle = self.low_pass_filter.filt(throttle_err)
            else:
                throttle = 0.

            # Handle brake.
            brake = 0.
            if throttle_err < -self.brake_deadband:
                brake = -throttle_err
            if target_linear_velocity <= 0.01 and brake < self.brake_deadband:
                brake = self.brake_deadband

            # Handle steering.
            steering = 0.
            if target_linear_velocity > self.max_velocity:
                self.max_velocity = target_linear_velocity

            if self.brake_deadband <= 0.1:
                steering = target_angular_velocity * self.steer_ratio
            else:
                if target_linear_velocity > 0.05:
                    steering = self.yaw_controller.get_steering(
                            self.max_velocity, target_angular_velocity,
                            cur_linear_velocity)
                else:
                    steering = 0.0

            return throttle, brake, steering

    def reset(self):
        self.pid.reset()
