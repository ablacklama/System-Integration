import rospy
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
        """
        :param ros_param:
        Note:
            sample time (sec) is based on the dbw node frequency 50Hz
            low_pass filter:
                val = w * cur_val + (1 - w) * prev_val
                w is 0 ~ 1
        """
        # used ros_param
        self.vehicle_mass = ros_param['vehicle_mass']
        # self.fuel_capacity = ros_param['fuel_capacity']
        self.brake_deadband = ros_param['brake_deadband']
        self.decel_limit = ros_param['decel_limit']
        self.accel_limit = ros_param['accel_limit']
        self.wheel_radius = ros_param['wheel_radius']

        self.last_time = rospy.get_time()

        # low pass filter for velocity
        self.vel_lpf = LowPassFilter(0.5, .02)

        # Init yaw controller
        min_speed = 0.1   # I think min_speed
        self.steer_controller = YawController(min_speed, **ros_param)
        self.throttle_lpf = LowPassFilter(0.05, 0.02)  # w = 0.28

        # Init throttle PID controller
        # TODO: tweaking
        kp = 0.5
        ki = 0.005
        kd = 0.1
        acc_min = 0.
        acc_max = self.accel_limit
        self.throttle_controller = PID(kp, ki, kd, acc_min, acc_max)

    def control(self, target_linear_velocity, target_angular_velocity,
                      cur_linear_velocity, dbw_status):
        # Check input info is ready
        if not dbw_status:
            self.throttle_controller.reset()
            return 0., 0., 0.

        # dbw enabled: control!
        cur_linear_velocity = self.vel_lpf.filt(cur_linear_velocity)

        # get steer value
        steering = self.steer_controller.get_steering(target_linear_velocity,
                                                      target_angular_velocity,
                                                      cur_linear_velocity)

        # get throttle (could be < 0 so it will be updated by `get brake` as well)
        vel_err = target_linear_velocity - cur_linear_velocity

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_err, sample_time)

        # get brake
        brake = 0
        if target_linear_velocity == 0. and cur_linear_velocity < 0.1:
            throttle = 0
            brake = 400  # N * m - hold the car in place for the red traffic light
        elif throttle < .1 and vel_err < 0.:
            throttle = 0.
            decel_velocity = max(vel_err, self.decel_limit)   # attention this value is < 0
            # if less than brake_deaband, we don't need to add brake
            # The car will deceleration by friction just release peddle
            if abs(decel_velocity) > self.brake_deadband:
                brake = abs(decel_velocity) * self.vehicle_mass * self.wheel_radius
            else:
                brake = 0

        return throttle, brake, steering

    def reset(self):
        self.throttle_controller.reset()