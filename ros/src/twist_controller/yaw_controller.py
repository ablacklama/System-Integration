from math import atan

class YawController(object):
    def __init__(self, **ros_param):
        # original param format:
        # https://github.com/udacity/CarND-Capstone/blob/master/ros/src/twist_controller/yaw_controller.py
        self.min_speed = ros_param['min_speed']
        self.wheel_base = ros_param['wheel_base']
        self.steer_ratio = ros_param['steer_ratio']
        self.max_lat_accel = ros_param['max_lat_accel']
        self.min_angle = -ros_param['max_steer_angle']
        self.max_angle = ros_param['max_steer_angle']

    def get_angle(self, radius):
        angle = atan(self.wheel_base / radius) * self.steer_ratio
        return max(self.min_angle, min(self.max_angle, angle))

    def get_steering(self, linear_velocity, angular_velocity, current_velocity):
        angular_velocity = current_velocity * angular_velocity / linear_velocity if abs(linear_velocity) > 0. else 0.

        if abs(current_velocity) > 0.1:
            max_yaw_rate = abs(self.max_lat_accel / current_velocity);
            angular_velocity = max(-max_yaw_rate, min(max_yaw_rate, angular_velocity))

        return self.get_angle(max(current_velocity, self.min_speed) / angular_velocity) if abs(angular_velocity) > 0. else 0.0;
