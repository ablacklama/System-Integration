#!/usr/bin/env python

import math

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.
'''


class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        # Save param from launch file as dictionary
        ros_param_dict = {
            'vehicle_mass': rospy.get_param('~vehicle_mass', 1736.35),  # must need for stoping the car
            'fuel_capacity': rospy.get_param('~fuel_capacity', 13.5),  # optional go precise with the weight
            'brake_deadband': rospy.get_param('~brake_deadband', .1),
            'decel_limit': rospy.get_param('~decel_limit', -5),
            'accel_limit': rospy.get_param('~accel_limit', 1.),
            'wheel_radius': rospy.get_param('~wheel_radius', 0.2413),  # must
            'wheel_base': rospy.get_param('~wheel_base', 2.8498),
            'steer_ratio': rospy.get_param('~steer_ratio', 14.8),
            'max_lat_accel': rospy.get_param('~max_lat_accel', 3.),
            'max_steer_angle': rospy.get_param('~max_steer_angle', 8.)
        }

        # Subscribe to all the topics you need to
        rospy.Subscriber('/current_velocity', TwistStamped, self.cur_velocity_cb)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cmd_cb)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)

        # Publish topics.
        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd', SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd', ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd', BrakeCmd, queue_size=1)

        self.controller = Controller(**ros_param_dict)

        # List of variables need to use
        self.dbw_enabled = None

        self.cur_linear_velocity = None    # may need low pass filter to reduce noisy
        self.cur_angular_velocity = None

        self.target_linear_velocity = None
        self.target_angular_velocity = None

        # flag for checking msg received
        self.twist_cmd = False
        self.current_velocity = False

        self.loop()

    def cur_velocity_cb(self, msg):
        self.current_velocity = True
        self.cur_linear_velocity = msg.twist.linear.x
        self.cur_angular_velocity = msg.twist.angular.z

    def twist_cmd_cb(self, msg):
        self.twist_cmd = True
        self.target_linear_velocity = msg.twist.linear.x
        self.target_angular_velocity = msg.twist.angular.z

    def dbw_enabled_cb(self, msg):
        # painful lesson: uncheck the manual to activate the topic otherwise nothing in echo topic
        self.dbw_enabled = msg.data
        rospy.loginfo(rospy.get_caller_id() + " dbw status is : %s", self.dbw_enabled)

    def loop(self):
        """
        Get predicted throttle, brake, and steering using `twist_controller`
        throttle, brake, steering = self.controller.control(<proposed linear velocity>,
                                                            <proposed angular velocity>,
                                                            <current linear velocity>,
                                                            <dbw status>,
                                                            <any other argument you need>)
        """
        rate = rospy.Rate(50)  # 50Hz Do not Change! Lower than 20Hz will shut down the vehicle
        while not rospy.is_shutdown():
            # When current_velocity or twist_cmd is not ready
            if ((not self.current_velocity) or (not self.twist_cmd)):
                continue

            throttle, brake, steer = self.controller.control(self.target_linear_velocity,
                                                             self.target_angular_velocity,
                                                             self.cur_linear_velocity,
                                                             self.dbw_enabled)

            # You should only publish the control commands if dbw is enabled
            if self.dbw_enabled:
                self.publish(throttle, brake, steer)
            else:
                self.controller.reset()

            rate.sleep()

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    try:
        DBWNode()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start DBW node.')
