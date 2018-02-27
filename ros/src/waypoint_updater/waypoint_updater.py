#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # rospy.Subscriber('/obstacle_waypoint', UnKnown, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.close_waypoint_id = -1
        self.cur_position = None
        self.base_waypoint = None
        self.base_waypoint_len = 0
        # styx_msgs/msg/TrafficLight.msg, unknown is 4.
        self.traffic_light = 4

        rospy.spin()

    def pose_cb(self, msg):
        """ current pose callback
            - update current position
        :param msg:
        """
        self.cur_position = msg.pose.position

        if self.base_waypoint is not None:
            self.find_closest_waypoint()
            self.publish()

    def waypoints_cb(self, msg):
        """ base_waypoints callback
            - load the base_waypoint to class
        :param modified input name: waypoints to msg (avoid confusion)
        """
        self.base_waypoint = msg.waypoints
        self.base_waypoint_len = len(msg.waypoints)

        rospy.loginfo(rospy.get_caller_id() + " base_waypoint len: %s", self.base_waypoint_len)
        # rospy.loginfo(rospy.get_caller_id() + " startid info: \n%s", min_id)

    def traffic_cb(self, msg):
        # Callback for /traffic_waypoint message. Implement
        self.traffic_light = msg.state

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    # Added function
    def find_closest_waypoint(self):
        min_id = -1
        min_dist = 0
        # dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        dl = lambda a, b: ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        for i, wp in enumerate(self.base_waypoint):
            dist = dl(wp.pose.pose.position, self.cur_position)
            if i == 0:
                min_dist = dist
            elif dist < min_dist:
                min_dist = dist
                min_id = i

        if self.cur_position.x > self.base_waypoint[min_id].pose.pose.position.x:
            min_id += 1

        self.close_waypoint_id = min_id

    def publish(self):
        """
            Format the message and publish.
            ref: waypoint_updater.py/publish()
        """
        # rospy.loginfo(rospy.get_caller_id() + " Close to the end! ")
        print "Current at waypoint id: " + str(self.close_waypoint_id)

        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)

        if self.close_waypoint_id + LOOKAHEAD_WPS > self.base_waypoint_len:
            # TODO: we may need to modify the code to deal with out of waypoint probably treat it as a red light or loop
            lane.waypoints = self.base_waypoint[self.close_waypoint_id : self.base_waypoint_len]
            rospy.loginfo(rospy.get_caller_id() + " Close to the end! %s ", self.close_waypoint_id)
        else:
            lane.waypoints = self.base_waypoint[self.close_waypoint_id : self.close_waypoint_id + LOOKAHEAD_WPS]

        self.final_waypoints_pub.publish(lane)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
