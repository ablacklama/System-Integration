#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
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
        rospy.loginfo('Waypoint Updater ...')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # Subscriber to traffic light location. tl_detector/tl_detector.py
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: NYI, Subscribe to obstacle location.
        # rospy.Subscriber('obstacle_waypoint', UnKnown, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.waypoints = None
        self.current_waypoint = None
        self.max_dist = None
        # styx_msgs/msg/TrafficLight.msg, unknown is 4.
        self.traffic_light = 4

        rospy.spin()

    def pose_cb(self, msg):
        '''
        Returns a list of waypoints close enough
        to the recorded waypoint of vehicle current position.
        '''

        #rospy.loginfo('New waypoint recorded {0}'.format(msg.header.seq))

        # Note: msg, docs.ros.org/api/geometry_msgs/html/msg/Pose.html
        self.current_waypoint = msg

        if self.waypoints is None:
            # No waypoints subscribed yet.
            return

        # Find the closest waypoint.
        index = self.find_closest_waypoint()

        # Format the message and publish.
        # ref: waypoint_updater.py/publish()
        lane = Lane()
        lane.header.frame_id = self.current_waypoint.header.frame_id
        lane.header.stamp = rospy.Time(0)
        # Closest waypoint to max of LOOKAHEAD_WPS.
        lane.waypoints = self.waypoints[index : index+LOOKAHEAD_WPS]
        self.final_waypoints_pub.publish(lane)

    def waypoints_cb(self, msg):
        ''' Load all waypoints. '''

        # Note: wayponints loaded are in Lane format. styx_msg/lane.msg
        self.waypoints = msg.waypoints
        self.max_dist = self.distance(self.waypoints, 0, len(self.waypoints)-1)
        rospy.loginfo('Total waypoints received: {0}'.format(len(self.waypoints)))
        #rospy.loginfo('Max waypoints distance: {0}'.format(self.max_distance))

    def traffic_cb(self, msg):
        ''' Load traffic light state.'''

        self.traffic_light = msg.state

    #def obstacle_cb(self, msg):
    #    # TODO: Callback for /obstacle_waypoint message. We will implement it later
    #    pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1_i, wp2_i):
        ''' Overall distance between 2 waypoint indexs. '''
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1_i, wp2_i+1):
            dist += dl(waypoints[wp1_i].pose.pose.position,
                    waypoints[i].pose.pose.position)
            wp1_i = i
        return dist

    def find_closest_waypoint(self):
        ''' Return index of the current waypoint in all waypoints. '''

        i = 0 # Not attempting Range here, Can have many waypoints.
        min_i = 0
        max_dist = self.max_distance
        wp_c = self.current_waypoint.pose.position

        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for next_waypoint in self.waypoints:
            wp_n = next_waypoint.pose.pose.position

            dist = dl(wp_c, wp_n)
            # Once current waypoint is hit in all waypoints, index is set
            # at that point. Giving only waypoints in forward direction.
            if dist < max_dist:
                min_i = i
                max_dist = dist
            i += 1

        return min_i


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
