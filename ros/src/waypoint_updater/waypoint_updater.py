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

# Maximum distance from any waypoint position to current car position. 10 KM radius.
MAX_DISTANCE_FROM_WPP = 10000

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
        self.total_waypoint_distance = None
        # styx_msgs/msg/TrafficLight.msg, unknown is 4.
        self.traffic_light = 4

        rospy.spin()

    def pose_cb(self, msg):
        '''
        Returns a list of waypoints close enough
        to the recorded waypoint of vehicle current position.
        '''

        rospy.loginfo('New waypoint recorded {0}, Position: {1}'.format(
                msg.header.seq, msg.pose.position))

        # Note: msg, docs.ros.org/api/geometry_msgs/html/msg/Pose.html
        self.current_waypoint = msg

        if self.waypoints is None:
            rospy.logwarn("Missing Waypoints, Check if /base_waypoints is "
                    "subscribed!")
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
        self.total_waypoint_distance = self.distance(self.waypoints, 0, len(self.waypoints)-1)
        self._print_waypoints()
        rospy.logdebug('Distance covered by waypoints: {0}'.format(self.total_waypoint_distance))

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
        ''' Overall distance between two waypoints identified by index. '''

        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1_i, wp2_i+1):
            dist += dl(waypoints[wp1_i].pose.pose.position,
                    waypoints[i].pose.pose.position)
            wp1_i = i
        return dist

    def find_closest_waypoint(self):
        ''' Return index of the waypoint closest to current position. '''

        i = 0 # Not attempting Range here, Can have many waypoints.
        min_i = 0
        max_dist = MAX_DISTANCE_FROM_WPP
        wp_c = self.current_waypoint.pose.position

        # z cordinate does not play a big role here as we are in 2-D space.
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2) # + (a.z-b.z)**2)
        for next_waypoint in self.waypoints:
            wp_n = next_waypoint.pose.pose.position

            dist = dl(wp_c, wp_n)
            # Once nearest waypoint is hit in all waypoints, index is set
            # at that point. Giving only waypoints in forward direction.
            # This however will not work in scenarios where if car intends to
            # drive reverse.
            if dist < max_dist:
                min_i = i
                max_dist = dist
            i += 1

        return min_i

    def _print_waypoints(self):
        ''' Print co-ordinates tracked by the waypoints. '''

        rospy.loginfo("Total waypoints: {0}".format(len(self.waypoints)))
        for i, waypoint in enumerate(self.waypoints):
            rospy.loginfo("{0} : {1}".format(i, waypoint.pose.pose.position))


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
