#!/usr/bin/env python
import os
import math
import tf
import rospy
import itertools
from collections import deque
import numpy as np
from std_msgs.msg import Int32, Bool
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

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

LOOKAHEAD_WPS = 100 # Number of waypoints to publish.
# Maximum distance from any waypoint position to current car position. 10 KM radius.
MAX_DISTANCE_FROM_WPP = 10000. # This can be total distance in wp's.
MAX_DECELERATION_RATE = 9 # m/s^2. Reference ReadMe.txt.
SAFE_DECELERATION_RATE = int(MAX_DECELERATION_RATE)/3 # m/s^2; Apply one-thrid

class ControlState:
    ''' Control states used by the vehicle to act upon. '''

    SafeDecelerate = 1
    QuickDecelerate = 2
    Stop = 3

class WaypointUpdater(object):
    def __init__(self):

        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb,
                queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb,
                queue_size=1)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)

        # Publishing.
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.waypoints = None
        self.waypoints_tree = None
        self.num_waypoints = 0
        self.max_velocity = 2.8 # This is updated in waypoints_cb.
        self.current_car_pos = None
        self.current_car_wpi = -1
        self.dbw_enabled = False # DBW enabled; Autonomous state.
        self.final_waypoints = deque()
        self.publish_sequence_num = 0
        # styx_msgs/msg/TrafficLight.msg, unknown is 4.
        self.traffic_light = 4
        self.publishing_rate = 12.0 #50.0 # Current rate of DBW and traffic-light pub.
        self.prev_stop_wpi = -1
        self.next_stop_wpi = -1
        self.queue_nextwp_idx = 0
        self.num_wps_to_stop = 0
        self.recalculate_velocity = False

        rospy.loginfo("Starting loop...")

        self.loop()
        #self.test_loop()

    def test_loop(self):
        ''' Test loop to validate following scenarios:
        1. To test waypoint updater: Disable camera and manual mode.
        2. To test tl_detector just enable Camera.
        3. To try velocity & get_next_wpi accuracy check the output before
        stopping.
        4. Test controllers with varying speeds.
        '''
        stop_iterations = 0
        print_once = True
        rospy_rate = rospy.Rate(self.publishing_rate)
        while not rospy.is_shutdown():
            if not (self.dbw_enabled and self.waypoints and self.current_car_pos):
                rospy_rate.sleep()
                continue

            self.current_car_wpi = self.get_next_wpi(self.current_car_pos)
            curr_idx = self.current_car_wpi
            self.final_waypoints_init(curr_idx)

            # START Test 1; Tests decreasing velocity wrt to current wpi; check for
            # stop at 400 + len(vel)
            # When running this test, disable Camera and Manual.
            # 400 is the curve point, so stopping here would be great.
            if curr_idx > 400:
                # Assuming these calculated velocities are given and rest are
                # 0.0
                vel = [4.0, 3.75, 3.5, 3.0, 2.75, 2.5, 2.25, 2.0, 1.75, 1.5,
                        1.25, 1.0, .75, .5, .25, 0.0]
                d = curr_idx - 400
                if d < len(vel)-2:
                    del vel[:d]
                else: # ignore last few
                    vel = None

                if vel:
                    new_v = vel
                    rospy.logwarn("car wpi:{0}".format(curr_idx))
                    new_v.extend([0.0 for i in range(len(vel), LOOKAHEAD_WPS)])
                    for idx, _ in enumerate(self.final_waypoints):
                        self.final_waypoints[idx].twist.twist.linear.x = new_v[idx]
                    #rospy.logwarn("Final waypoints velocities:{0} published".format(
                    #        [x.twist.twist.linear.x for x in self.final_waypoints]))
                else:
                    if print_once and curr_idx >= 417:
                        rospy.logwarn("Car Stopped?. Car wpi:{0}".format(curr_idx))
                        print_once = False
                        rospy.logerr("FAILED, If more than 417; WPI:{0}!!!".
                                format(curr_idx))

                    for idx, _ in enumerate(self.final_waypoints):
                        self.final_waypoints[idx].twist.twist.linear.x = 0.0
                    # Disable 2 lines above and lines below to test 0.0 to
                    # max_velocity start.
                    #stop_iterations += 1
                    #if stop_iterations <= 100: # wait for few seconds
                    #    # Stop now!
                    #    for idx, _ in enumerate(self.final_waypoints):
                    #        self.final_waypoints[idx].twist.twist.linear.x = 0.0
                    #else:
                    #    for idx, _ in enumerate(self.final_waypoints):
                    #        self.final_waypoints[idx].twist.twist.linear.x = self.max_velocity

            self.publish_waypoints(self.final_waypoints)
            # END TEST1.
            rospy_rate.sleep()

    def loop(self):
        prev_idx = 0
        wp_velocities = None
        eval_control = True
        rospy_rate = rospy.Rate(self.publishing_rate)
        while not rospy.is_shutdown():

            if not (self.dbw_enabled and self.waypoints and self.current_car_pos):
                rospy_rate.sleep()
                continue

            self.current_car_wpi = self.get_next_wpi(self.current_car_pos)
            # Halt if we are close to final waypoints.
            if self.current_car_wpi >= self.num_waypoints - 10:
                self.final_waypoints_init(self.current_car_wpi)
                self.update_waypoints_velocity(None, len(self.final_waypoints))
                self.publish_waypoints(self.final_waypoints)
                rospy_rate.sleep()
                continue

            num_new_wps = 0
            # Final waypoints init.
            curr_idx = self.current_car_wpi
            num_new_wps = curr_idx - prev_idx
            if ((not self.final_waypoints) or (self.queue_nextwp_idx < curr_idx)):
                self.queue_nextwp_idx = self.final_waypoints_init(curr_idx)
            else: # Vehicle is between the final waypoints.
                for i in xrange(num_new_wps):
                    self.final_waypoints.popleft()

                    # Append new waypoint.
                    waypoint = self.waypoints[self.queue_nextwp_idx]
                    self.queue_nextwp_idx = ((self.queue_nextwp_idx+1) %
                            self.num_waypoints)
                    self.final_waypoints.append(waypoint)

            # Set Control State.
            if self.recalculate_velocity:
                if eval_control: # Avoid extra compute.
                    control_state = self.set_stopping_state()
                    wp_velocities = self.calculate_waypoints_velocity(control_state)
                    rospy.loginfo("car wpi: {0}, car pos wpi: {1}, "
                            "Stop line wpi: {2}, Waypoints to stop line velocities: {3}".
                            format(curr_idx, self.current_car_wpi, self.next_stop_wpi, wp_velocities))
                    self.update_waypoints_velocity(wp_velocities, 0)
                    eval_control = False
                else:
                    # when enabling this debug log; uncomment position
                    # calculation in pose_cb method.
                    if curr_idx != prev_idx:
                        rospy.loginfo("car wpi: {0}, car pos wpi:{1},"
                                "velocity given for car wpi:{2}".
                                format(curr_idx, self.current_car_wpi,
                                    self.final_waypoints[0].twist.twist.linear.x))
                    self.update_waypoints_velocity(wp_velocities, num_new_wps)
            else:
                eval_control = True
                wp_velocities = None
                for idx, _ in enumerate(self.final_waypoints):
                    self.final_waypoints[idx].twist.twist.linear.x = self.max_velocity

            self.publish_waypoints(self.final_waypoints)
            rospy_rate.sleep()
            prev_idx = curr_idx

    def final_waypoints_init(self, start):
        ''' Creates an inital list of waypoints to publish.
        Returns next starting index for processing.
        '''

        final_waypoints_idx = [idx % self.num_waypoints for idx in
                xrange(start, start + LOOKAHEAD_WPS)]
        self.final_waypoints = deque([self.waypoints[i] for i in
                final_waypoints_idx])
        next_start_idx = ((start + LOOKAHEAD_WPS) % self.num_waypoints)
        return next_start_idx

    def publish_waypoints(self, final_waypoints):

        lane = Lane()
        lane.waypoints = final_waypoints
        self.final_waypoints_pub.publish(lane)

    def calculate_waypoints_velocity(self, control_state):
        '''
        Calculate velocity for waypoints based on current state.
        Returns a list of calculated velocities upto stopline.
        '''

        '''
        Calculation:
        Use V = V0 + a*t. Since we deal with negative velocity,
         we use Vt = V(t-1)-Constant_acceleration * delta_time.
        delta_time = Total_time/Number_of_indexes
        Total time = Velocity of Car/Acceleration (Constants based on
        Control state.)
        number_of_indexs = stop_index - current_car_wpi.
        At every waypoint between car and traffic light is updated
        such that car stops by traffic light at constant rate.
        Final time is sliced by number of waypoints between car and traffic
        light-Delta_stopping distance.
        '''

        next_wps_velocities = deque()
        stop_time = 0

        if control_state == ControlState.SafeDecelerate:
            stop_time = self.max_velocity / SAFE_DECELERATION_RATE
            # First index is car's current position.
            delta_time = stop_time / self.num_wps_to_stop
        elif control_state == ControlState.QuickDecelerate:
            stop_time = self.max_velocity / MAX_DECELERATION_RATE
            delta_time = stop_time / self.num_wps_to_stop
        else: # STOP
            vi = 0

        v_prev = self.max_velocity
        # For absolute stops (v=0); 0.0 is set at delta wp for late detection
        # cases.
        delta_wp = 3 if self.max_velocity <= 4.0 else 5
        # Fill the velocities queue.
        for idx in xrange(self.num_wps_to_stop):
            if control_state == ControlState.SafeDecelerate:
                # Velocity at waypoint i= vi = v0 - a*delta_time.
                vi = v_prev - (
                        SAFE_DECELERATION_RATE * delta_time)
            elif control_state == ControlState.QuickDecelerate:
                vi = v_prev - (
                        MAX_DECELERATION_RATE * delta_time)
            else: # Stop
                vi = 0.0

            # Avoid tripping at .0 floats.
            vi = round((vi*8.0)/8.1, 2)
            vi = vi if vi >= .5 else 0.
            # With current deceleration rates, we could end up stopping early.
            # Add low acceleration values to end.
            if vi == 0. and idx < self.num_wps_to_stop - delta_wp:
                next_wps_velocities.appendleft(self.max_velocity)
            else:
                next_wps_velocities.append(vi)
            v_prev = vi

        if next_wps_velocities:
            return next_wps_velocities
        else:
            rospy.logwarn("No waypoints between Car and Stopline.")
            return None

    def update_waypoints_velocity(self, wp_velocities, num_new_wps):
        ''' Update final waypoints velocities with calculated velocities.'''

        # Remove any past indexes.
        for i in xrange(num_new_wps):
            if wp_velocities:
                wp_velocities.popleft()

        wp_v_len = len(wp_velocities) if wp_velocities else 0
        for idx, _ in enumerate(self.final_waypoints):
            if idx < wp_v_len:
                vi = wp_velocities[idx]
            else:
                vi = 0.0
            self.final_waypoints[idx].twist.twist.linear.x = vi

    def set_stopping_state(self):
        '''
        Returns one of the stopping ranges from StoppingRange.
        '''

        '''
        Using kinematic formula
        V' = U + a*t; If we inverse the points starting from Traffic light to
        current position. U = 0 (assuming its stopped). or
        U = V'+at; U = 0 so V' = -at.
        Assuming a = safe deceleration rate;  V' = a * t; t = V'/a.
        safe stopping distance would be d = V' * t; replacing 't' from above,
        safe_distance = (V' * V') / a.

        Like above, we calculate a threshold distance for
        Emergency_stop_distance using High deceleration rate (constant above).

        Relating traffic light state machine.
        G -> Y -> R -> G1
        |              |
        |______<_______|

        a) safe_stop_dist >= car_to_stopline > emergency_stop_dist =
                Decelerate. Green -> yellow.
        b) emergency_stop_dist >= car_to_stopline > STOP_DELTA = Stop.
            This state is not preferred or could be improved.
            This is either Y=>R or Y|R detected very late.
            Former is okay, later is bad; So log error and continue.
        '''
        curr_velocity = self.max_velocity
        car_wpi = self.current_car_wpi
        stop_wpi = self.next_stop_wpi - 3

        car_to_stopline = self.distance(self.waypoints, car_wpi, stop_wpi)
        self.num_wps_to_stop = stop_wpi - car_wpi
        rospy.loginfo("Stopline wpi: {0}, Distance to stop: {1},"
                "num wpi to stop:{2}".
                format(stop_wpi, car_to_stopline, self.num_wps_to_stop))

        car_velocity_pow = math.pow(curr_velocity, 2)
        safe_stop_dist = car_velocity_pow/SAFE_DECELERATION_RATE
        emergency_stop_dist = car_velocity_pow/MAX_DECELERATION_RATE

        rate = None
        if car_to_stopline >= safe_stop_dist:
            rate = ControlState.SafeDecelerate
        elif (car_to_stopline < safe_stop_dist and
            car_to_stopline >= emergency_stop_dist):
            rate = ControlState.QuickDecelerate
        else:
            rate =  ControlState.Stop

        rospy.loginfo("Car to stop line: {0}; Safe stop dist: {1}, "
                " Emergency stop distance: {2}".format(car_to_stopline,
                    safe_stop_dist, emergency_stop_dist))
        rospy.loginfo("Setting control state:{0} 1:Safe Deceleration, "
                "2: Quick decelerate, 3: Hard Stop".format(rate))
        return rate


    def dbw_enabled_cb(self, msg):
        '''
        Returns if vehicle is in Drive-By-Wire mode/ Autonomous mode.
        '''

        self.dbw_enabled = msg.data
        rospy.loginfo("DBW enabled: {0}".format(self.dbw_enabled))

        # Reset control variables.
        if not self.dbw_enabled:
            self.current_car_wpi = -1
            self.next_stop_wpi = -1
            self.prev_stop_wpi = -1
            self.final_waypoints = deque()


    def pose_cb(self, msg):
        '''
        Callback method for reciveing vehicle position.
        Returns a list of waypoints close enough
        to the recorded waypoint of vehicle current position.
        '''
        self.current_car_pos = msg.pose
        # Leave it for future debug.
        #if self.waypoints:
        #    self.current_car_wpi = self.get_next_wpi(self.current_car_pos)



    def waypoints_cb(self, msg):
        '''Call back to load all waypoints. '''

        # Return if we get similar waypoints.
        if self.waypoints and (sorted(self.waypoints) ==
                sorted(msg.waypoints)):
            return

        self.waypoints = msg.waypoints
        waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for
                wp in self.waypoints]
        self.waypoints_tree = KDTree(waypoints_2d)

        self.num_waypoints = len(self.waypoints)

        # Get average velocity and use it if no TL is detected.
        mid_i = int(self.num_waypoints/2)
        avg_velocity = sum([self.waypoints[i].twist.twist.linear.x for i in
            xrange(mid_i, mid_i+10)])
        self.max_velocity = round(avg_velocity/10, 2)
        rospy.loginfo("Avg velocity:{0}".format(self.max_velocity))

        total_waypoint_distance = self.distance(self.waypoints, 0, self.num_waypoints-1)
        rospy.loginfo("Total waypoints: {0}".format(self.num_waypoints))
        rospy.loginfo('Distance covered by waypoints: {0}'.format(total_waypoint_distance))

    def traffic_cb(self, msg):
        ''' Load traffic light waypoint index.'''

        if not self.waypoints:
            return

        # next_stop_wpi is where the car will attempt to stop. It will be
        # car center at the line instead of front of car. Skip few
        # waypoints from the stop line index.
        # Recieves the index of the stop line closest to traffic light.
        self.next_stop_wpi = msg.data if msg.data > 0 else -1

        if self.next_stop_wpi > 0:
            if self.prev_stop_wpi != self.next_stop_wpi:
                rospy.loginfo("Recieved new Stopline: {0}; Recalculating velocity..".
                        format(self.next_stop_wpi))
                self.recalculate_velocity = True
                self.prev_stop_wpi = self.next_stop_wpi
            else: # Same stop light reported.
                pass
        else:
            self.recalculate_velocity = False

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, idx, velocity):
        '''
        Updates final waypoints velocity.
        '''
        self.final_waypoints[idx].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1_i, wp2_i):
        ''' Overall distance between two waypoints identified by index. '''

        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)
        for i in xrange(wp1_i, wp2_i+1):
            dist += dl(waypoints[wp1_i].pose.pose.position,
                    waypoints[i].pose.pose.position)
            wp1_i = i
        return dist

    def get_next_wpi(self, from_pos):
        wp_c = from_pos.position
        min_i = self.waypoints_tree.query([wp_c.x, wp_c.y], k=1, eps=0.5)[1]

        # Ref: Udacity walkthrough.
        wp = self.waypoints[min_i]
        closest_coord = [wp.pose.pose.position.x, wp.pose.pose.position.y]
        wp_prev = self.waypoints[min_i -1]
        prev_coord = [wp_prev.pose.pose.position.x, wp_prev.pose.pose.position.y]

        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([wp_c.x, wp_c.y])

        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)
        if val > 0:
            min_i = (min_i + 1) % self.num_waypoints
        return min_i

        return min_i

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
