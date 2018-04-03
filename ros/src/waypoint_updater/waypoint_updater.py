#!/usr/bin/env python
import os
import math
import tf
import rospy
import itertools
from collections import deque
from std_msgs.msg import Int32, Bool
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint


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
STOP_OFFSET = 3 # meters; Delta distance from traffic light to car final stop.
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

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)

        # Publishing.
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.waypoints = None
        self.num_waypoints = 0
        self.max_velocity = 2.8 # Carla max velocity. 10 Km/h.
        self.current_car_pos = None
        self.current_car_wpi = -1
        self.total_waypoint_distance = -1
        self.distance_to_tlw = -1 # Distance to traffic light waypoint.
        self.dbw_enabled = False # DBW enabled; Autonomous state.
        self.final_waypoints = deque()
        self.publish_sequence_num = 0
        self.is_dbw_warn_logged = False
        # styx_msgs/msg/TrafficLight.msg, unknown is 4.
        self.traffic_light = 4
        self.publishing_rate = 20.0 #50.0 # Current rate of DBW and traffic-light pub.
        # Tracking vehicle control states.
        self.next_control_state = None
        # Current and previous traffic light waypoint index.
        # These will determine what will be vehicles state.
        self.prev_tl_wpi = -1
        self.next_tl_wpi = -1
        self.queue_nextwp_idx = 0
        self.num_wpi_to_tl = 0
        self.recalculate_velocity = False

        self.batch_count = 0 # Maitain queue in batch.
        rospy.loginfo("Starting loop...")

        self.loop()

    def loop(self):
        prev_vi = self.max_velocity
        rospy_rate = rospy.Rate(self.publishing_rate)
        while not rospy.is_shutdown():

            if not self.validate_dbw():
                if not self.is_dbw_warn_logged:
                    self.is_dbw_warn_logged = True
                continue

            self.batch_count += 1
            # Final waypoints queue.
            if not self.final_waypoints:
                self.queue_nextwp_idx = self.final_waypoints_init()

            # Do batch processing; Publish in batch.
            if (not self.recalculate_velocity) and (self.batch_count <
                    int(LOOKAHEAD_WPS/3)):
                continue
            else: # Update queue.
                self.batch_count = 0
                next_wpi = self.get_next_wpi(self.final_waypoints,
                        self.current_car_pos)
                for i in xrange(next_wpi):
                    self.final_waypoints.popleft()

                # Append new waypoints.
                for i in xrange(next_wpi):
                    waypoint = self.waypoints[self.queue_nextwp_idx]
                    self.queue_nextwp_idx = ((self.queue_nextwp_idx+1) %
                            self.num_waypoints)
                    waypoint.twist.twist.linear.x = self.max_velocity
                    self.final_waypoints.append(waypoint)

            # Set Control State.
            # Update velocity till tl index in final waypoints, continue if
            # final waypoints is less than tl index.
            if self.recalculate_velocity:
                #rospy.logwarn("Recalculating velocity..")
                self.next_control_state = self.set_stopping_state()
                self.recalculate_velocity, prev_vi = \
                        self.update_waypoints_velocity(prev_vi)
            else:
                prev_vi = self.max_velocity
                # If the vehicle is waiting at stop, Keep waiting until green.
                if self.next_tl_wpi > 0:
                    continue
                else:
                    # Re-update velocity if needed.
                    for idx in xrange(LOOKAHEAD_WPS):
                        self.set_waypoint_velocity(idx, self.max_velocity)


            # Skip few index's if car is behind.
            #p1 = self.final_waypoints[0].pose.pose.position
            #p2 = self.current_car_pos.position
            #skip_i = 0
            #delta = math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)
            #if delta > 1.0: # Greater than 1 meter.
            #    skip_i = 3
            #    rospy.logwarn("publishing car_pos and current car_pos delta:{0}".
            #            format(delta))

            # TODO: Notice that if this is missing, re-calculate waypoints but
            # keep their velocities.

            # Publish Final waypoints queue.
            self.publish_waypoints(self.final_waypoints, 0)
            self.is_dbw_warn_logged = False
            rospy_rate.sleep()

    def final_waypoints_init(self):
        ''' Creates an inital list of waypoints to publish.
        Returns next starting index for processing.
        '''

        final_waypoints_idx = self.get_waypoints_idx(self.current_car_wpi,
                self.current_car_wpi + LOOKAHEAD_WPS)
        self.final_waypoints = deque([self.waypoints[i] for i in
                final_waypoints_idx])
        next_start_idx = ((self.current_car_wpi + LOOKAHEAD_WPS) %
                self.num_waypoints)

        for idx, _ in enumerate(self.final_waypoints):
                self.set_waypoint_velocity(idx, self.max_velocity)
        return next_start_idx

    def publish_waypoints(self, final_waypoints, start_index=0):

        lane = Lane()
        lane.waypoints = list(final_waypoints)[start_index:]
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time.now()
        lane.header.seq = self.publish_sequence_num
        self.publish_sequence_num += 1
        self.final_waypoints_pub.publish(lane)

    def update_waypoints_velocity(self, current_velocity):
        '''
        Calculate velocity for waypoints based on current state and
        distance.
        Returns if update continues for next iteration and last index velocity.
        '''

        '''
        Approch 1:
            Use V = V0 + a*t. Since we deal with negative velocity,
            we use Vt = V(t-1)-Constant_acceleration * delta_time.
            delta_time = Total_time/Number_of_indexes
            Total time = Velocity of Car/Acceleration (Constants based on
            Control state.)
            number_of_indexs = tl_index - current_car_position.
            At every waypoint between car and traffic light is updated
            such that car stops by traffic light at constant rate.
            Final time is sliced by number of waypoints between car and traffic
            light-Delta_stopping distance.

        Approch 2:
            Inversing map, if Final velocity is the car curr poisiton and initial
            velocity is traffic_light-delta_distance position;
            Do velocity calculation such that;
            distance_to_next_waypoint/total_distance * (-V) at every step.
            V - V*(Distance_to_next_waypoint)/total.
            V - V * (distance between all waypoints/total)
            V - V * 1 = 0 // At final waypoint.
            Issue: This is easy solution but compute intensive as distance
            between waypoint is calcuated at every index between car current
            position and Traffic light.

        Following uses Approch 1. Kinematic solution to apply velocity.
        '''

        if not self.final_waypoints:
            rospy.loginfo("No Final waypoints to be updated.")
            return

        # Note: Final waypoints queue always have this count.
        if self.num_wpi_to_tl > LOOKAHEAD_WPS:
            # For any future waypoints in queue update the velocity on add.
            rospy.logwarn("Distance to TL > LOOKAHEAD_WPS")

        #rospy.loginfo("Prev_tl_wpi:{0}, next_tl_wpi:{1}".
        #        format(self.prev_tl_wpi, self.next_tl_wpi))
        stop_time = 0
        v_prev = current_velocity

        if self.next_control_state == ControlState.SafeDecelerate:
            stop_time = current_velocity / SAFE_DECELERATION_RATE
            # First index is car's current position.
            delta_time = stop_time / self.num_wpi_to_tl
        elif self.next_control_state == ControlState.QuickDecelerate:
            stop_time = current_velocity / MAX_DECELERATION_RATE
            delta_time = stop_time / self.num_wpi_to_tl
        else: # STOP
            vi = 0

        for idx in xrange(self.num_wpi_to_tl):
            if self.next_control_state == ControlState.SafeDecelerate:
                # Velocity at waypoint i= vi = v0 - a*delta_time.
                vi = v_prev - (
                        SAFE_DECELERATION_RATE * delta_time)
            elif self.next_control_state == ControlState.QuickDecelerate:
                vi = v_prev - (
                        MAX_DECELERATION_RATE * delta_time)
            else: # Stop
                vi = 0.0

            # Update velocity for each waypoint index.
            self.set_waypoint_velocity(idx, vi)
            v_prev = vi

        # set remaining waypoints from TL to beyond as 0.
        for idx in xrange(self.num_wpi_to_tl, LOOKAHEAD_WPS):
            self.set_waypoint_velocity(idx, 0.0)

        #rospy.logwarn("Updating car_to_tl with:{0} velocities".format(
        #        [x.twist.twist.linear.x for x in
        #            itertools.islice(self.final_waypoints, 0,
        #                self.num_wpi_to_tl)]))

        # To recalculate on next set of queue waypoints.
        if self.num_wpi_to_tl > LOOKAHEAD_WPS:
            return True, v_prev
        else:
            return False, 0.

    def get_waypoints_idx(self, start, end):
        ''' Returns a list of waypoint index's in circular fashion. '''
        return [idx % self.num_waypoints for idx in xrange(start, end)]

    def set_stopping_state(self):
        '''
        Returns one of the stopping ranges from StoppingRange.
        '''

        '''
        Using kinematic formula
        V' = U + a*t; If we inverse the points starting from Traffic light to
        current position. U = 0 (assuming its stopped). or
        U = V'+at; U = 0 so V' = -at.
        Assuming a = safe deceleration rate.  V' = a * t.
        t = V'/a.
        safe stopping distance would be d = V' * t; replacing 't' from above,
        safe_distance = (V' * V') / a.

        Similararly like above, we calculate a threshold distance for
        Emergency_stop_distance using High deceleration rate allowed.

        Relating traffic light state machine.
        G -> Y -> R -> G1
        |              |
        |______<_______|

        a) safe_stop_dist >= self.distance_to_tlw > emergency_stop_dist =
                Decelerate. Green -> yellow.
        b) emergency_stop_dist >= self.distance_to_tlw > STOP_DELTA = Stop.
            This state is not preferred or could be improved.
            This is either Y=>R or Y|R detected very late.
            Former is okay, later is bad; So log error and continue.
        c) STOP_DELTA <= self.distance_to_tlw => Accelerate;
            i) R->G state.
            ii) Y-> R
            iii) Y|R detected too late.
            For i & ii its okay to skip light. For ii, we probably are
            decelerating so hard stop is fine and it really is not a hard stop.
            No way to differenciate between i, ii & iii.

        This method relies on 3 constants.
        1) SAFE_DECELERATION_RATE.
        2) MAX_DECELERATION_RATE.
        3) STOP_OFFSET. Distance from light to final car stop position.
        Note: Create a ROS subscription so that TL detector can publish,
              current traffic light along with index. Becomes easy.
        '''


        car_velocity_pow = math.pow(self.max_velocity, 2)
        safe_stop_dist = ((car_velocity_pow/SAFE_DECELERATION_RATE) -
                STOP_OFFSET)
        emergency_stop_dist = ((car_velocity_pow/MAX_DECELERATION_RATE) -
                STOP_OFFSET)

        rate = None
        #rospy.logwarn("Distance to TL:{0}".format(self.distance_to_tlw))
        if (self.distance_to_tlw <= safe_stop_dist and
                self.distance_to_tlw > emergency_stop_dist):
            rate = ControlState.SafeDecelerate
        elif (self.distance_to_tlw <= emergency_stop_dist and
                self.distance_to_tlw > STOP_OFFSET):
            rate = ControlState.QuickDecelerate
        else:
            rate =  ControlState.Stop

        #rospy.logwarn("Setting control state:{0} 1:SD, 2:QD, 3:Stop".format(rate))
        return rate

    def validate_dbw(self):
        '''
        Returns True if vehicle can follow drive-by-wire instructions.
        '''

        if not self.dbw_enabled:
            if not self.is_dbw_warn_logged:
                rospy.logwarn("Vehicle in manual mode.")
            return False
        if not self.waypoints:
            rospy.logwarn("No base waypoints yet!")
            return False
        if not self.current_car_pos:
            rospy.logwarn("Vehicle postion is invalid. (x:{0}, y:{1})".format(
                self.current_car_pos.position.x, self.current_car_pos.position.y))
            return False

        return True

    def dbw_enabled_cb(self, msg):
        '''
        Returns if vehicle is in Drive-By-Wire mode/ Autonomous mode.
        '''

        self.dbw_enabled = msg.data
        rospy.loginfo("DBW enabled: {0}".format(self.dbw_enabled))

        # Reset tracking values if DBW is toggled.
        if not self.dbw_enabled:
            self.next_tl_wpi = -1

    def pose_cb(self, msg):
        '''
        Callback method for reciveing vehicle position.
        Returns a list of waypoints close enough
        to the recorded waypoint of vehicle current position.
        '''
        self.current_car_pos = msg.pose

        if self.waypoints:
            self.current_car_wpi = self.get_next_wpi(
                    self.waypoints, self.current_car_pos)

    def waypoints_cb(self, msg):
        '''Call back to load all waypoints. '''

        # Return if we get similar waypoints.
        if self.waypoints and (sorted(self.waypoints) ==
                sorted(msg.waypoints)):
            return

        self.waypoints = msg.waypoints
        self.num_waypoints = len(self.waypoints)

        # Get average velocity.
        mid_i = self.num_waypoints/2
        avg_velocity = sum([self.waypoints[i].twist.twist.linear.x for i in
            xrange(mid_i, mid_i+20)])
        self.max_velocity = 5.0 #avg_velocity/20 - 2.0
        rospy.loginfo("Avg velocity:{0}".format(self.max_velocity))

        self.total_waypoint_distance = self.distance(self.waypoints, 0, self.num_waypoints-1)
        rospy.loginfo("Total waypoints: {0}".format(self.num_waypoints))
        rospy.loginfo('Distance covered by waypoints: {0}'.format(self.total_waypoint_distance))

    def traffic_cb(self, msg):
        ''' Load traffic light waypoint index.'''

        if not self.waypoints:
            return

        # Recieves closest wpi for nearest Traffic light.
        self.next_tl_wpi = msg.data if msg.data >= 0 else -1

        if self.next_tl_wpi >= 0:
            #rospy.logwarn("traffic_cb: recieved TL:{0}".format(self.next_tl_wpi))
            rospy.loginfo("Recieved stop traffic light. WPI: {0}".
                    format(self.next_tl_wpi))

            if self.prev_tl_wpi != self.next_tl_wpi:
                self.recalculate_velocity = True
                self.distance_to_tlw = self.distance(self.waypoints,
                        self.current_car_wpi, self.next_tl_wpi)
                self.num_wpi_to_tl = self.next_tl_wpi - self.current_car_wpi
                self.prev_tl_wpi = self.next_tl_wpi

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
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)#  + (a.z-b.z)**2)
        for i in xrange(wp1_i, wp2_i+1):
            dist += dl(waypoints[wp1_i].pose.pose.position,
                    waypoints[i].pose.pose.position)
            wp1_i = i
        return dist

    def get_next_wpi(self, waypoints, from_pos):
        '''
        Return index of the waypoint closest to current position
        from the list of waypoints noted.
        from_pos: Position point.
        '''

        min_i = 0
        max_dist = MAX_DISTANCE_FROM_WPP
        wp_c = from_pos.position

        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2) # + (a.z-b.z)**2)
        for idx, next_waypoint in enumerate(waypoints):
            wp_n = next_waypoint.pose.pose.position

            dist = dl(wp_c, wp_n)
            # Pick closest waypoint in positive direction
            if dist < max_dist:
                min_i = idx
                max_dist = dist
            if max_dist < 1.0:
                break

        # Did the car pass the waypoint?
        # Ref: https://discussions.udacity.com/t/how-can-we-tell-if-the-car-have-past-the-nearest-waypoint-or-not/381909/5
        wp_n = self.waypoints[min_i].pose.pose.position
        dx = wp_n.x - wp_c.x
        dy = wp_n.y - wp_c.y
        car_o = from_pos.orientation

        _,_,transform = tf.transformations.euler_from_quaternion([car_o.x,
                car_o.y, car_o.z, car_o.w])
        if math.cos(-transform)*dx - math.sin(-transform)*dy > 0.0:
            min_i += 1
        return min_i

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
