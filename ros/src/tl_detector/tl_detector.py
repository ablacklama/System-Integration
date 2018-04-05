#!/usr/bin/env python

import cv2
import math
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import yaml

STATE_COUNT_THRESHOLD = 3

# These should be similar to STOP_OFFSET in waypoint_updator.py
MAX_DISTANCE_FROM_WPP  = 10000. # Max distance between 2 waypoints; 10KM radius
# This is 2 times that of STOP_OFFSET from waypoint_updator.py
MIN_STOPPING_DISTANCE = 20 # meters; Threshold distance from traffic light to car.

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.current_pos = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.tl_wpi_needs_update = True
        self.tl_waypoints_idx = dict()
        self.distance_to_tl = 0
        self.publishing_rate = 10.0 #50.0 # 50 Hz
        self.new_image = False
        self.tl_sight_dist = 70. # meters before sending images to
                                          # classifier.
        self.is_simulator = True

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray,
                self.traffic_cb)
        rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(is_simulator=self.is_simulator)

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_tl_wp = -1
        self.state_count = 0

        self.loop()

    def loop(self):
        '''
        Loop at sampling rate to publish detected traffic light indexs.
        '''

        rate = rospy.Rate(self.publishing_rate)
        while not rospy.is_shutdown():

            if not self.new_image:
                continue

            light_wp, state = self.process_traffic_lights()

            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise use previous stable
            state.
            '''

            # Yellow > Min_stopping_distance is considered RED.
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                light_wp = light_wp if state == TrafficLight.RED else -1
                self.last_tl_wp = light_wp
                self.upcoming_red_light_pub.publish(Int32(self.last_tl_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_tl_wp))
            self.state_count += 1

        rate.sleep()

    def pose_cb(self, msg):
        self.current_pos = msg.pose.position

    def waypoints_cb(self, msg):
        '''
        Callback for getting a list of waypoints.
        '''

        # Dont process further if we are getting similar waypoints.
        if self.waypoints and (sorted(self.waypoints) ==
                sorted(msg.waypoints)):
            self.tl_wpi_needs_update = False
            return

        self.tl_wpi_needs_update = True
        self.waypoints = msg.waypoints
        rospy.loginfo("Number of Waypoints received: {0}".format(
                len(self.waypoints)))

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """

        self.new_image = True
        self.camera_image = msg

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """

        i = 0
        min_i = 0
        max_dist = MAX_DISTANCE_FROM_WPP

        dl = lambda x,y,x1,y1: math.sqrt((x-x1)**2 + (y-y1)**2)
        for next_waypoint in self.waypoints:
            wp_n = next_waypoint.pose.pose.position

            dist = dl(x, y, wp_n.x, wp_n.y)
            if dist < max_dist:
                min_i = i
                max_dist = dist
            i += 1

        # Reset to  0 if we reached the end of track.
        if min_i >= len(self.waypoints):
            min_i = 0

        return min_i

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # If the behaviour is different between simulator and real driving then
        # that logic should go here about classifying the image.
        # Either the image size, distance to stop light etc., For now, this
        # assumes the image detection works the same for simulator and real
        # driving.
        if(not self.new_image):
            return TrafficLight.RED, 1.0

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")
        #Get classification
        traffic_class, score = self.light_classifier.get_classification(cv_image)

        return traffic_class, score

    def update_tl_waypoints(self):
        '''
        Map traffic lights to its closest waypoint index.
        '''

        tl_positions = self.config['stop_line_positions']
        for light_pos in tl_positions:
            idx = self.get_closest_waypoint(light_pos[0], light_pos[1])
            self.tl_waypoints_idx[idx] = light_pos

        rospy.loginfo("Updating Traffic light waypoints")
        for idx, pos in self.tl_waypoints_idx.items():
            rospy.loginfo("Traffic light position:{0}, waypoint index:{1}".
                    format(pos, idx))

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closest to the upcoming stop line for a
            traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light_pos = None
        if self.tl_wpi_needs_update or not self.tl_waypoints_idx:
            self.update_tl_waypoints()
            self.tl_wpi_needs_update = False

        # Traffic light and waypoints are fixed; or updated less frequently.
        # If * = Waypoints; # = Traffic light waypoints. Representation:
        #  * * * # * * # * * * * # * * * * # * * # * # ** * * * * # * *

        if self.current_pos:
            car_wpi = self.get_closest_waypoint(self.current_pos.x,
                    self.current_pos.y)
        else:
            return -1, TrafficLight.UNKNOWN # We dont know car position, so is the TL index.

        next_tl_wpi = -1 # This value is expected for green light.

        # Get closest TL waypoint to car position.
        if car_wpi < max(self.tl_waypoints_idx.keys()):
            next_tl_wpi = min([x for x in self.tl_waypoints_idx.keys() if x >
                car_wpi])
        else: # Circular.
            next_tl_wpi = min(self.tl_waypoints_idx.keys())

        # Get the light position corrosponding to the waypoint index.
        light_pos = self.tl_waypoints_idx.values()[
                    self.tl_waypoints_idx.keys().index(next_tl_wpi)]

        if not light_pos:
            return -1, TrafficLight.UNKNOWN

        dl = lambda x, y, x1, y1 : math.sqrt((x-x1)**2 + (y-y1)**2)
        self.distance_to_tl = dl(self.current_pos.x, self.current_pos.y,
                light_pos[0], light_pos[1])

        # Process TL images only if the car is in delta range.
        if self.distance_to_tl <= self.tl_sight_dist:
            state, score = self.get_light_state(light_pos)

            if state == TrafficLight.RED:
                rospy.loginfo("Traffic light:{0}; Red, score:{1}".format(
                    state, score))
                return next_tl_wpi, state
            elif (state == TrafficLight.YELLOW and
                    self.distance_to_tl > MIN_STOPPING_DISTANCE):
                # Treat as Red; stop tl.
                state = TrafficLight.RED
                rospy.loginfo("Traffic light:{0}; Yellow, score:{1}".format(
                    state, score))
                return next_tl_wpi, state
            else:
                # For green and yellow (where car is close to min stopping
                # distance) Just accelerate.
                return -1, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
