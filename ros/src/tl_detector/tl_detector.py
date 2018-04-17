#!/usr/bin/env python

import cv2
import math
import rospy
from std_msgs.msg import Int32, Bool
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree
import yaml

# These should be similar to STOP_OFFSET in waypoint_updator.py
MAX_DISTANCE_FROM_WPP  = 10000. # Max distance between 2 waypoints; 10KM radius
# This is 2 times that of STOP_OFFSET from waypoint_updator.py
MIN_STOPPING_DISTANCE = 7 # meters; Threshold distance from traffic light to car.
USE_CLASSIFIER = True  # Use only ground truth value of Traffic light.
STATE_COUNT_THRESHOLD = 2 if USE_CLASSIFIER else 3

# These should be similar to STOP_OFFSET in waypoint_updator.py
MAX_DISTANCE_FROM_WPP  = 10000. # Max distance between 2 waypoints; 10KM radius
# This is 2 times that of STOP_OFFSET from waypoint_updator.py
MIN_STOPPING_DISTANCE = 7 # meters; Threshold distance from traffic light to car.
USE_CLASSIFIER = False  # Use only ground truth value of Traffic light.

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.current_pos = None
        self.waypoints = None
        self.waypoints_tree = None
        self.camera_image = None
        self.lights = []
        self.wpi_stop_pos_dict = dict()
        self.distance_to_stop = 0
        self.publishing_rate = 10.0 #50.0
        self.new_image = False
        self.tl_sight_dist = 50. # meters before sending images to classifier.
        self.is_simulator = True if rospy.get_param("~simulator") == 1 else False
        rospy.loginfo("SIMULATOR:{0}".format(self.is_simulator))

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        # subscribers and publishers.
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray,
                self.traffic_cb)
        rospy.Subscriber('/image_color', Image, self.image_cb)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        # Stop light config
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.bridge = CvBridge()
        model_path = 'light_classification/frozen_inference_graph.pb'
        if not self.is_simulator:
            model_path = 'light_classification/real_frozen_graph.pb'

        self.light_classifier = None
        if USE_CLASSIFIER:
            self.light_classifier = TLClassifier(model_path)
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_tl_wp = -1
        self.state_count = 0
        self.dbw_enabled = False

        self.loop()

    def loop(self):
        '''
        Loop at sampling rate to publish detected traffic light indexs.
        '''

        rate = rospy.Rate(self.publishing_rate)
        while not rospy.is_shutdown():

            if not (self.new_image and self.current_pos and self.waypoints_tree
                    and self.dbw_enabled):
                rate.sleep()
                continue

            light_wp, state = self.process_traffic_lights()

            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise use previous stable
            state.
            '''

            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                light_wp = light_wp if state == TrafficLight.RED else -1
                self.last_tl_wp = light_wp
                self.upcoming_red_light_pub.publish(Int32(light_wp))
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
            return

        self.waypoints = msg.waypoints
        waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y]
                for wp in self.waypoints]
        self.waypoints_tree = KDTree(waypoints_2d)

        self.update_stop_light_waypoints()
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

    def dbw_enabled_cb(self, msg):
        '''
        Returns if vehicle is in DBW mode.
        '''
        self.dbw_enabled = msg.data

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """

        closest_id = self.waypoints_tree.query([x, y], 1)[1]
        return closest_id

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        except CvBridgeError as e:
            rospy.logwarn(e)
            return TrafficLight.UNKNOWN, 0

        traffic_class, score = self.light_classifier.get_classification(cv_image)

        return traffic_class, score

    def update_stop_light_waypoints(self):
        '''
        Map traffic lights to its closest waypoint index.
        '''

        sl_positions = self.config['stop_line_positions']
        for stop_pos in sl_positions:
            idx = self.get_closest_waypoint(stop_pos[0], stop_pos[1])
            self.wpi_stop_pos_dict[idx] = stop_pos

        rospy.loginfo("Stop line positions from config: {0}".
                format(self.wpi_stop_pos_dict))

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closest to the upcoming stop line for a
            traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        stop_pos = None

        # Traffic light and waypoints are fixed; or updated less frequently.
        # If * = Waypoints; # = Traffic light waypoints. Representation:
        #  * * * # * * # * * * * # * * * * # * * # * # ** * * * * # * *

        next_stop_line_wpi = -1
        car_wpi = self.get_closest_waypoint(self.current_pos.x,
                self.current_pos.y)

        # Get closest stopline waypoint to car position.
        if car_wpi < max(self.wpi_stop_pos_dict.keys()):
            next_stop_line_wpi = min([x for x in self.wpi_stop_pos_dict.keys()
                if x >= car_wpi])
        else: # Circular.
            next_stop_line_wpi = min(self.wpi_stop_pos_dict.keys())

        # Get the stop position corrosponding to the waypoint index.
        stop_pos = self.wpi_stop_pos_dict.values()[
                    self.wpi_stop_pos_dict.keys().index(next_stop_line_wpi)]
        dl = lambda x, y, x1, y1 : math.sqrt((x-x1)**2 + (y-y1)**2)
        self.distance_to_stop = dl(self.current_pos.x, self.current_pos.y,
                stop_pos[0], stop_pos[1])

        # Process TL images only if the car is in delta range.
        if self.distance_to_stop <= self.tl_sight_dist:
            if USE_CLASSIFIER:
                state, score = self.get_light_state(stop_pos)
                if state == TrafficLight.RED:
                    rospy.loginfo("Traffic light:{0}; Red, score:{1}".format(
                            state, score))
                    return next_stop_line_wpi, state
                elif (state == TrafficLight.YELLOW and
                        self.distance_to_stop >= MIN_STOPPING_DISTANCE):
                    # Treat as Red; stop tl.
                    state = TrafficLight.RED
                    rospy.loginfo("Traffic light:{0}; Yellow, score:{1}".format(
                            state, score))
                    return next_stop_line_wpi, state
                else:
                    return -1, state
            else:
                for light in self.lights:
                    if dl(light.pose.pose.position.x, light.pose.pose.position.y,
                            stop_pos[0], stop_pos[1]) <= 25.0:
                        return next_stop_line_wpi, light.state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
