#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')
        # for pose cb
        self.pose = None
        # for waypoints cb
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.waypoints_len = None
        # for image cb
        self.has_image = False
        self.camera_image = None
        # ground truth traffic light state
        self.lights = []
        self.stop_line_positions = None

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        # List of positions that correspond to the line to stop in front of for a given intersection
        self.stop_line_wp_id = None
        self.stop_line_positions = self.config['stop_line_positions']

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        tf_model_dict = {'simulator' : 'light_classification/frozen_inference_graph.pb',
                         'realworld' : 'light_classification/real_frozen_inference_graph.pb'}
        self.light_classifier = TLClassifier(PATH_TO_MODEL=tf_model_dict['simulator'])

        # self.listener = tf.TransformListener()
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.loop()

    def loop(self):
        '''
        Publish upcoming red lights at camera frequency (10 Hz).
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.has_image and self.waypoints_tree and self.stop_line_positions:
                light_wp, state = self.process_traffic_lights()
                if self.state != state:
                    self.state_count = 0
                    self.state = state
                elif self.state_count >= STATE_COUNT_THRESHOLD:
                    self.last_state = self.state
                    light_wp = light_wp if state == TrafficLight.RED else -1
                    self.last_wp = light_wp
                    self.upcoming_red_light_pub.publish(Int32(light_wp))
                else:
                    self.upcoming_red_light_pub.publish(Int32(self.last_wp))
                self.state_count += 1
            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg.pose

    def waypoints_cb(self, msg):
        self.waypoints = msg.waypoints
        self.waypoints_len = len(msg.waypoints)
        if not self.waypoints_2d:
            self.waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in self.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """ Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        self.has_image = True
        self.camera_image = msg

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
              traffic_class = self.lights[0].state  # for debug
              cv_image.shape
        Args:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")
        except CvBridgeError as e:
            print(e)
            return TrafficLight.UNKNOWN

        # Get classification
        traffic_class, score = self.light_classifier.get_classification(cv_image)
        # if traffic_class is not TrafficLight.UNKNOWN:
        #     print("Traffic Light is {}, Score is {}".format(traffic_class, score))
        return traffic_class


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        # check if everything is ready
        if (not self.pose) or (not self.waypoints_tree) or (not self.camera_image) or (not self.stop_line_positions):
            return -1, TrafficLight.UNKNOWN

        # Update stop_line_wp_id (we only need to calculate it once)
        if not self.stop_line_wp_id:
            self.stop_line_wp_id = [self.get_closest_waypoint(line[0], line[1]) for line in self.stop_line_positions]

        closest_light = None
        line_wp_idx = None

        car_position_id = self.get_closest_waypoint(self.pose.position.x, self.pose.position.y)
        # find the closest visible traffic light (if one exists)
        mindiff = self.waypoints_len   # TODO: this should be a parameter
                                       #      (if traffic light is too far away, no need to call classifier)
        for i, light in enumerate(self.lights):
            tmp_wp_id = self.stop_line_wp_id[i]
            diff = tmp_wp_id - car_position_id
            if diff > 0 and diff < mindiff:
                mindiff = diff
                closest_light = light
                line_wp_idx = tmp_wp_id

        if closest_light:
            state = self.get_light_state(closest_light)
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
