#!/usr/bin/env python

#hy, rd 09/07/2018
#lb 12/07/2018

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import numpy as np

from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoint_tree = None
        self.waypoints_2d = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides the location of the traffic light in 3D map space and
        helps acquiring an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. It will be
        necessary to rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
        
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        
        self.image_cnt = 0
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.image_cnt += 1

        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
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

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """  
        if self.waypoint_tree != None:
            closest_ind = self.waypoint_tree.query([x,y], 1)[1]
        else: # didn't receive base_waypoint yet
            closest_ind = 0
 
        num_points = len(self.waypoints_2d)

        closest_coord = self.waypoints_2d[closest_ind]
        closest_m1_coord = self.waypoints_2d[(closest_ind-1)%num_points]

        #check to ensure closest_coord is ahead of vehicle
        #just check dot prod of waypoint_dir and vehicle to closest is > 0

        wp_forward_vec = np.array(closest_coord)-np.array(closest_m1_coord)
        veh_to_closest_vec = np.array(closest_coord)-np.array([x,y])
        dot_wp = np.dot(wp_forward_vec, veh_to_closest_vec)
        if dot_wp < 0:
            closest_ind = (closest_ind+1)%num_points
        
        return closest_ind

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #for now just test with known light state
        #return light.state
        #uncomment below when classifier works

        return light.state
        """
        if not self.has_image:
            self.prev_light_loc = None
            return False
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        predicted = self.light_classifier.get_classification(cv_image)
        rospy.logwarn('traffic light.state={} and predicted = {}'.fotmat(light.state, predicted))
        return predicted
        """
        
    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_ind = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            x = self.pose.pose.position.x
            y = self.pose.pose.position.y
            car_position_ind = self.get_closest_waypoint(x,y)

        # Find the closest visible traffic light (if one exists)
        diff = len(self.waypoints.waypoints)
        for i, light in enumerate(self.lights):
            line = stop_line_positions[i]
            temp_wp_ind = self.get_closest_waypoint(line[0], line[1])
            dist_ind = temp_wp_ind - car_position_ind
            if dist_ind >= 0 and dist_ind<diff:
                diff = dist_ind
                closest_light = light
                line_wp_ind = temp_wp_ind
        if closest_light:
            state = self.get_light_state(closest_light)
            return line_wp_ind, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
