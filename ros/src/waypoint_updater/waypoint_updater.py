#!/usr/bin/env python

#hy june-29-18

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math
import numpy as np
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

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

#        rospy.spin()
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopline_wp_ind = -1
        self.loop()
        #self.spin()
    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                #find closest waypoint
                closest_wp_ind = self.get_closest_wp_ind()
                self.publish_waypoints(closest_wp_ind)
            rate.sleep()

    def get_closest_wp_ind(self):     
        #hy 6/29
        car_coord = [self.pose.pose.position.x, self.pose.pose.position.y]
        if self.waypoint_tree != None:
            closest_ind = self.waypoint_tree.query(car_coord, 1)[1]
        else: # didn't receive base_waypoint yet
            closest_ind = 0
 
        num_points = len(self.waypoints_2d)

        closest_coord = self.waypoints_2d[closest_ind]
        closest_m1_coord = self.waypoints_2d[(closest_ind-1)%num_points]

        #check to ensure closest_coord is ahead of vehicle
        #just check dot prod of waypoint_dir and vehicle to closest is > 0

        wp_forward_vec = np.array(closest_coord)-np.array(closest_m1_coord)
        veh_to_closest_vec = np.array(closest_coord)-np.array(car_coord)
        dot_wp = np.dot(wp_forward_vec, veh_to_closest_vec)
        if dot_wp < 0:
            closest_ind = (closest_ind+1)%num_points
        return closest_ind

    def publish_waypoints(self, closest_ind):
        #hy 6/29
        lane = Lane()
        lane.header = self.base_waypoints.header
        far_ind = closest_ind + LOOKAHEAD_WPS
        lane.waypoints = self.base_waypoints.waypoints[closest_ind:far_ind]
        if self.stopline_wp_ind != -1 and self.stopline_wp_ind <= far_ind:
        # need to slow down
           
            temp = [] 
            for i, wp in enumerate(lane.waypoints):
                p = Waypoint()
                p.pose = wp.pose

                stop_ind = max(self.stopline_wp_ind- closest_ind - 2, 0)
                dist = self.distance(lane.waypoints, i, stop_ind)
                vel = math.sqrt(2*MAX_DECEL*dist)
                if vel < 1.0:
                    vel = 0.0
                #set velocity
                vel = 0
                p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
                temp.append(p)
            lane.waypoints = temp

        self.final_waypoints_pub.publish(lane)


    def pose_cb(self, msg):
        #hy 6/29
        self.pose = msg

    def waypoints_cb(self, waypoints):
        '''Stores the /base_waypoints in KDTree, should only occur once'''
        #hy 6/29
        self.base_waypoints = waypoints
        #if not self.waypoints_2d: # callback is only called once,
        self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
        self.waypoint_tree = KDTree(self.waypoints_2d)
        

    def traffic_cb(self, msg):
        #Callback for /traffic_waypoint message
        # -1 if no 
       self.stopline_wp_ind = msg

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


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
