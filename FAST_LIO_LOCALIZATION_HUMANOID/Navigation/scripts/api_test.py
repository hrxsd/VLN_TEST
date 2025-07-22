#!/usr/bin/env python

import rospy
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion, Point
from std_msgs.msg import String
from semantic_nav_api import NavigationByName

if __name__ == '__main__':
    rospy.init_node('navigation_by_name_test')
    
    nav = NavigationByName()
    
    rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, nav.update_robot_pose)
    
    semantic_info, room_info = nav.load_semantic_info()
    
    nav.update_robot_pose(PoseWithCovarianceStamped(pose=PoseWithCovarianceStamped().pose))
    
    nav.find_object_by_name("table")
    
    nav.navigate_to_object(1)