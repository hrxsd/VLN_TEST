#!/usr/bin/env python3

import rospy
import yaml
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped

class UserInputNode:
    def __init__(self):
        self.object_name_publisher = rospy.Publisher('/object_name', String, queue_size=10)
        self.object_id_publisher = rospy.Publisher('/object_id', String, queue_size=10)
        rospy.Subscriber('/robot_pose_ekf/odom_combined', PoseWithCovarianceStamped, self.update_robot_pose)

        self.robot_pose = None
        self.semantic_info_file = '/home/nuc1003a/ARTS_TEST/src/hdl_localization/config/map/semantic_map.yaml'
        self.semantic_info = self.load_semantic_info()

    def load_semantic_info(self):
        with open(self.semantic_info_file, 'r') as file:
            return yaml.safe_load(file)['semantic_info']
    
    def update_robot_pose(self, msg):
        self.robot_pose = msg.pose.pose

    def publish_object_name(self, object_name):
        rospy.loginfo("Publishing object name: %s", object_name)
        self.object_name_publisher.publish(String(data=object_name))
    
    def publish_object_id(self, object_id):
        rospy.loginfo("Publishing object ID: %s", object_id)
        self.object_id_publisher.publish(String(data=str(object_id)))

    def run(self):
        while not rospy.is_shutdown():
            input_str = input("Enter object name and ID (e.g., bed, 3): ")  # Python 3
            # input_str = raw_input("Enter object name and ID (e.g., bed, 3): ")  # Python 2

            # Parse input
            parts = input_str.split(',')
            if len(parts) != 2:
                rospy.logwarn("Invalid input format. Please enter in the format 'name, id'.")
                continue
            
            object_name = parts[0].strip()
            object_id = parts[1].strip()

            # Publish name and ID
            self.publish_object_name(object_name)
            self.publish_object_id(object_id)

def main():
    rospy.init_node('user_input_node')
    user_input_node = UserInputNode()
    user_input_node.run()

if __name__ == "__main__":
    main()
