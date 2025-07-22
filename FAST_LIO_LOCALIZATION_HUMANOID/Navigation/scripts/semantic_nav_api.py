#!/usr/bin/env python

import rospy
import yaml
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion, Point, PoseStamped
import math
from nav_msgs.srv import GetPlan
from nav_msgs.msg import Path

class NavigationByName:
    def __init__(self):
        self.semantic_info_file = '/home/nuc1003a/ARTS_TEST/src/hdl_localization/config/map/semantic_map.yaml'
        self.semantic_info, self.room_info = self.load_semantic_info()
        self.robot_pose = None
        
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        
        rospy.Subscriber('/object_name', String, self.find_object_by_name_callback)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.update_robot_pose)
        rospy.Subscriber('/object_id', String, self.navigate_to_object_callback)
    
    def load_semantic_info(self):
        with open(self.semantic_info_file, 'r') as file:
            data = yaml.safe_load(file)
            return data['semantic_info'], data['rooms']
    
    def update_robot_pose(self, msg):
        self.robot_pose = msg.pose.pose
    
    def find_object_by_name_callback(self, msg):
        object_name = msg.data
        self.find_object_by_name(object_name)
    
    def find_object_by_name(self, object_name):
        rospy.loginfo("Received request to find objects with name: %s", object_name)
        matching_objects = self.get_semantic_info_by_name(object_name)
        if matching_objects:
            rospy.loginfo("Found %d objects with name '%s'", len(matching_objects), object_name)
            for obj in matching_objects:
                rospy.loginfo("Object ID '%d', Position: (%f, %f)", obj['id'], obj['position']['x'], obj['position']['y'])
        else:
            rospy.logwarn("No objects found with name '%s'", object_name)
    
    def navigate_to_object_callback(self, msg):
        object_id = int(msg.data)
        self.navigate_to_object(object_id)
    
    def navigate_to_object(self, object_id):
        rospy.loginfo("Received request to navigate to the object with ID: %d", object_id)
        semantic_info = self.get_semantic_info_by_id(object_id)
        if semantic_info:
            position = semantic_info['position']
            size = semantic_info.get('size', {'x': 0.5, 'y': 0.5})
            room = semantic_info['room']
            rospy.loginfo("Detected objected ID %d in room: %s", object_id, room)
            safe_distance = 0.25
            if self.robot_pose:
                adjusted_position, orientation = self.calculate_safe_position_and_orientation(position, size, safe_distance, room)
                self.send_goal(adjusted_position, orientation)
            else:
                rospy.logwarn("Robot's current position is not available.")
        else:
            rospy.logwarn("Object with ID '%d' not found in semantic info.", object_id)
    
    def get_semantic_info_by_name(self, name):
        return [semantic for semantic in self.semantic_info if semantic['name'] == name]
    
    def get_semantic_info_by_id(self, object_id):
        for semantic in self.semantic_info:
            if semantic['id'] == object_id:
                return semantic
        return None
    
    def calculate_safe_position_and_orientation(self, position, size, safe_distance, room):    
        vector_x = position['x'] - self.robot_pose.position.x
        vector_y = position['y'] - self.robot_pose.position.y
        angle = math.atan2(vector_y, vector_x)
        
        if abs(vector_x) > abs(vector_y):
            if vector_x > 0:
                adjusted_position = {
                    'x': position['x'] - size['x'] / 2 - safe_distance,
                    'y': position['y']
                }
            else:
                adjusted_position = {
                    'x': position['x'] + size['x'] / 2 + safe_distance,
                    'y': position['y']
                }
        else:
            if vector_y > 0:
                adjusted_position = {
                    'x': position['x'],
                    'y': position['y'] - size['y'] / 2 - safe_distance
                }
            else:
                adjusted_position = {
                    'x': position['x'],
                    'y': position['y'] + size['y'] / 2 + safe_distance
                }
        
        room_corners = self.room_info[room]['corners']
        while self.is_in_inflation_layer(adjusted_position):
            safe_distance += 0.1
            if abs(vector_x) > abs(vector_y):
                if vector_x > 0:
                    adjusted_position['x'] -= 0.1
                else:
                    adjusted_position['x'] += 0.1
            else:
                if vector_y > 0:
                    adjusted_position['y'] -= 0.1
                else:
                    adjusted_position['y'] += 0.1
        
        if not self.is_point_in_room(adjusted_position, room_corners):
            rospy.logwarn("Adjusted position is not within the room boundaries. Adjusting...")
            adjusted_position = self.recalculate_position_within_room(position, size, room_corners, safe_distance)
        
        facing_angle = math.atan2(position['y'] - adjusted_position['y'], position['x'] - adjusted_position['x'])
        orientation = Quaternion()
        orientation.z = math.sin(facing_angle / 2.0)
        orientation.w = math.cos(facing_angle / 2.0)
        
        return adjusted_position, orientation
    
    def recalculate_position_within_room(self, position, size, room_corners, safe_distance):
        room_center = {
            'x': sum([corner['x'] for corner in room_corners]) / len(room_corners),
            'y': sum([corner['y'] for corner in room_corners]) / len(room_corners)
        }
        
        direction_vector_center = {
            'x': room_center['x'] - position['x'],
            'y': room_center['y'] - position['y']
        }
        
        length = math.sqrt(direction_vector_center['x'] ** 2 + direction_vector_center['y'] ** 2)
        if length == 0:
            direction_vector_center['x'], direction_vector_center['y'] = 0, 0
        else:
            direction_vector_center['x'] /= length
            direction_vector_center['y'] /= length
        
        if size['x'] < size['y']:
            direction_vector_width = {'x': size['x'], 'y': 0}
        else:
            direction_vector_width = {'x': 0, 'y': size['y']}
        
        length_width = math.sqrt(direction_vector_width['x']**2 + direction_vector_width['y']**2)
        if length_width == 0:
            direction_vector_width['x'], direction_vector_width['y'] = 0, 0
        else:
            direction_vector_width['x'] /= length_width
            direction_vector_width['y'] /= length_width
        
        dot_product = direction_vector_center['x'] * direction_vector_width['x'] + direction_vector_center['y'] * direction_vector_width['y']
        projected_vector = {
            'x': direction_vector_width['x'] * dot_product,
            'y': direction_vector_width['y'] * dot_product
        }
        
        adjusted_position = {
            'x': position['x'] + projected_vector['x'] * safe_distance,
            'y': position['y'] + projected_vector['y'] * safe_distance
        }
        
        while self.is_in_inflation_layer(adjusted_position) or not self.is_point_in_room(adjusted_position, room_corners):
            adjusted_position['x'] += projected_vector['x'] * 0.2
            adjusted_position['y'] += projected_vector['y'] * 0.2
        
        return adjusted_position
    
    def is_point_in_room(self, point, room_corners):
        n = len(room_corners)
        inside = False
        p1x, p1y = room_corners[0]['x'], room_corners[0]['y']
        for i in range(n+1):
            p2x, p2y = room_corners[i % n]['x'], room_corners[i % n]['y']
            if point['y'] > min(p1y, p2y):
                if point['y'] <= max(p1y, p2y):
                    if point['x'] <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (point['y'] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or point['x'] <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def is_in_inflation_layer(self, position):
        rospy.wait_for_service('/move_base/make_plan')
        make_plan = rospy.ServiceProxy('/move_base/make_plan', GetPlan)
        start = PoseStamped()
        start.header.frame_id = 'map'
        start.pose.position = self.robot_pose.position
        start.pose.orientation = self.robot_pose.orientation
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position = Point(position['x'], position['y'], 0)
        goal.pose.orientation = self.robot_pose.orientation
        tolerance = 0.1
        
        plan_request = GetPlan()
        plan_request.start = start
        plan_request.goal = goal
        plan_request.tolerance = tolerance
        try:
            response = make_plan(plan_request.start, plan_request.goal, plan_request.tolerance)
            if response.plan.poses:
                return False
            return True
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)
            return True
    
    def send_goal(self, position, orientation):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = position['x']
        goal.target_pose.pose.position.y = position['y']
        goal.target_pose.pose.orientation = orientation
        
        self.client.send_goal(goal)
        wait = self.client.wait_for_result()
        if not wait:
            rospy.logerr("Action server not available!")
        else:
            return self.client.get_result()

def main():
    rospy.init_node('navigation_by_name')
    nav = NavigationByName()
    rospy.spin()

if __name__ == '__main__':
    main()
