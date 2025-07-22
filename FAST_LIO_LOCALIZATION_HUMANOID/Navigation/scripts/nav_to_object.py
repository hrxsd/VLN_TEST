#!/usr/bin/env python3

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
        # load semantic info from yaml file
        self.semantic_info_file = '/home/nuc1003a/ARTS_TEST/src/hdl_localization/config/map/semantic_map.yaml'
        self.semantic_info, self.room_info = self.load_semantic_info()
        self.robot_pose = None  
        
        # Initialize the move_base action client
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        
        # Subscribe to the semantic where the object name is pulished
        self.object_name_subscriber = rospy.Subscriber('/object_name', String, self.find_object_by_name)
        
        # Subscribe to the robot's current pose
        self.robot_pose_subscriber = rospy.Subscriber('/robot_pose_ekf/odom_combined', PoseWithCovarianceStamped, self.update_robot_pose)
        
        # Subscribe to the topic where the user selects a specific object ID
        self.object_id_subscriber = rospy.Subscriber('/object_id', String, self.navigate_to_object)
        
    def load_semantic_info(self):
        with open(self.semantic_info_file, 'r') as file:
            data = yaml.safe_load(file)
            return data['semantic_info'], data['rooms']
    
    def update_robot_pose(self, msg):
        self.robot_pose = msg.pose.pose
    
    def find_object_by_name(self, msg):
        object_name = msg.data
        rospy.loginfo("Received request to find objects with name: %s", object_name)
        matching_objects = self.get_semantic_info_by_name(object_name)
        if matching_objects:
            rospy.loginfo("Found %d objects with name '%s'", len(matching_objects), object_name)
            for obj in matching_objects:
                rospy.loginfo("Object ID '%d', Position: (%f, %f)", obj['id'], obj['position']['x'], obj['position']['y'])
        else:
            rospy.logwarn("No objects found with name '%s'", object_name)
        
    def navigate_to_object(self, msg):
        object_id = int(msg.data)
        rospy.loginfo("Received request to navigate to the object with ID: %d", object_id)
        semantic_info = self.get_semantic_info_by_id(object_id)
        if semantic_info:
            position = semantic_info['position']
            size = semantic_info.get('size', {'x': 0.5, 'y': 0.5})
            room = semantic_info['room']
            rospy.loginfo("Detected objected ID %d in room: %s", object_id, room)
            safe_distance = 0.25  # Define a safe distance in meters
            if self.robot_pose:
                adjusted_position, orientation = self.calculate_safe_position_and_orientation(position, size, safe_distance, room)
                # 取消所有的导航目标，确保机器人不会在导航到目标之前导航到其他目标
                self.client.cancel_all_goals()
                
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
    
    def is_point_in_room(self, point, room_corners):
        # 检查目标点是否在指定房间的多边形范围内 
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
    
    def calculate_safe_position_and_orientation(self, position, size, safe_distance, room):    
        vector_x = position['x'] - self.robot_pose.position.x
        vector_y = position['y'] - self.robot_pose.position.y
        angle = math.atan2(vector_y, vector_x)
        
        # 计算最优的接近位置
        if abs(vector_x) > abs(vector_y):
            if vector_x > 0:
                # 从左边接近
                adjusted_position = {
                    'x': position['x'] - size['x'] / 2 - safe_distance,
                    'y': position['y']
                }
            else:
                # 从右边接近
                adjusted_position = {
                    'x': position['x'] + size['x'] / 2 + safe_distance,
                    'y': position['y']
                }
        else:
            if vector_y > 0:
                # 从下边接近
                adjusted_position = {
                    'x': position['x'],
                    'y': position['y'] - size['y'] / 2 - safe_distance
                }
            else:
                # 从上边接近
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
                    
        # 确保目标点在指定房间内
        if not self.is_point_in_room(adjusted_position, room_corners):
            rospy.logwarn("Adjusted position is not within the room boundaries. Adjusting...")
            adjusted_position = self.recalculate_position_within_room(position, size, room_corners, safe_distance)
        
        # 计算朝向
        facing_angle = math.atan2(position['y'] - adjusted_position['y'], position['x'] - adjusted_position['x'])
        orientation = Quaternion()
        orientation.z = math.sin(facing_angle / 2.0)
        orientation.w = math.cos(facing_angle / 2.0)
        
        return adjusted_position, orientation
    
    def recalculate_position_within_room(self, position, size, room_corners, safe_distance):
        # 获取房间中心点
        room_center = {
            'x': sum([corner['x'] for corner in room_corners]) / len(room_corners),
            'y': sum([corner['y'] for corner in room_corners]) / len(room_corners)
        }
        
        # 计算目标点与房间中心的方向向量
        direction_vector_center = {
            'x': room_center['x'] - position['x'],
            'y': room_center['y'] - position['y']
        }
        
        # 归一化方向向量
        length = math.sqrt(direction_vector_center['x'] ** 2 + direction_vector_center['y'] ** 2)
        if length == 0:
            direction_vector_center['x'], direction_vector_center['y'] = 0, 0
        else:
            direction_vector_center['x'] /= length
            direction_vector_center['y'] /= length
            
        # 计算物体的宽边方向向量
        if size['x'] < size['y']:
            direction_vector_width = {'x': size['x'], 'y': 0}
        else:
            direction_vector_width = {'x': 0, 'y': size['y']}
            
        # 归一化宽边方向向量
        length_width = math.sqrt(direction_vector_width['x']**2 + direction_vector_width['y']**2)
        if length_width == 0:
            direction_vector_width['x'], direction_vector_width['y'] = 0, 0
        else:
            direction_vector_width['x'] /= length_width
            direction_vector_width['y'] /= length_width
            
        # 计算在宽边方向上的投影
        dot_product = direction_vector_center['x'] * direction_vector_width['x'] + direction_vector_center['y'] * direction_vector_width['y']
        projected_vector = {
            'x': direction_vector_width['x'] * dot_product,
            'y': direction_vector_width['y'] * dot_product
        }
        
        # 初始化调整后的目标点
        adjusted_position = {
            'x': position['x'] + projected_vector['x'] * safe_distance,
            'y': position['y'] + projected_vector['y'] * safe_distance
        }
        
        # 动态调整目标点直到满足条件
        while self.is_in_inflation_layer(adjusted_position) or not self.is_point_in_room(adjusted_position, room_corners):
            adjusted_position['x'] += projected_vector['x'] * 0.2
            adjusted_position['y'] += projected_vector['y'] * 0.2
            
        return adjusted_position
        
    def is_in_inflation_layer(self, position):
        # 利用move_base的make_plan service判断目标点是否在障碍物膨胀层里
        try:
            rospy.wait_for_service('/move_base/GlobalPlanner/make_plan')
            make_plan = rospy.ServiceProxy('/move_base/GlobalPlanner/make_plan', GetPlan)
            start = PoseStamped()
            start.header.frame_id = 'map'
            start.pose = self.robot_pose
            
            goal = PoseStamped()
            goal.header.frame_id = 'map'
            goal.pose.position.x = position['x']
            goal.pose.position.y = position['y']
            goal.pose.orientation.w = 1.0
            
            tolerance = 0.0
            response = make_plan(start=start, goal=goal, tolerance=tolerance)
            
            if len(response.plan.poses) > 0:
                last_pose = response.plan.poses[-1].pose
                if last_pose.position.x == position['x'] and last_pose.position.y == position['y']:
                    return False
                else:
                    return True
            else:
                return True
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" %e)
            return True
    
    def send_goal(self, position, orientation):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = position['x']
        goal.target_pose.pose.position.y = position['y']
        goal.target_pose.pose.orientation = orientation
        
        rospy.loginfo("Sending goal to position: x=%f, y=%f", position['x'], position['y'])
        self.client.send_goal(goal)
        
        self.client.wait_for_result(rospy.Duration.from_sec(60.0))
        state = self.client.get_state()
        # self.client.wait_for_result()
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Goal reached.")
        else:
            rospy.logerr("Failed to reach goal. State: %s", self.client.get_goal_status_text())
            
def main():
    rospy.init_node('navigation_by_name')
    NavigationByName()
    rospy.spin()
        
if __name__ == "__main__":
    main()