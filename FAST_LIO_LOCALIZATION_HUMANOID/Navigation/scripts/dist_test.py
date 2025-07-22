#!/usr/bin/env python3
# coding: utf-8

import rospy
import threading
import math
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseActionGoal
from std_msgs.msg import String
import tf

goal_pose = None
current_pose_map_frame = None
tf_listener = None

def pose_distance(pose1, pose2):
    """计算两个位姿之间的欧氏距离和角度差"""
    position_dist = math.sqrt(
        (pose1.position.x - pose2.position.x) ** 2 +
        (pose1.position.y - pose2.position.y) ** 2
    )
    
    # 提取四元数中的偏航角（绕Z轴旋转）
    yaw1 = 2 * math.atan2(pose1.orientation.z, pose1.orientation.w)
    yaw2 = 2 * math.atan2(pose2.orientation.z, pose2.orientation.w)
    angle_dist = abs(yaw1 - yaw2)

    
    
    return position_dist, angle_dist

def goal_callback(msg):
    global goal_pose
    goal_pose = msg.goal.target_pose.pose  # 从MoveBaseActionGoal中提取目标位姿
    rospy.loginfo("目标点已更新")

def robot_pose_callback(msg):
    global current_pose_map_frame
    try:
        # 获取 Pose 和其对应的 frame_id
        pose_odom = msg.pose.pose
        header = msg.header

        # 创建 PoseStamped 消息并赋予 header
        pose_stamped = PoseStamped()
        pose_stamped.pose = pose_odom
        pose_stamped.header = header

        # 使用最近的时间戳进行转换
        pose_stamped.header.stamp = rospy.Time(0)

        # 将 Pose 转换到 map 坐标系下
        current_pose_map_frame = tf_listener.transformPose("map", pose_stamped).pose
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        rospy.logerr(f"转换为map坐标系失败: {e}")

    position_dist, angle_dist = pose_distance(goal_pose, current_pose_map_frame)

    rospy.loginfo(f"位置差: {position_dist}")
    rospy.loginfo(f"角度差: {angle_dist}")

def main():
    global tf_listener

    rospy.init_node("dist_test_node", anonymous=True)
    tf_listener = tf.TransformListener()
    rospy.Subscriber("/move_base/goal", MoveBaseActionGoal, goal_callback)
    rospy.Subscriber("/robot_pose_ekf/odom_combined", PoseWithCovarianceStamped, robot_pose_callback)

    rospy.spin()

if __name__ == '__main__':
    main()