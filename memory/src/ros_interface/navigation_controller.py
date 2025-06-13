#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROS导航控制器模块
负责与move_base交互，发送导航目标和监控导航状态
"""

import rospy
import actionlib
import threading
import time
from typing import List, Tuple, Optional

from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
import tf
from tf.transformations import quaternion_from_euler


class NavigationController:
    """导航控制器类"""
    
    def __init__(self, map_frame: str = "map", robot_frame: str = "base_link"):
        """
        初始化导航控制器
        
        Args:
            map_frame: 地图坐标系
            robot_frame: 机器人坐标系
        """
        self.map_frame = map_frame
        self.robot_frame = robot_frame
        
        # 初始化move_base客户端
        self.move_base_client = actionlib.SimpleActionClient(
            'move_base', MoveBaseAction
        )
        
        rospy.loginfo("Waiting for move_base action server...")
        if self.move_base_client.wait_for_server(rospy.Duration(10)):
            rospy.loginfo("Connected to move_base server")
        else:
            rospy.logwarn("Could not connect to move_base server")
        
        # 初始化TF监听器
        self.tf_listener = tf.TransformListener()
        
        # 导航状态
        self.is_navigating = False
        self.navigation_thread = None
        self.current_goal = None
        
        # 状态映射
        self.status_map = {
            GoalStatus.PENDING: "等待中",
            GoalStatus.ACTIVE: "执行中",
            GoalStatus.PREEMPTED: "已取消",
            GoalStatus.SUCCEEDED: "已完成",
            GoalStatus.ABORTED: "已中止",
            GoalStatus.REJECTED: "已拒绝",
            GoalStatus.PREEMPTING: "取消中",
            GoalStatus.RECALLING: "召回中",
            GoalStatus.RECALLED: "已召回",
            GoalStatus.LOST: "已丢失"
        }
    
    def send_goal(self, position: List[float]) -> bool:
        """
        发送导航目标
        
        Args:
            position: [x, y, w] 坐标和朝向
            
        Returns:
            bool: 是否成功发送
        """
        try:
            # 取消之前的导航
            if self.is_navigating:
                self.cancel_navigation()
                time.sleep(0.5)
            
            # 创建导航目标
            goal = self._create_goal(position)
            self.current_goal = position
            
            rospy.loginfo(
                f"Sending navigation goal: x={position[0]:.2f}, "
                f"y={position[1]:.2f}, yaw={position[2]:.2f}"
            )
            
            # 发送目标
            self.move_base_client.send_goal(goal)
            self.is_navigating = True
            
            # 启动监控线程
            self.navigation_thread = threading.Thread(
                target=self._monitor_navigation
            )
            self.navigation_thread.daemon = True
            self.navigation_thread.start()
            
            return True
            
        except Exception as e:
            rospy.logerr(f"Failed to send navigation goal: {e}")
            return False
    
    def cancel_navigation(self):
        """取消当前导航"""
        if self.is_navigating:
            rospy.loginfo("Canceling navigation...")
            self.move_base_client.cancel_goal()
            self.is_navigating = False
            self.current_goal = None
    
    def get_navigation_status(self) -> str:
        """
        获取导航状态
        
        Returns:
            str: 导航状态描述
        """
        if not self.is_navigating:
            return "空闲"
        
        state = self.move_base_client.get_state()
        return self.status_map.get(state, f"未知状态({state})")
    
    def get_robot_position(self) -> Optional[Tuple[float, float, float]]:
        """
        获取机器人当前位置
        
        Returns:
            Tuple[float, float, float]: (x, y, yaw) 或 None
        """
        try:
            (trans, rot) = self.tf_listener.lookupTransform(
                self.map_frame, self.robot_frame, rospy.Time(0)
            )
            
            # 转换四元数到欧拉角
            euler = tf.transformations.euler_from_quaternion(rot)
            yaw = euler[2]
            
            return (trans[0], trans[1], yaw)
            
        except (tf.LookupException, tf.ConnectivityException, 
                tf.ExtrapolationException) as e:
            rospy.logdebug(f"Failed to get robot position: {e}")
            return None
    
    def shutdown(self):
        """关闭控制器"""
        if self.is_navigating:
            self.cancel_navigation()
            if self.navigation_thread:
                self.navigation_thread.join(timeout=2.0)
    
    def _create_goal(self, position: List[float]) -> MoveBaseGoal:
        """
        创建导航目标
        
        Args:
            position: [x, y, w] 坐标和朝向
            
        Returns:
            MoveBaseGoal: 导航目标
        """
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.map_frame
        goal.target_pose.header.stamp = rospy.Time.now()
        
        # 设置位置
        goal.target_pose.pose.position.x = position[0]
        goal.target_pose.pose.position.y = position[1]
        goal.target_pose.pose.position.z = 0.0
        
        # 设置朝向（从弧度转换为四元数）
        q = quaternion_from_euler(0, 0, position[2])
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]
        
        return goal
    
    def _monitor_navigation(self):
        """监控导航状态的线程函数"""
        while self.is_navigating and not rospy.is_shutdown():
            state = self.move_base_client.get_state()
            
            if state == GoalStatus.SUCCEEDED:
                rospy.loginfo("Navigation goal reached successfully!")
                self.is_navigating = False
                break
            elif state == GoalStatus.ABORTED:
                rospy.logwarn("Navigation aborted!")
                self.is_navigating = False
                break
            elif state == GoalStatus.PREEMPTED:
                rospy.loginfo("Navigation preempted!")
                self.is_navigating = False
                break
            
            time.sleep(0.5)