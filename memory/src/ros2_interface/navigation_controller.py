#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2导航控制器模块
负责与Nav2交互，发送导航目标和监控导航状态
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.task import Future
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav2_msgs.action import NavigateToPose
from tf2_ros import TransformListener, Buffer, TransformException
# from tf2_ros.transform_listener import TransformException
import tf_transformations
import threading
import time
from typing import List, Tuple, Optional


class NavigationController:
    """ROS2导航控制器类"""
    
    def __init__(self, node: Node, map_frame: str = "map", robot_frame: str = "base_link"):
        """
        初始化导航控制器
        
        Args:
            node: ROS2节点实例
            map_frame: 地图坐标系
            robot_frame: 机器人坐标系
        """
        self.node = node
        self.map_frame = map_frame
        self.robot_frame = robot_frame
        
        # 初始化TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        
        # 初始化Nav2动作客户端
        self.nav_client = ActionClient(
            self.node, 
            NavigateToPose, 
            'navigate_to_pose'
        )
        
        self.node.get_logger().info("Waiting for Nav2 action server...")
        if self.nav_client.wait_for_server(timeout_sec=10.0):
            self.node.get_logger().info("Connected to Nav2 server")
        else:
            self.node.get_logger().warn("Could not connect to Nav2 server")
        
        # 导航状态
        self.is_navigating = False
        self.navigation_thread = None
        self.current_goal = None
        self.goal_handle = None
        
        # 状态映射
        self.status_map = {
            GoalStatus.STATUS_ACCEPTED: "已接受",
            GoalStatus.STATUS_EXECUTING: "执行中", 
            GoalStatus.STATUS_CANCELING: "取消中",
            GoalStatus.STATUS_SUCCEEDED: "已完成",
            GoalStatus.STATUS_CANCELED: "已取消",
            GoalStatus.STATUS_ABORTED: "已中止"
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
            goal_msg = self._create_goal(position)
            self.current_goal = position
            
            self.node.get_logger().info(
                f"Sending navigation goal: x={position[0]:.2f}, "
                f"y={position[1]:.2f}, yaw={position[2]:.2f}"
            )
            
            # 异步发送目标
            self.send_goal_future = self.nav_client.send_goal_async(
                goal_msg,
                feedback_callback=self._feedback_callback
            )
            self.send_goal_future.add_done_callback(self._goal_response_callback)
            
            self.is_navigating = True
            
            return True
            
        except Exception as e:
            self.node.get_logger().error(f"Failed to send navigation goal: {str(e)}")
            return False
    
    def cancel_navigation(self):
        """取消当前导航"""
        if self.is_navigating and self.goal_handle:
            self.node.get_logger().info("Canceling navigation...")
            cancel_future = self.goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self._cancel_done_callback)
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
        
        if self.goal_handle:
            status = self.goal_handle.status
            return self.status_map.get(status, f"未知状态({status})")
        
        return "等待目标响应"
    
    def get_robot_position(self) -> Optional[Tuple[float, float, float]]:
        """
        获取机器人当前位置
        
        Returns:
            Tuple[float, float, float]: (x, y, yaw) 或 None
        """
        try:
            # 获取变换
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                now,
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            # 提取位置
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            
            # 转换四元数到欧拉角
            quaternion = (
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            )
            euler = tf_transformations.euler_from_quaternion(quaternion)
            yaw = euler[2]
            
            return (x, y, yaw)
            
        except TransformException as e:
            self.node.get_logger().debug(f"Failed to get robot position: {str(e)}")
            return None
    
    def shutdown(self):
        """关闭控制器"""
        if self.is_navigating:
            self.cancel_navigation()
            if self.navigation_thread:
                self.navigation_thread.join(timeout=2.0)
    
    def _create_goal(self, position: List[float]) -> NavigateToPose.Goal:
        """
        创建导航目标
        
        Args:
            position: [x, y, w] 坐标和朝向
            
        Returns:
            NavigateToPose.Goal: 导航目标
        """
        goal_msg = NavigateToPose.Goal()
        
        # 设置目标位姿
        goal_msg.pose.header.frame_id = self.map_frame
        goal_msg.pose.header.stamp = self.node.get_clock().now().to_msg()
        
        # 设置位置
        goal_msg.pose.pose.position.x = position[0]
        goal_msg.pose.pose.position.y = position[1]
        goal_msg.pose.pose.position.z = 0.0
        
        # 设置朝向（从弧度转换为四元数）
        q = tf_transformations.quaternion_from_euler(0, 0, position[2])
        goal_msg.pose.pose.orientation.x = q[0]
        goal_msg.pose.pose.orientation.y = q[1]
        goal_msg.pose.pose.orientation.z = q[2]
        goal_msg.pose.pose.orientation.w = q[3]
        
        return goal_msg
    
    def _goal_response_callback(self, future):
        """目标响应回调"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.node.get_logger().info('Goal rejected')
            self.is_navigating = False
            return
        
        self.node.get_logger().info('Goal accepted')
        self.goal_handle = goal_handle
        
        # 获取结果
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self._get_result_callback)
    
    def _get_result_callback(self, future):
        """结果回调"""
        result = future.result().result
        status = future.result().status
        
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info('Navigation goal reached successfully!')
        elif status == GoalStatus.STATUS_ABORTED:
            self.node.get_logger().warn('Navigation aborted!')
        elif status == GoalStatus.STATUS_CANCELED:
            self.node.get_logger().info('Navigation canceled!')
        
        self.is_navigating = False
        self.goal_handle = None
    
    def _feedback_callback(self, feedback_msg):
        """反馈回调"""
        feedback = feedback_msg.feedback
        # 可以在这里处理导航反馈信息
        pass
    
    def _cancel_done_callback(self, future):
        """取消完成回调"""
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.node.get_logger().info('Goal successfully canceled')
        else:
            self.node.get_logger().info('Goal failed to cancel')