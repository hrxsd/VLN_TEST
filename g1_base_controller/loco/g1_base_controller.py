#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import sys
import os
import time
from threading import Lock

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

class G1BaseController:
    def __init__(self):
        rospy.init_node('g1_base_controller_ros1', anonymous=True)
        
        # 初始化机器人客户端
        self.sport_client = LocoClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        
        # 初始化控制相关变量
        self.cmd_mutex = Lock()
        self.last_cmd = Twist()
        self.last_cmd_time = rospy.Time.now()
        
        # 急停状态
        self.emergency_stop_active = False
        self.emergency_stop_mutex = Lock()
        
        # 添加循环计数器用于调试
        self.loop_count = 0
        
        rospy.loginfo("G1 LocoClient Initialized.")
        
        # 订阅cmd_vel话题
        self.cmd_sub = rospy.Subscriber("/cmd_vel", Twist, self.cmd_vel_callback)
        rospy.loginfo("Subscribed to /cmd_vel")
        
        # 订阅急停信号
        self.emergency_stop_sub = rospy.Subscriber("/emergency_stop", Bool, self.emergency_stop_callback)
        rospy.loginfo("Subscribed to /emergency_stop")
        
        # 启动周期性控制定时器（50Hz）
        self.timer = rospy.Timer(rospy.Duration(0.02), self.control_loop)
        rospy.loginfo("G1 Base Controller started with 50Hz control loop (No LowState).")
    
    def cmd_vel_callback(self, msg):
        """处理来自ROS的cmd_vel消息"""
        with self.cmd_mutex:
            self.last_cmd = msg
            self.last_cmd_time = rospy.Time.now()
            rospy.logdebug(f"Received cmd_vel: vx={msg.linear.x:.2f}, vy={msg.linear.y:.2f}, yaw={msg.angular.z:.2f}")
    
    def emergency_stop_callback(self, msg):
        """处理急停信号"""
        with self.emergency_stop_mutex:
            if msg.data:
                self.emergency_stop_active = True
                rospy.logwarn("EMERGENCY STOP ACTIVATED! All movements will be stopped.")
                
                # 立即发送停止指令
                try:
                    self.sport_client.Move(0.0, 0.0, 0.0)
                    rospy.loginfo("Emergency stop command sent to robot.")
                except Exception as e:
                    rospy.logerr(f"Failed to send emergency stop command: {e}")
                
                # 关闭节点
                rospy.signal_shutdown("Emergency stop requested.")
    
    def control_loop(self, event):
        """50Hz控制循环"""
        if rospy.is_shutdown():
            return
        
        # 检查急停状态
        with self.emergency_stop_mutex:
            if self.emergency_stop_active:
                # 如果急停激活，持续发送零速度
                try:
                    self.sport_client.Move(0.0, 0.0, 0.0)
                except Exception as e:
                    rospy.logerr(f"Failed to send stop command during emergency: {e}")
                return
        
        # 增加循环计数器
        self.loop_count += 1
        
        # 获取当前指令
        with self.cmd_mutex:
            now = rospy.Time.now()
            dt = (now - self.last_cmd_time).to_sec()
            
            # 每50次循环打印一次状态信息
            if self.loop_count % 50 == 0:
                rospy.loginfo(f"Control loop {self.loop_count}: dt={dt:.3f}s since last cmd")
            
            # 检查指令超时
            if dt > 1.0:
                vx, vy, yaw = 0.0, 0.0, 0.0
                if self.loop_count % 50 == 0:
                    rospy.logwarn("Command timeout! Setting velocities to zero.")
            else:
                vx = self.last_cmd.linear.x
                vy = self.last_cmd.linear.y
                yaw = self.last_cmd.angular.z
        
        # 发送控制指令到机器人
        try:
            start_time = rospy.Time.now()
            self.sport_client.Move(vx, vy, yaw)
            end_time = rospy.Time.now()
            duration = (end_time - start_time).to_sec()
            
            # 监控Move()性能
            if duration > 0.1:  # 超过100ms
                rospy.logwarn(f"SLOW Move() in loop {self.loop_count}: {duration*1000:.1f}ms")
            
            # 定期打印正常的控制信息
            if self.loop_count % 50 == 0:
                rospy.loginfo(f"[{self.loop_count}] Sending cmd_vel: vx={vx:.2f}, vy={vy:.2f}, yaw={yaw:.2f}")
            
        except Exception as e:
            rospy.logerr(f"Failed to send movement command in loop {self.loop_count}: {e}")
    
    def stop_all_movements(self):
        """停止所有运动"""
        try:
            self.sport_client.Move(0.0, 0.0, 0.0)
            rospy.loginfo("All movements stopped. Velocities set to zero.")
        except Exception as e:
            rospy.logerr(f"Failed to stop movements: {e}")

if __name__ == '__main__':
    try:
        # 初始化通道工厂
        ChannelFactoryInitialize(0)
        
        # 创建控制器实例
        controller = G1BaseController()
        
        rospy.loginfo("G1 Base Controller is running. Press Ctrl+C to stop.")
        rospy.loginfo("Emergency stop can be triggered via /emergency_stop topic.")
        
        # 保持主线程活跃，处理消息
        rospy.spin()
        
    except rospy.ROSInitException:
        rospy.logerr("ROS initialization failed.")
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupt received. Shutting down.")
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
    finally:
        rospy.loginfo("G1 Base Controller shutdown complete.")