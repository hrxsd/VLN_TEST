#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from remote_controller import RemoteController, KeyMap

class EmergencyStopMonitor:
    def __init__(self):
        rospy.init_node('emergency_stop_monitor', anonymous=True)
        
        # 初始化遥控器
        self.remote_controller = RemoteController()
        
        # 初始化状态订阅
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateHG)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)
        
        # 发布急停信号
        self.emergency_stop_pub = rospy.Publisher('/emergency_stop', Bool, queue_size=1)
        
        # 状态变量
        self.last_button_state = 0
        
        rospy.loginfo("Emergency Stop Monitor initialized.")
        
        # 10Hz检查频率，足够响应急停
        self.timer = rospy.Timer(rospy.Duration(0.1), self.check_emergency_stop)
    
    def LowStateHandler(self, msg: LowStateHG):
        """处理来自机器人的低级状态消息"""
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)
    
    def check_emergency_stop(self, event):
        """检查急停按键"""
        try:
            current_button_state = self.remote_controller.button[KeyMap.select]
            
            # 检测按键按下事件（边沿触发）
            if current_button_state == 1 and self.last_button_state == 0:
                rospy.logwarn("EMERGENCY STOP TRIGGERED!")
                
                # 发布急停信号
                emergency_msg = Bool()
                emergency_msg.data = True
                self.emergency_stop_pub.publish(emergency_msg)
                
                # 也可以直接调用系统shutdown
                rospy.signal_shutdown("Emergency stop requested by remote controller.")
            
            self.last_button_state = current_button_state
            
        except Exception as e:
            rospy.logwarn(f"Emergency stop check failed: {e}")

if __name__ == '__main__':
    try:
        ChannelFactoryInitialize(0)
        monitor = EmergencyStopMonitor()
        rospy.spin()
    except rospy.ROSInitException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo("Emergency stop monitor shutdown.")