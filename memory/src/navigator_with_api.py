#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vector Retrieval ROS System - Main Entry Point
主程序：负责系统初始化和主循环
"""

import os
import sys
import json
import yaml
import rospy
from typing import Dict

from core.retrieval_engine import RetrievalEngine
from ros_interface.navigation_controller import NavigationController
from llm.llm_client import LLMClient
from utils.data_loader import DataLoader


class VectorRetrievalSystem:
    """向量检索系统主类"""
    
    def __init__(self, config_path: str):
        """初始化系统"""
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化ROS节点
        rospy.init_node('vector_retrieval_ros_node', anonymous=True)
        rospy.loginfo("Vector Retrieval ROS System Starting...")
        
        # 初始化各个组件
        self._init_components()
        
    def _init_components(self):
        """初始化系统组件"""
        # 数据加载器
        self.data_loader = DataLoader(self.config['database']['path'])
        self.data = self.data_loader.load_data()
        
        # 检索引擎
        self.retrieval_engine = RetrievalEngine(
            data=self.data,
            model_id=self.config['embedding']['model_id']
        )
        
        # LLM客户端
        self.llm_client = LLMClient(
            api_key=self.config['openai']['api_key'],
            api_base=self.config['openai']['api_base'],
            model=self.config['openai']['model']
        )
        
        # ROS导航控制器
        self.nav_controller = NavigationController(
            map_frame=self.config['ros']['map_frame'],
            robot_frame=self.config['ros']['robot_frame']
        )
        
    def process_query(self, user_input: str) -> Dict:
        """处理用户查询"""
        rospy.loginfo(f"Processing query: {user_input}")
        
        # 步骤1: 分析用户意图
        function_call = self.llm_client.analyze_intent(user_input)
        rospy.loginfo(f"Intent analysis result: {function_call}")
        
        # 步骤2: 执行检索
        retrieved_data = self.retrieval_engine.execute_retrieval(function_call)
        rospy.loginfo(f"Retrieved {len(retrieved_data)} items")
        
        # 步骤3: 生成答案
        answer = self.llm_client.generate_answer(user_input, retrieved_data)
        
        # 步骤4: 处理导航请求
        if "position" in answer:
            nav_success = self.nav_controller.send_goal(answer["position"])
            answer["navigation_status"] = "started" if nav_success else "failed"
            if nav_success:
                answer["text"] += " 我正在导航到目标位置。"
            else:
                answer["text"] += " 导航启动失败，请检查move_base状态。"
        
        return answer
    
    def run(self):
        """运行主循环"""
        print("=== ROS向量数据库检索系统 ===")
        print("命令说明:")
        print("- 输入问题进行查询")
        print("- 'cancel': 取消导航")
        print("- 'status': 查看状态")
        print("- 'quit/exit': 退出系统")
        print("="*40 + "\n")
        
        while not rospy.is_shutdown():
            try:
                user_input = input("请输入您的问题: ").strip()
                
                # 处理特殊命令
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("系统关闭中...")
                    self.shutdown()
                    break
                
                if user_input.lower() == 'cancel':
                    self.nav_controller.cancel_navigation()
                    print("导航已取消")
                    continue
                
                if user_input.lower() == 'status':
                    self._print_status()
                    continue
                
                if not user_input:
                    continue
                
                # 处理查询
                result = self.process_query(user_input)
                
                # 显示结果
                print("\n=== 查询结果 ===")
                print(json.dumps(result, ensure_ascii=False, indent=2))
                print("="*40 + "\n")
                
            except KeyboardInterrupt:
                print("\n系统关闭中...")
                self.shutdown()
                break
            except Exception as e:
                rospy.logerr(f"Error: {e}")
    
    def _print_status(self):
        """打印系统状态"""
        nav_status = self.nav_controller.get_navigation_status()
        robot_pos = self.nav_controller.get_robot_position()
        
        print("\n--- 系统状态 ---")
        print(f"导航状态: {nav_status}")
        if robot_pos:
            print(f"机器人位置: x={robot_pos[0]:.2f}, y={robot_pos[1]:.2f}, yaw={robot_pos[2]:.2f}")
        print("-"*20 + "\n")
    
    def shutdown(self):
        """关闭系统"""
        self.nav_controller.shutdown()
        rospy.signal_shutdown("User requested shutdown")


def main():
    """主函数"""
    # 获取配置文件路径
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # 默认配置文件路径
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'config/config.yaml'
        )
    
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)
    
    try:
        # 创建并运行系统
        system = VectorRetrievalSystem(config_path)
        system.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()