#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vector Retrieval ROS2 System - Main Entry Point
主程序：负责系统初始化和主循环
"""

import os
import sys
import json
import yaml
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import threading
from typing import Dict

from core.retrieval_engine import RetrievalEngine
from ros2_interface.navigation_controller import NavigationController
from llm.llm_client import LLMClient
from utils.data_loader import DataLoader


class VectorRetrievalSystem(Node):
    """向量检索系统主类 - ROS2版本"""

    def __init__(self, config_path: str):
        """初始化系统"""
        super().__init__('vector_retrieval_ros2_node')
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.get_logger().info("Vector Retrieval ROS2 System Starting...")
        
        # 初始化各个组件
        self._init_components()
        
        # 创建用户输入线程
        self.input_thread = None
        self.running = True
        
    def _init_components(self):
        """初始化系统组件"""
        # 数据加载器
        self.data_loader = DataLoader(self.config['database']['path'])
        self.data = self.data_loader.load_data()
        self.get_logger().info(f"Loaded {len(self.data)} memory items")
        
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
        
        # ROS2导航控制器
        self.nav_controller = NavigationController(
            node=self,
            map_frame=self.config['ros']['map_frame'],
            robot_frame=self.config['ros']['robot_frame']
        )
        
    def process_query(self, user_input: str) -> Dict:
        """处理用户查询"""
        self.get_logger().info(f"Processing query: {user_input}")
        
        # 步骤1: 分析用户意图
        function_call = self.llm_client.analyze_intent(user_input)
        self.get_logger().info(f"Intent analysis result: {function_call}")
        
        # 步骤2: 执行检索
        retrieved_data = self.retrieval_engine.execute_retrieval(function_call)
        self.get_logger().info(f"Retrieved {len(retrieved_data)} items")
        
        # 步骤3: 生成答案
        answer = self.llm_client.generate_answer(user_input, retrieved_data)
        
        # 步骤4: 处理导航请求
        if "position" in answer:
            nav_success = self.nav_controller.send_goal(answer["position"])
            answer["navigation_status"] = (
                "started"
                if nav_success
                else "failed"
            )
            if nav_success:
                answer["text"] += " 我正在导航到目标位置。"
            else:
                answer["text"] += " 导航启动失败，请检查Nav2状态。"
        
        return answer
    
    def start_input_thread(self):
        """启动用户输入线程"""
        self.input_thread = threading.Thread(target=self._input_loop)
        self.input_thread.daemon = True
        self.input_thread.start()
    
    def _input_loop(self):
        """用户输入循环"""  
        print("=== ROS2向量数据库检索系统 ===")
        print("命令说明:")
        print("- 输入问题进行查询")
        print("- 'cancel': 取消导航")
        print("- 'status': 查看状态")
        print("- 'quit/exit': 退出系统")
        print("="*40 + "\n")
        
        while self.running:
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
                answer = self.process_query(user_input)
                
                # 打印结果
                print("\n" + "="*60)
                print(f"回答: {answer['text']}")
                print(f"时间: {answer['time']}")
                if "position" in answer:
                    print(f"目标位置: {answer['position']}")
                    if "navigation_status" in answer:
                        print(f"导航状态: {answer['navigation_status']}")
                print("="*60 + "\n")
                
            except KeyboardInterrupt:
                print("\n检测到中断信号")
                self.shutdown()
                break
            except Exception as e:
                print(f"处理错误: {e}")
                self.get_logger().error(f"Query processing error: {str(e)}")
    
    def _print_status(self):
        """打印系统状态"""
        print("\n=== 系统状态 ===")
        
        # 导航状态
        nav_status = self.nav_controller.get_navigation_status()
        print(f"导航状态: {nav_status}")
        
        # 机器人位置
        robot_pos = self.nav_controller.get_robot_position()
        if robot_pos:
            print(f"机器人位置: x={robot_pos[0]:.2f}, y={robot_pos[1]:.2f}, yaw={robot_pos[2]:.2f}")
        else:
            print("机器人位置: 未知")
        
        # 当前目标
        if self.nav_controller.current_goal:
            goal = self.nav_controller.current_goal
            print(f"当前目标: x={goal[0]:.2f}, y={goal[1]:.2f}, yaw={goal[2]:.2f}")
        
        print("="*30 + "\n")
    
    def shutdown(self):
        """关闭系统"""
        self.running = False
        self.nav_controller.shutdown()
        self.get_logger().info("System shutdown complete")


def main(args=None):
    """主函数"""
    # 初始化ROS2
    rclpy.init(args=args)
    
    # 配置文件路径
    config_path = 'config/config.yaml'
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # 创建节点
    node = VectorRetrievalSystem(config_path)
    
    # 启动输入线程
    node.start_input_thread()
    
    # 使用多线程执行器
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        # 运行节点
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # 清理
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()