#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多步导航规划扩展模块 - ROS2集成版本
支持解析复合语言指令，生成多步导航目标序列
"""

import os
import sys
import json
import yaml
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import threading
import time
from typing import Dict, List, Tuple

from core.retrieval_engine import RetrievalEngine
from ros2_interface.navigation_controller import NavigationController
from llm.llm_client import LLMClient
from utils.data_loader import DataLoader
from navigator_with_api_ros2 import VectorRetrievalSystem


class MultiStepNavigationSystem(VectorRetrievalSystem):
    """扩展的多步导航系统 - ROS2版本"""
    
    def __init__(self, config_path: str):
        """初始化系统"""
        super().__init__(config_path)
        
        # 添加多步导航相关的prompt
        self._load_multi_step_prompts()
        
        # 导航计划执行状态
        self.navigation_plan = None
        self.current_step = 0
        self.is_executing_plan = False
        self.plan_execution_thread = None
        
    def _load_multi_step_prompts(self):
        """加载多步导航相关的提示词"""
        self.multi_step_analysis_prompt = """你是一个智能机器人的多步导航规划助手。

用户会用自然语言描述一个包含多个目标点的导航任务。你需要：
1. 识别并拆解用户指令中的多个子目标
2. 保持正确的顺序（先后关系）
3. 为每个子目标生成简洁的描述

示例输入："请先带我去门口的打印机，然后去你看到过最多人的地方，最后去大堂休息区。"

请输出JSON格式的子目标序列：
[
  {"step": 1, "goal_name": "门口的打印机", "description": "printer at entrance"},
  {"step": 2, "goal_name": "看到过最多人的地方", "description": "most crowded place"},
  {"step": 3, "goal_name": "大堂休息区", "description": "lobby rest area"}
]

只输出JSON，不要输出其他内容。"""

        self.multi_step_answer_prompt = """你是一个智能机器人助手。我已经为用户规划了多步导航路径。

请生成一个友好的回复，告诉用户导航计划。输出JSON格式：
{
  "text": "明白了。我将按照以下顺序导航：首先去..., 然后去..., 最后去...",
  "plan": [导航计划数组],
  "time": "YYYY/MM/DD HH:MM:SS"
}"""

    def analyze_multi_step_intent(self, user_input: str) -> List[Dict]:
        """分析多步导航意图"""
        try:
            import openai
            response = openai.ChatCompletion.create(
                model=self.llm_client.model,
                messages=[
                    {"role": "system", "content": self.multi_step_analysis_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            result = response.choices[0].message.content.strip()
            return json.loads(result)
            
        except Exception as e:
            self.get_logger().error(f"Multi-step analysis error: {str(e)}")
            return []
    
    def retrieve_goal_positions(self, goals: List[Dict]) -> List[Dict]:
        """为每个子目标检索位置信息"""
        navigation_plan = []
        
        for goal in goals:
            # 使用fl()函数检索每个目标
            query = goal.get('description', goal['goal_name'])
            function_call = f'fl("{query}")'
            
            self.get_logger().info(f"Retrieving position for: {query}")
            retrieved_data = self.retrieval_engine.execute_retrieval(function_call)
            
            if retrieved_data:
                # 选择最相关的结果
                best_match = retrieved_data[0]
                goal_info = {
                    "step": goal['step'],
                    "goal_name": goal['goal_name'],
                    "position": best_match['position'],
                    "caption": best_match['caption'],
                    "time": best_match['time']
                }
                navigation_plan.append(goal_info)
                self.get_logger().info(f"Found position for {goal['goal_name']}: {best_match['position']}")
            else:
                self.get_logger().warn(f"No position found for: {goal['goal_name']}")
        
        return navigation_plan
    
    def process_multi_step_query(self, user_input: str) -> Dict:
        """处理多步导航查询"""
        # 分析多步意图
        goals = self.analyze_multi_step_intent(user_input)
        
        if not goals:
            # 如果不是多步查询，回退到单步处理
            return self.process_query(user_input)
        
        self.get_logger().info(f"Identified {len(goals)} navigation goals")
        
        # 检索每个目标的位置
        self.navigation_plan = self.retrieve_goal_positions(goals)
        
        if not self.navigation_plan:
            return {
                "text": "抱歉，我无法找到您提到的目标位置。",
                "time": time.strftime("%Y/%m/%d %H:%M:%S")
            }
        
        # 生成响应文本
        response_text = self._generate_multi_step_response()
        
        # 开始执行导航计划
        self.current_step = 0
        self.execute_navigation_plan()
        
        return {
            "text": response_text,
            "plan": self.navigation_plan,
            "time": time.strftime("%Y/%m/%d %H:%M:%S"),
            "total_steps": len(self.navigation_plan)
        }
    
    def _generate_multi_step_response(self) -> str:
        """生成多步导航响应文本"""
        descriptions = []
        for i, item in enumerate(self.navigation_plan):
            descriptions.append(f"{i+1}. {item['goal_name']} (位置: {item['position']})")
        
        text = "明白了。我将按照以下顺序导航：\n"
        text += "\n".join(descriptions)
        text += "\n\n现在开始执行导航计划。"
        
        return text
    
    def execute_navigation_plan(self):
        """执行多步导航计划"""
        if not self.navigation_plan or self.is_executing_plan:
            self.get_logger().warn("No navigation plan or already executing")
            return
        
        self.is_executing_plan = True
        self.plan_execution_thread = threading.Thread(target=self._execute_plan_thread)
        self.plan_execution_thread.daemon = True
        self.plan_execution_thread.start()
    
    def _execute_plan_thread(self):
        """导航计划执行线程"""
        self.get_logger().info("Starting multi-step navigation execution")
        
        while self.current_step < len(self.navigation_plan) and self.running:
            current_goal = self.navigation_plan[self.current_step]
            
            self.get_logger().info(f"Executing step {current_goal['step']}: {current_goal['goal_name']}")
            print(f"\n=== 正在导航到第 {current_goal['step']} 个目标: {current_goal['goal_name']} ===")
            
            # 发送导航目标
            success = self.nav_controller.send_goal(current_goal['position'])
            
            if not success:
                self.get_logger().error(f"Failed to send navigation goal for step {current_goal['step']}")
                print(f"导航到 {current_goal['goal_name']} 失败")
                break
            
            # 等待导航完成
            while self.nav_controller.is_navigating and self.running:
                status = self.nav_controller.get_navigation_status()
                if status in ["已完成", "已中止", "已取消"]:
                    break
                time.sleep(0.5)
            
            # 检查导航结果
            final_status = self.nav_controller.get_navigation_status()
            if final_status == "已完成":
                print(f"成功到达: {current_goal['goal_name']}")
                self.current_step += 1
                
                # 如果还有下一个目标，稍作停留
                if self.current_step < len(self.navigation_plan):
                    print("稍作停留，准备前往下一个目标...")
                    time.sleep(2.0)
            else:
                print(f"导航被中断或失败: {final_status}")
                break
        
        # 导航计划执行完成
        if self.current_step >= len(self.navigation_plan):
            print("\n=== 多步导航计划已完成 ===")
            self.get_logger().info("Multi-step navigation plan completed")
        else:
            print("\n=== 多步导航计划未完成 ===")
            self.get_logger().warn("Multi-step navigation plan incomplete")
        
        self.is_executing_plan = False
        self.navigation_plan = None
    
    def cancel_multi_step_navigation(self):
        """取消多步导航计划"""
        if self.is_executing_plan:
            self.get_logger().info("Canceling multi-step navigation plan")
            self.is_executing_plan = False
            self.nav_controller.cancel_navigation()
            print("多步导航计划已取消")
    
    def get_plan_status(self) -> Dict:
        """获取导航计划状态"""
        if not self.navigation_plan:
            return {"status": "无导航计划"}
        
        return {
            "status": "执行中" if self.is_executing_plan else "已停止",
            "current_step": self.current_step + 1,
            "total_steps": len(self.navigation_plan),
            "current_goal": self.navigation_plan[self.current_step]['goal_name'] if self.current_step < len(self.navigation_plan) else "无"
        }
    
    def _input_loop(self):
        """重写用户输入循环以支持多步导航"""
        print("=== ROS2多步导航规划系统 ===")
        print("命令说明:")
        print("- 输入查询进行单步或多步导航")
        print("- 'cancel': 取消当前导航")
        print("- 'cancel_plan': 取消整个导航计划")
        print("- 'status': 查看状态")
        print("- 'plan_status': 查看导航计划状态")
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
                    print("当前导航已取消")
                    continue
                
                if user_input.lower() == 'cancel_plan':
                    self.cancel_multi_step_navigation()
                    continue
                
                if user_input.lower() == 'status':
                    self._print_status()
                    continue
                
                if user_input.lower() == 'plan_status':
                    self._print_plan_status()
                    continue
                
                if not user_input:
                    continue
                
                # 处理查询（支持多步）
                answer = self.process_multi_step_query(user_input)
                
                # 打印结果
                print("\n" + "="*60)
                print(f"回答: {answer['text']}")
                print(f"时间: {answer['time']}")
                
                if "plan" in answer:
                    print(f"\n导航计划包含 {answer['total_steps']} 个目标:")
                    for goal in answer['plan']:
                        print(f"  {goal['step']}. {goal['goal_name']} - 位置: {goal['position']}")
                elif "position" in answer:
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
    
    def _print_plan_status(self):
        """打印导航计划状态"""
        plan_status = self.get_plan_status()
        print("\n=== 导航计划状态 ===")
        print(f"状态: {plan_status['status']}")
        if self.navigation_plan:
            print(f"进度: {plan_status['current_step']}/{plan_status['total_steps']}")
            print(f"当前目标: {plan_status['current_goal']}")
        print("="*30 + "\n")
    
    def shutdown(self):
        """关闭系统"""
        self.cancel_multi_step_navigation()
        super().shutdown()


def main(args=None):
    """主函数"""
    # 初始化ROS2
    rclpy.init(args=args)
    
    # 配置文件路径
    config_path = 'config/config.yaml'
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # 创建多步导航节点
    node = MultiStepNavigationSystem(config_path)
    
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