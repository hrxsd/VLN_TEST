#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多步导航规划扩展模块 - ROS集成版本
支持解析复合语言指令，生成多步导航目标序列
"""

import os
import sys
import json
import yaml
import rospy
import threading
import time
from typing import Dict, List, Tuple

from core.retrieval_engine import RetrievalEngine
from ros_interface.navigation_controller import NavigationController
from llm.llm_client import LLMClient
from utils.data_loader import DataLoader
from navigator_with_api import VectorRetrievalSystem


class MultiStepNavigationSystem(VectorRetrievalSystem):
    """扩展的多步导航系统"""
    
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
            rospy.logerr(f"Multi-step analysis error: {e}")
            return []
    
    def retrieve_goal_positions(self, goals: List[Dict]) -> List[Dict]:
        """为每个子目标检索位置信息"""
        navigation_plan = []
        
        for goal in goals:
            # 使用fl()函数检索每个目标
            query = goal.get('description', goal['goal_name'])
            function_call = f'fl("{query}")'
            
            rospy.loginfo(f"Retrieving position for step {goal['step']}: {query}")
            retrieved_data = self.retrieval_engine.execute_retrieval(function_call)
            
            if retrieved_data:
                # 使用最相关的结果
                best_match = retrieved_data[0]
                navigation_plan.append({
                    "step": goal['step'],
                    "goal_name": goal['goal_name'],
                    "position": best_match['position'],
                    "caption": best_match['caption'],
                    "similarity": best_match.get('similarity', 0)
                })
                rospy.loginfo(f"Found position for {goal['goal_name']}: {best_match['position']}")
            else:
                rospy.logwarn(f"No position found for {goal['goal_name']}")
                
        return navigation_plan
    
    def process_multi_step_query(self, user_input: str) -> Dict:
        """处理多步导航查询"""
        rospy.loginfo(f"Processing multi-step query: {user_input}")
        
        # 步骤1: 分析并拆解多步导航意图
        goals = self.analyze_multi_step_intent(user_input)
        if not goals:
            return {
                "text": "抱歉，我无法理解您的多步导航请求。",
                "time": rospy.Time.now().to_sec()
            }
        
        rospy.loginfo(f"Identified {len(goals)} navigation goals")
        
        # 步骤2: 为每个目标检索位置
        navigation_plan = self.retrieve_goal_positions(goals)
        
        if not navigation_plan:
            return {
                "text": "抱歉，我无法找到您提到的目标位置。",
                "time": rospy.Time.now().to_sec()
            }
        
        # 步骤3: 生成回复
        plan_text = self._generate_plan_description(navigation_plan)
        
        # 保存导航计划
        self.navigation_plan = navigation_plan
        self.current_step = 0
        
        return {
            "text": plan_text,
            "plan": navigation_plan,
            "time": rospy.Time.now().to_sec(),
            "total_steps": len(navigation_plan)
        }
    
    def _generate_plan_description(self, plan: List[Dict]) -> str:
        """生成导航计划的文字描述"""
        descriptions = []
        for item in plan:
            descriptions.append(f"{item['step']}. {item['goal_name']} (位置: {item['position']})")
        
        text = "明白了。我将按照以下顺序导航：\n"
        text += "\n".join(descriptions)
        text += "\n\n现在开始执行导航计划。"
        
        return text
    
    def execute_navigation_plan(self):
        """执行多步导航计划"""
        if not self.navigation_plan or self.is_executing_plan:
            rospy.logwarn("No navigation plan or already executing")
            return
        
        self.is_executing_plan = True
        self.plan_execution_thread = threading.Thread(target=self._execute_plan_thread)
        self.plan_execution_thread.start()
    
    def _execute_plan_thread(self):
        """导航计划执行线程"""
        rospy.loginfo("Starting multi-step navigation execution")
        
        while self.current_step < len(self.navigation_plan) and not rospy.is_shutdown():
            current_goal = self.navigation_plan[self.current_step]
            
            rospy.loginfo(f"Executing step {current_goal['step']}: {current_goal['goal_name']}")
            print(f"\n=== 正在导航到第 {current_goal['step']} 个目标: {current_goal['goal_name']} ===")
            
            # 发送导航目标
            success = self.nav_controller.send_goal(current_goal['position'])
            
            if not success:
                rospy.logerr(f"Failed to send navigation goal for step {current_goal['step']}")
                print(f"导航到 {current_goal['goal_name']} 失败")
                break
            
            # 等待导航完成
            while self.nav_controller.is_navigating and not rospy.is_shutdown():
                status = self.nav_controller.get_navigation_status()
                if status in ["已完成", "已中止", "已拒绝"]:
                    break
                rospy.sleep(0.5)
            
            if self.nav_controller.get_navigation_status() == "已完成":
                print(f"✓ 成功到达: {current_goal['goal_name']}")
                self.current_step += 1
                
                # 如果不是最后一步，稍作停留
                if self.current_step < len(self.navigation_plan):
                    print("稍作停留，准备前往下一个目标...")
                    rospy.sleep(2.0)
            else:
                print(f"✗ 未能到达: {current_goal['goal_name']}")
                break
        
        if self.current_step >= len(self.navigation_plan):
            print("\n=== 多步导航计划已完成！ ===")
        else:
            print("\n=== 多步导航计划中断 ===")
        
        self.is_executing_plan = False
        self.navigation_plan = None
        self.current_step = 0
    
    def cancel_navigation_plan(self):
        """取消多步导航计划"""
        if self.is_executing_plan:
            rospy.loginfo("Cancelling multi-step navigation plan")
            self.nav_controller.cancel_navigation()
            self.is_executing_plan = False
            self.navigation_plan = None
            self.current_step = 0
            print("多步导航计划已取消")
    
    def get_plan_status(self) -> Dict:
        """获取导航计划执行状态"""
        if not self.navigation_plan:
            return {"status": "无导航计划"}
        
        return {
            "status": "执行中" if self.is_executing_plan else "已规划",
            "current_step": self.current_step + 1,
            "total_steps": len(self.navigation_plan),
            "current_goal": self.navigation_plan[self.current_step]['goal_name'] if self.current_step < len(self.navigation_plan) else "无",
            "nav_status": self.nav_controller.get_navigation_status()
        }
    
    def run(self):
        """运行主循环"""
        print("=== ROS多步导航规划系统 ===")
        print("命令说明:")
        print("- 输入包含多个目标的导航请求")
        print("- 'execute': 执行导航计划")
        print("- 'cancel': 取消导航")
        print("- 'status': 查看状态")
        print("- 'plan': 查看当前导航计划")
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
                    self.cancel_navigation_plan()
                    continue
                
                if user_input.lower() == 'execute':
                    if self.navigation_plan and not self.is_executing_plan:
                        self.execute_navigation_plan()
                    else:
                        print("没有可执行的导航计划或计划正在执行中")
                    continue
                
                if user_input.lower() == 'status':
                    status = self.get_plan_status()
                    print("\n--- 导航计划状态 ---")
                    for key, value in status.items():
                        print(f"{key}: {value}")
                    self._print_status()
                    continue
                
                if user_input.lower() == 'plan':
                    if self.navigation_plan:
                        print("\n--- 当前导航计划 ---")
                        for item in self.navigation_plan:
                            print(f"步骤 {item['step']}: {item['goal_name']} -> {item['position']}")
                    else:
                        print("当前没有导航计划")
                    continue
                
                if not user_input:
                    continue
                
                # 检查是否是多步导航请求
                if any(keyword in user_input for keyword in ['先', '然后', '接着', '最后', '再']):
                    # 处理多步导航查询
                    result = self.process_multi_step_query(user_input)
                    
                    # 显示结果
                    print("\n=== 多步导航规划结果 ===")
                    print(result['text'])
                    if 'plan' in result:
                        print("\n是否立即执行导航计划？(yes/no)")
                        if input().lower() in ['yes', 'y', '是']:
                            self.execute_navigation_plan()
                else:
                    # 处理单步查询
                    result = self.process_query(user_input)
                    print("\n=== 查询结果 ===")
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                
                print("="*40 + "\n")
                
            except KeyboardInterrupt:
                print("\n系统关闭中...")
                self.shutdown()
                break
            except Exception as e:
                rospy.logerr(f"Error: {e}")


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
        # 创建并运行多步导航系统
        system = MultiStepNavigationSystem(config_path)
        system.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()