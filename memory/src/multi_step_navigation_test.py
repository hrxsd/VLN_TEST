#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多步导航规划测试版本
使用真实的LLM和检索引擎，但不依赖ROS环境
"""

import os
import sys
import json
import yaml
from typing import Dict, List, Tuple
from datetime import datetime

# 导入原有系统的组件
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.retrieval_engine import RetrievalEngine
from llm.llm_client import LLMClient
from utils.data_loader import DataLoader


class MultiStepNavigationTester:
    """多步导航测试系统 - 不依赖ROS"""
    
    def __init__(self, config_path: str):
        """初始化测试系统"""
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print("初始化多步导航测试系统...")
        
        # 初始化各个组件（与ROS版本相同）
        self._init_components()
        
        # 添加多步导航相关的prompt
        self._load_multi_step_prompts()
        
        # 导航计划
        self.navigation_plan = None
        self.current_step = 0
        
    def _init_components(self):
        """初始化系统组件"""
        # 数据加载器
        self.data_loader = DataLoader(self.config['database']['path'])
        self.data = self.data_loader.load_data()
        print(f"加载了 {len(self.data)} 条记忆数据")
        
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
        """分析多步导航意图 - 使用真实的LLM"""
        try:
            import openai
            openai.api_key = self.config['openai']['api_key']
            openai.api_base = self.config['openai']['api_base']
            
            response = openai.ChatCompletion.create(
                model=self.config['openai']['model'],
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
            print(f"LLM多步分析错误: {e}")
            return []
    
    def retrieve_goal_positions(self, goals: List[Dict]) -> List[Dict]:
        """为每个子目标检索位置信息 - 使用真实的检索引擎"""
        navigation_plan = []
        
        for goal in goals:
            # 使用fl()函数检索每个目标
            query = goal.get('description', goal['goal_name'])
            function_call = f'fl("{query}")'
            
            print(f"检索步骤 {goal['step']}: {query}")
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
                print(f"  ✓ 找到位置: {best_match['position']} - {best_match['caption'][:30]}...")
            else:
                print(f"  ✗ 未找到 {goal['goal_name']} 的位置")
                
        return navigation_plan
    
    def process_multi_step_query(self, user_input: str) -> Dict:
        """处理多步导航查询"""
        print(f"\n处理多步导航查询: {user_input}")
        print("-" * 60)
        
        # 步骤1: 分析并拆解多步导航意图
        goals = self.analyze_multi_step_intent(user_input)
        if not goals:
            return {
                "text": "抱歉，我无法理解您的多步导航请求。",
                "time": datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            }
        
        print(f"\n识别到 {len(goals)} 个导航目标:")
        for goal in goals:
            print(f"  步骤 {goal['step']}: {goal['goal_name']}")
        
        # 步骤2: 为每个目标检索位置
        print("\n开始检索位置信息...")
        navigation_plan = self.retrieve_goal_positions(goals)
        
        if not navigation_plan:
            return {
                "text": "抱歉，我无法找到您提到的目标位置。",
                "time": datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            }
        
        # 步骤3: 生成回复
        plan_text = self._generate_plan_description(navigation_plan)
        
        # 保存导航计划
        self.navigation_plan = navigation_plan
        self.current_step = 0
        
        return {
            "text": plan_text,
            "plan": navigation_plan,
            "time": datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            "total_steps": len(navigation_plan)
        }
    
    def process_query(self, user_input: str) -> Dict:
        """处理用户查询 - 兼容单步和多步"""
        # 检查是否是多步导航请求
        if any(keyword in user_input for keyword in ['先', '然后', '接着', '最后', '再', '依次']):
            return self.process_multi_step_query(user_input)
        else:
            # 单步查询处理
            print(f"处理单步查询: {user_input}")
            
            # 步骤1: 分析用户意图
            function_call = self.llm_client.analyze_intent(user_input)
            print(f"意图分析结果: {function_call}")
            
            # 步骤2: 执行检索
            retrieved_data = self.retrieval_engine.execute_retrieval(function_call)
            print(f"检索到 {len(retrieved_data)} 条结果")
            
            # 步骤3: 生成答案
            answer = self.llm_client.generate_answer(user_input, retrieved_data)
            
            # 注意：测试版本不发送导航目标，只返回结果
            if "position" in answer:
                answer["navigation_status"] = "planned_only"
                answer["text"] += " [测试模式：不执行实际导航]"
            
            return answer
    
    def _generate_plan_description(self, plan: List[Dict]) -> str:
        """生成导航计划的文字描述"""
        descriptions = []
        for item in plan:
            descriptions.append(f"{item['step']}. {item['goal_name']} (位置: {item['position']})")
        
        text = "明白了。我将按照以下顺序导航：\n"
        text += "\n".join(descriptions)
        text += "\n\n[测试模式：导航计划已生成但不会执行]"
        
        return text
    
    def simulate_navigation_execution(self):
        """模拟导航执行过程"""
        if not self.navigation_plan:
            print("没有导航计划可以执行")
            return
        
        print("\n=== 模拟导航执行 ===")
        for i, goal in enumerate(self.navigation_plan):
            print(f"\n步骤 {goal['step']}: 导航到 {goal['goal_name']}")
            print(f"  目标位置: {goal['position']}")
            print(f"  描述: {goal['caption']}")
            print(f"  [模拟] 发送目标到 move_base...")
            print(f"  [模拟] 等待导航完成...")
            print(f"  ✓ 到达目标点")
            
            if i < len(self.navigation_plan) - 1:
                print(f"  准备前往下一个目标...")
        
        print("\n=== 导航计划执行完成 ===")
    
    def export_navigation_plan(self, filename: str = None):
        """导出导航计划为JSON"""
        if not self.navigation_plan:
            print("没有导航计划可以导出")
            return
        
        if not filename:
            filename = f"nav_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model": self.config['openai']['model'],
                "database": self.config['database']['path']
            },
            "plan": self.navigation_plan,
            "total_steps": len(self.navigation_plan),
            "total_distance": self._calculate_total_distance()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"导航计划已导出到: {filename}")
    
    def _calculate_total_distance(self) -> float:
        """计算总导航距离"""
        if not self.navigation_plan:
            return 0.0
        
        total_distance = 0.0
        last_pos = [0.0, 0.0]  # 假设从原点开始
        
        for goal in self.navigation_plan:
            pos = goal['position'][:2]  # 只使用x,y坐标
            distance = ((pos[0] - last_pos[0])**2 + (pos[1] - last_pos[1])**2)**0.5
            total_distance += distance
            last_pos = pos
        
        return round(total_distance, 2)
    
    def run(self):
        """运行测试系统主循环"""
        print("\n=== 多步导航规划测试系统 ===")
        print("使用真实LLM和检索引擎，但不执行ROS导航")
        print("\n命令说明:")
        print("- 输入查询进行测试（支持单步和多步）")
        print("- 'simulate': 模拟导航执行")
        print("- 'export': 导出导航计划")
        print("- 'plan': 查看当前导航计划")
        print("- 'quit/exit': 退出系统")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("请输入您的问题: ").strip()
                
                # 处理特殊命令
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("系统关闭中...")
                    break
                
                if user_input.lower() == 'simulate':
                    self.simulate_navigation_execution()
                    continue
                
                if user_input.lower() == 'export':
                    self.export_navigation_plan()
                    continue
                
                if user_input.lower() == 'plan':
                    if self.navigation_plan:
                        print("\n--- 当前导航计划 ---")
                        print(json.dumps(self.navigation_plan, ensure_ascii=False, indent=2))
                    else:
                        print("当前没有导航计划")
                    continue
                
                if not user_input:
                    continue
                
                # 处理查询
                result = self.process_query(user_input)
                
                # 显示结果
                print("\n=== 查询结果 ===")
                print(json.dumps(result, ensure_ascii=False, indent=2))
                print("="*60 + "\n")
                
            except KeyboardInterrupt:
                print("\n系统关闭中...")
                break
            except Exception as e:
                print(f"错误: {e}")
                import traceback
                traceback.print_exc()


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
        # 创建并运行测试系统
        tester = MultiStepNavigationTester(config_path)
        tester.run()
    except Exception as e:
        print(f"系统启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()