#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import math
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Any
from transformers import AutoModel, AutoTokenizer
from sentence_transformers.util import cos_sim
import openai
import re
import threading
import time

# ROS imports
import rospy
import actionlib
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from std_msgs.msg import Header
import tf
from tf.transformations import quaternion_from_euler

class VectorDatabaseRetrievalROS:
    def __init__(self, json_path: str, api_key: str, api_base: str = "https://api.gpt.ge/v1"):
        """
        初始化向量数据库检索系统（ROS版本）
        
        Args:
            json_path: 向量数据库JSON文件路径
            api_key: OpenAI API密钥
            api_base: API基础URL
        """
        # 初始化原有功能
        self.json_path = json_path
        self.data = self._load_data()
        
        # 初始化OpenAI客户端
        openai.api_key = api_key
        openai.api_base = api_base
        
        # 初始化文本嵌入模型
        self.model_id = 'mixedbread-ai/mxbai-embed-large-v1'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
        
        # 如果有GPU可用，使用GPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # 初始化ROS节点
        rospy.init_node('vector_retrieval_ros_node', anonymous=True)
        rospy.loginfo("Vector Retrieval ROS Node initialized")
        
        # 初始化move_base客户端
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
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
        self.current_goal_handle = None
        
        # 参数配置
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.robot_frame = rospy.get_param('~robot_frame', 'base_link')
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.5)
    
    def _load_data(self) -> List[Dict]:
        """加载向量数据库数据"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def transform_query(self, query: str) -> str:
        """转换查询语句以适应嵌入模型"""
        return f"Represent this sentence for searching relevant passages: {query}"
    
    def pooling(self, outputs: torch.Tensor, inputs: Dict, strategy: str = 'cls') -> np.ndarray:
        """池化操作"""
        if strategy == 'cls':
            outputs = outputs[:, 0]
        elif strategy == 'mean':
            outputs = torch.sum(
                outputs * inputs["attention_mask"][:, :, None], dim=1
            ) / torch.sum(inputs["attention_mask"], dim=1, keepdim=True)
        else:
            raise NotImplementedError
        return outputs.detach().cpu().numpy()
    
    def fl(self, object_query: str) -> List[Dict]:
        """
        文本检索函数 (Text Lookup)
        
        Args:
            object_query: 要检索的关键词或短语
            
        Returns:
            List[Dict]: Top-5最相关的记忆片段
        """
        query_prompted = self.transform_query(object_query)
        
        # 编码查询
        inputs = self.tokenizer(query_prompted, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state
        query_embedding = self.pooling(outputs, inputs, 'cls')
        query_tensor = torch.from_numpy(query_embedding.astype(np.float32))
        
        # 计算相似度
        results = []
        for idx, item in enumerate(self.data):
            embedding = np.array(item['embedding'], dtype=np.float32)
            embedding_tensor = torch.from_numpy(embedding).unsqueeze(0)
            sim = cos_sim(query_tensor, embedding_tensor)
            
            results.append({
                'index': idx,
                'caption': item['caption'],
                'time': item['time'],
                'position': item['position'],
                'similarity': sim.item()
            })
        
        # 排序并返回Top-5
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:5]
    
    def fp(self, position: Tuple[float, float, float]) -> List[Dict]:
        """
        空间位置检索函数 (Position Lookup)
        
        Args:
            position: 目标位置坐标 (x, y, w)
            
        Returns:
            List[Dict]: Top-5最近的记忆片段
        """
        x_target, y_target, w_target = position
        
        results = []
        for idx, item in enumerate(self.data):
            x, y, w = item['position']
            # 计算欧几里得距离（只考虑x,y坐标）
            distance = math.sqrt((x - x_target) ** 2 + (y - y_target) ** 2)
            
            results.append({
                'index': idx,
                'caption': item['caption'],
                'time': item['time'],
                'position': item['position'],
                'distance': distance
            })
        
        # 按距离排序并返回Top-5
        results.sort(key=lambda x: x['distance'])
        return results[:5]
    
    def ft(self, time_query: str) -> List[Dict]:
        """
        时间检索函数 (Time Lookup)
        
        Args:
            time_query: 目标时间字符串 "YYYY/MM/DD HH:MM:SS"
            
        Returns:
            List[Dict]: Top-5时间最接近的记忆片段
        """
        target_time = datetime.strptime(time_query, "%Y/%m/%d %H:%M:%S")
        
        results = []
        for idx, item in enumerate(self.data):
            item_time = datetime.strptime(item['time'], "%Y/%m/%d %H:%M:%S")
            time_diff = abs((target_time - item_time).total_seconds())
            
            results.append({
                'index': idx,
                'caption': item['caption'],
                'time': item['time'],
                'position': item['position'],
                'time_diff': time_diff
            })
        
        # 按时间差排序并返回Top-5
        results.sort(key=lambda x: x['time_diff'])
        return results[:5]
    
    def analyze_user_intent(self, user_input: str) -> str:
        """
        使用LLM分析用户意图并构造检索请求
        
        Args:
            user_input: 用户输入的自然语言指令
            
        Returns:
            str: 包含函数调用的响应
        """
        system_prompt = """你是一个智能机器人的记忆检索助手。用户会用自然语言向你提问，你需要分析用户的意图，并构造相应的检索函数调用。

你有三个检索函数可以使用：
1. fl(object) - 文本检索：基于关键词或描述检索语义相关的记忆
2. fp((x,y,w)) - 位置检索：检索指定位置附近的记忆
3. ft("YYYY/MM/DD HH:MM:SS") - 时间检索：检索指定时间点附近的记忆

分析用户问题，确定需要调用哪个或哪些函数，然后输出函数调用。

示例：
用户："我在哪里见过自动售货机？"
分析：用户想找与自动售货机相关的记忆
输出：fl("vending machine")

用户："昨天下午3点我在做什么？"
分析：用户想查询特定时间的记忆
输出：ft("2025/02/24 15:00:00")

用户："带我去咖啡厅"
分析：用户想要导航到咖啡厅，需要先找到咖啡厅的位置
输出：fl("coffee shop")

请只输出函数调用，不要输出其他内容。"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.1,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            rospy.logerr(f"LLM调用错误: {e}")
            return f"fl(\"{user_input}\")"  # 默认使用文本检索
    
    def execute_function_call(self, function_call: str) -> List[Dict]:
        """
        执行函数调用
        
        Args:
            function_call: 函数调用字符串
            
        Returns:
            List[Dict]: 检索结果
        """
        try:
            # 解析函数调用
            if function_call.startswith("fl("):
                # 提取fl函数的参数
                match = re.search(r'fl\(["\']([^"\']+)["\']\)', function_call)
                if match:
                    query = match.group(1)
                    return self.fl(query)
            
            elif function_call.startswith("fp("):
                # 提取fp函数的参数
                match = re.search(r'fp\(\(([^)]+)\)\)', function_call)
                if match:
                    coords = match.group(1).split(',')
                    x, y, w = map(float, coords)
                    return self.fp((x, y, w))
            
            elif function_call.startswith("ft("):
                # 提取ft函数的参数
                match = re.search(r'ft\(["\']([^"\']+)["\']\)', function_call)
                if match:
                    time_str = match.group(1)
                    return self.ft(time_str)
            
            rospy.logwarn(f"无法解析函数调用: {function_call}")
            return []
            
        except Exception as e:
            rospy.logerr(f"执行函数调用错误: {e}")
            return []
    
    def generate_final_answer(self, user_input: str, retrieved_data: List[Dict]) -> Dict:
        """
        基于检索到的数据生成最终答案
        
        Args:
            user_input: 用户原始输入
            retrieved_data: 检索到的记忆片段
            
        Returns:
            Dict: 最终答案JSON对象
        """
        # 构造上下文
        context = "检索到的相关记忆片段：\n"
        for i, item in enumerate(retrieved_data):
            context += f"记忆{i+1}:\n"
            context += f"- 描述: {item['caption']}\n"
            context += f"- 时间: {item['time']}\n"
            context += f"- 位置: {item['position']}\n\n"
        
        system_prompt = """你是一个智能机器人助手。用户向你提问，我已经从记忆数据库中检索了相关信息。请根据这些信息回答用户的问题。

重要要求：
1. 如果用户的问题是关于导航（如"带我去..."、"怎么到..."等），你需要在答案中包含position字段
2. 如果不是导航问题，只包含text和time字段
3. 回答要自然、有用，基于检索到的记忆信息

请严格按照以下JSON格式输出：

导航问题格式：
{
  "text": "具体的回答内容",
  "position": [x, y, w],
  "time": "YYYY/MM/DD HH:MM:SS"
}

非导航问题格式：
{
  "text": "具体的回答内容",
  "time": "YYYY/MM/DD HH:MM:SS"
}

举例：
用户提问A：快递盒子在哪？(非导航问题)
系统回答A(非导航问题格式)：
{
  "text": "具体的回答内容",
  "time": "YYYY/MM/DD HH:MM:SS"
}
用户提问B：带我去快递盒子那里。(导航问题)
系统回答B(导航问题格式)：
{
  "text": "具体的回答内容",
  "position": [12.34, 56.78, 1.0],
  "time": "YYYY/MM/DD HH:MM:SS"
}

只输出JSON，不要输出其他内容。"""

        user_prompt = f"用户问题：{user_input}\n\n{context}"
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            answer_text = response.choices[0].message.content.strip()
            
            # 尝试解析JSON
            try:
                return json.loads(answer_text)
            except json.JSONDecodeError:
                # 如果解析失败，返回默认格式
                return {
                    "text": answer_text,
                    "time": datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                }
                
        except Exception as e:
            rospy.logerr(f"生成最终答案错误: {e}")
            return {
                "text": "抱歉，我无法处理您的请求。",
                "time": datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            }
    
    def send_navigation_goal(self, position: List[float]) -> bool:
        """
        发送导航目标到move_base
        
        Args:
            position: [x, y, w] 坐标，其中w是朝向（弧度）
            
        Returns:
            bool: 是否成功发送目标
        """
        try:
            # 取消之前的导航目标
            if self.is_navigating:
                rospy.loginfo("Canceling previous navigation goal...")
                self.move_base_client.cancel_goal()
                self.is_navigating = False
            
            # 创建导航目标
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
            
            rospy.loginfo(f"Sending navigation goal to position: x={position[0]:.2f}, y={position[1]:.2f}, yaw={position[2]:.2f}")
            
            # 发送目标
            self.move_base_client.send_goal(goal)
            self.is_navigating = True
            
            # 启动监控线程
            self.navigation_thread = threading.Thread(target=self._monitor_navigation)
            self.navigation_thread.start()
            
            return True
            
        except Exception as e:
            rospy.logerr(f"Failed to send navigation goal: {e}")
            return False
    
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
    
    def cancel_navigation(self):
        """取消当前导航"""
        if self.is_navigating:
            rospy.loginfo("Canceling navigation...")
            self.move_base_client.cancel_goal()
            self.is_navigating = False
    
    def get_robot_position(self) -> Tuple[float, float, float]:
        """
        获取机器人当前位置
        
        Returns:
            Tuple[float, float, float]: (x, y, yaw) 或 None
        """
        try:
            (trans, rot) = self.tf_listener.lookupTransform(
                self.map_frame, self.robot_frame, rospy.Time(0))
            
            # 转换四元数到欧拉角
            euler = tf.transformations.euler_from_quaternion(rot)
            yaw = euler[2]
            
            return (trans[0], trans[1], yaw)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"Failed to get robot position: {e}")
            return None
    
    def query(self, user_input: str) -> Dict:
        """
        主查询接口（增强版，支持ROS导航）
        
        Args:
            user_input: 用户自然语言输入
            
        Returns:
            Dict: 最终答案JSON对象
        """
        rospy.loginfo(f"用户输入: {user_input}")
        
        # 第1步：分析用户意图，构造检索请求
        function_call = self.analyze_user_intent(user_input)
        rospy.loginfo(f"构造的函数调用: {function_call}")
        
        # 第2步：执行检索
        retrieved_data = self.execute_function_call(function_call)
        rospy.loginfo(f"检索到 {len(retrieved_data)} 条相关记忆")
        
        # 第3步：生成最终答案
        final_answer = self.generate_final_answer(user_input, retrieved_data)
        
        # 第4步：如果答案包含position字段，发送导航目标
        if "position" in final_answer:
            position = final_answer["position"]
            rospy.loginfo(f"Detected navigation request to position: {position}")
            
            # 发送导航目标
            if self.send_navigation_goal(position):
                # 在回答中添加导航状态
                final_answer["navigation_status"] = "started"
                final_answer["text"] += " 我正在导航到目标位置。"
            else:
                final_answer["navigation_status"] = "failed"
                final_answer["text"] += " 导航启动失败，请检查move_base状态。"
        
        return final_answer
    
    def shutdown(self):
        """关闭系统"""
        if self.is_navigating:
            self.cancel_navigation()
        rospy.signal_shutdown("User requested shutdown")


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: rosrun your_package vector_retrieval_ros.py <数据库JSON文件路径>")
        return
    
    json_path = sys.argv[1]
    api_key = "sk-iP0rxrsx3P8WevFlAe14F026043c4fC3B33b19E6DfE2Ae7e"
    api_base = "https://api.gpt.ge/v1"
    
    try:
        # 初始化检索系统
        retrieval_system = VectorDatabaseRetrievalROS(json_path, api_key, api_base)
        
        print("=== ROS向量数据库检索系统启动 ===")
        print("输入 'quit' 或 'exit' 退出系统")
        print("输入 'cancel' 取消当前导航")
        print("输入 'status' 查看导航状态\n")
        
        while not rospy.is_shutdown():
            try:
                user_input = input("请输入您的问题: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("系统正在关闭...")
                    retrieval_system.shutdown()
                    break
                
                if user_input.lower() == 'cancel':
                    retrieval_system.cancel_navigation()
                    print("导航已取消")
                    continue
                
                if user_input.lower() == 'status':
                    if retrieval_system.is_navigating:
                        state = retrieval_system.move_base_client.get_state()
                        print(f"导航状态: {state}")
                    else:
                        print("当前没有进行导航")
                    
                    # 显示机器人当前位置
                    pos = retrieval_system.get_robot_position()
                    if pos:
                        print(f"机器人位置: x={pos[0]:.2f}, y={pos[1]:.2f}, yaw={pos[2]:.2f}")
                    continue
                
                if not user_input:
                    continue
                
                # 执行查询
                result = retrieval_system.query(user_input)
                
                # 输出结果
                print("\n=== 查询结果 ===")
                print(json.dumps(result, ensure_ascii=False, indent=2))
                print("="*50 + "\n")
                
            except KeyboardInterrupt:
                print("\n系统正在关闭...")
                retrieval_system.shutdown()
                break
            except Exception as e:
                rospy.logerr(f"发生错误: {e}")
                
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()