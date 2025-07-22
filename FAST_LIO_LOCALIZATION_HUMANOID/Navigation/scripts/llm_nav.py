#!/usr/bin/env python3
# coding: utf-8

import yaml
import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import String
from openai import OpenAI
import json
import cv2
from sensor_msgs.msg import Image
import base64
import requests
from cv_bridge import CvBridge, CvBridgeError
import os
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatusArray, GoalStatus
import actionlib
import time

api_key = "sk-iP0rxrsx3P8WevFlAe14F026043c4fC3B33b19E6DfE2Ae7e"
api_base = "https://api.gpt.ge/v1"

def load_semantic_map(file_path):
    with open(file_path, 'r') as file:
        semantic_map = yaml.safe_load(file)
    return semantic_map['semantic_info']

def chatgpt_query(system_prompt, user_prompt):    
    client = OpenAI(api_key=api_key, base_url=api_base)
    completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}]
    )
    result = json.loads(completion.choices[0].message.content.strip())
    return result


def save_to_yaml(data, filename='data.yaml'):
    with open(filename, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, allow_unicode=True, default_flow_style=False)

def load_from_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def save_to_json(data, filename='/home/nuc1003a/ARTS_test/src/hdl_localization/json/data.json'):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def read_data_json(file_path='/home/nuc1003a/ARTS_test/src/hdl_localization/json/data.json'):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

class SemanticNavigator:
    def __init__(self, semantic_info):
        rospy.init_node('semantic_navigator', anonymous=True)
        self.semantic_info = semantic_info
        self.pub_name = rospy.Publisher('/object_name', String, queue_size=10, latch=True)
        self.pub_id = rospy.Publisher('/object_id', String, queue_size=10, latch=True)
        self.pub_response = rospy.Publisher('response', String, queue_size=10, latch=True)
        self.pub_user_command = rospy.Publisher('user_commands', String, queue_size=10)
        
        self.rate = rospy.Rate(1)  # 1 Hz
        
        rospy.Subscriber("user_commands", String, self.callback)
        # rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        rospy.Subscriber("/move_base/status", GoalStatusArray, self.goal_status_callback)
        rospy.Subscriber("voiceWords", String, self.voice_callback)
        
        self.bridge = CvBridge()
        self.current_image = None
        self.image_save_path = '/home/nuc1003a/ARTS_test/src/hdl_localization/images'
        self.image_saved = False
        self.previous_goal_status = None
        self.goal_reached_time = None
        

    def callback(self, msg):
        user_input = msg.data
        rospy.loginfo(f"Received user command: {user_input}")
        system_prompt = (
            "现在想象一下，你是一个家庭服务机器人, "
            "你现在有能力根据用户的语言命令和包含环境语义的 yaml 文件，输出目标对象的名称和 id。."
            + str(self.semantic_info) + "是环境中包含的语义信息, 它包含:"
            "- id = 1----物体的id"
            "- name = 'bed'----物体的名字"
            "- position = {x: 3, y: 4}----物体在地图中的位置"
            "- size = {x: 1, y: 2}----物体的二维尺寸"
            "请理解所提供的语义信息, 并根据用户提供的指令输出目标对象的名称和 ID(输出 JSON 数据结构)."
            """例如, 如果用户的指令是: "去最大的桌子那里", 则需要根据所提供的信息找到尺寸最大的桌子, 并按以下格式输出:
            {
                "name": "table",
                "id": "1",
                "response": "好的，我正在前往最大的桌子那里。"
            }
            请根据上述要求生成一个JSON格式的响应, 并只回复JSON数据本身而不回复任何其他内容。
            """
        )
        user_prompt = user_input
        try:
            target_name = chatgpt_query(system_prompt, user_prompt)

            save_to_json(target_name)
            data = read_data_json()
            name = data['name']
            id = data['id']
            response = data['response']
            print("WALLE:\n", response)
            self.pub_name.publish(name)
            self.pub_id.publish(str(id))
            self.pub_response.publish(response)
        except Exception as e:
            rospy.logerr(f"Error processing user command: {e}")
            
            
        # self.prompt_user_input()
        
        
    def voice_callback(self, msg):
        voice_input = msg.data
        rospy.loginfo(f"Received voice input: {voice_input}")
        self.pub_user_command.publish(voice_input)
    
    def goal_status_callback(self, msg):
        
        for status in msg.status_list:
            # rospy.loginfo(f"Goal ID: {status.goal_id.id}, Status: {status.status}, Text: {status.text}")
            if status.status in [GoalStatus.PENDING, GoalStatus.ACTIVE] and self.previous_goal_status != status.status:
                # self.image_saved = False  # 新目标到达，重置图像保存标志
                self.goal_reached_time = None
                
            # 检查状态是否为 "Goal reached." 并且尚未保存图像
            if status.status == GoalStatus.SUCCEEDED:
                if self.goal_reached_time is None:
                    self.goal_reached_time = time.time()
                    
                # if (time.time() - self.goal_reached_time) > 1.0 and not self.image_saved:
                #     self.save_image_if_available()
                #     self.image_saved = True  # 设置标志，表示图像已保存
                
            self.previous_goal_status = status.status
    
    def image_callback(self, msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Failed to convert image: {e}")
                
    def save_image_if_available(self):
        if self.current_image is not None:
            try:
                image_filename = os.path.join(self.image_save_path, "current_image.jpg")
                cv2.imwrite(image_filename, self.current_image)
                rospy.loginfo(f"Image saved to {image_filename}")
            except Exception as e:
                rospy.logerr(f"Failed to save image: {e}")
        else:
            rospy.logwarn("No image available to save.")

    def prompt_user_input(self):
        user_input = input("请输入您的指令: \n")
        if user_input:
            self.pub_user_command.publish(user_input)
        else:
            rospy.logwarn("No input received, awaiting next command.")
            
    def run(self):
        # self.pub_user_command = rospy.Publisher('user_commands', String, queue_size=10)
        rospy.loginfo("WALLE 正在等待指令...")
        # self.prompt_user_input()
        rospy.spin()

def main():
    semantic_info = load_semantic_map('/home/nuc1003a/ARTS_test/src/hdl_localization/config/map/semantic_map.yaml')
    navigator = SemanticNavigator(semantic_info)
    navigator.run()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
