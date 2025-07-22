#!/usr/bin/env python3
# coding: utf-8

import base64
import urllib
import requests
import json
import time
import cv2
from openai import OpenAI
import rospy
from actionlib_msgs.msg import GoalStatusArray
import threading
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

API_KEY = "w1tpR06Fq5yhnFGCYFl88eA2"
SECRET_KEY = "fWsqb2A0GO1ngJaB5xSFqXuw6tCvz80I"
IMAGE_PATH = "/home/nuc1003a/ARTS_TEST/src/hdl_localization/images/current_image.jpg"
QUESTION = "你是一台机器人这是你当前看到的内容请结合user prompt和图像内容进行十分简洁的回答不需要描述情绪等抽象内容以第一视角回答"
TASK_ID_FILE = "task_id.txt"
RETRY_INTERVAL = 10  # 秒
CHECK_INTERVAL = 0.1  # 检查 GoalStatus 间隔

GPT4_API_KEY = "sk-iP0rxrsx3P8WevFlAe14F026043c4fC3B33b19E6DfE2Ae7e"
GPT4_API_BASE = "https://api.gpt.ge/v1"

# 初始化全局变量
goal_status_succeeded = False
goal_status_time = None
processed_goal = False
user_command = ""
bridge = CvBridge()
last_image = None

def get_access_token():
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

def get_file_content_as_base64(path, urlencoded=False):
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content

def request_image_understanding():
    url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/image-understanding/request?access_token=" + get_access_token()
    payload = json.dumps({
        "image": get_file_content_as_base64(IMAGE_PATH, False),
        "question": QUESTION,
        "output_CHN": True
    })
    headers = {'Content-Type': 'application/json'}
    response = requests.request("POST", url, headers=headers, data=payload)
    response_data = response.json()
    task_id = response_data.get("result", {}).get("task_id")
    if task_id:
        with open(TASK_ID_FILE, "w") as file:
            file.write(task_id)
    else:
        print("Error: task_id not found in the response")
    return task_id

def get_result(task_id):
    url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/image-understanding/get-result?access_token=" + get_access_token()
    payload = json.dumps({"task_id": task_id})
    headers = {'Content-Type': 'application/json'}
    while True:
        response = requests.request("POST", url, headers=headers, data=payload)
        response_data = response.json()
        ret_code = response_data.get("result", {}).get("ret_code")
        if ret_code == 0:
            description_to_llm = response_data.get("result", {}).get("description_to_llm")
            print("处理成功: ", description_to_llm)
            return description_to_llm
        elif ret_code == 1:
            print("处理中，请稍候...")
            time.sleep(RETRY_INTERVAL)
        else:
            print("发生错误: ", response_data)
            break

def get_chatgpt_response(description_to_llm):    
    client = OpenAI(api_key=GPT4_API_KEY, base_url=GPT4_API_BASE)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": description_to_llm},
            {"role": "user", "content": user_command}
        ],
        max_tokens=400
    )
    return response.choices[0].message.content.strip()

def image_processing_workflow(pub):
    task_id = request_image_understanding()
    if task_id:
        description_to_llm = get_result(task_id)
        if description_to_llm:
            print("上传到ChatGPT中...")
            gpt_response = get_chatgpt_response(description_to_llm)
            print("WALLE:", gpt_response)
            pub.publish(gpt_response)
        else:
            print("未能获取description_to_llm")
    else:
        print("请求图像理解失败。")

def goal_status_callback(msg):
    global goal_status_succeeded, processed_goal

    if not msg.status_list:
        rospy.loginfo("No status in status_list")
        return

    for status_item in msg.status_list:
        rospy.loginfo(f"Received status: {status_item.status}")
        if status_item.status == 3:  # GoalStatus.SUCCEEDED
            if not goal_status_succeeded:
                goal_status_succeeded = True
                processed_goal = False
                rospy.loginfo("Goal status succeeded!")
            return

    goal_status_succeeded = False

def user_command_callback(msg):
    global user_command, processed_goal
    user_command = msg.data
    print(f"接收到用户指令: {user_command}")

def image_callback(msg):
    global last_image
    # 将ROS图像消息转换为OpenCV格式
    last_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

def save_image(image, path):
    if image is not None:
        cv2.imwrite(path, image)
        print(f"图像已保存到: {path}")
    else:
        print("未收到图像，无法保存")

def check_goal_status(pub):
    global goal_status_succeeded, goal_status_time, processed_goal, last_image

    while not rospy.is_shutdown():
        # if goal_status_succeeded and (time.time() - goal_status_time) > 1:
        if goal_status_succeeded and not processed_goal:
            if "看" in user_command or "检查" in user_command:
                print("GoalStatus 为 SUCCEEDED，保存当前图像并开始图像处理工作流...")
                image_path = "/home/nuc1003a/ARTS_TEST/src/hdl_localization/images/current_image.jpg"
                save_image(last_image, image_path)
                image_processing_workflow(pub)
            else:
                print("GoalStatus 为 SUCCEEDED，但未检测到用户指令中包含'看'或'检查'，不执行图像处理工作流。")

            processed_goal = True
        time.sleep(CHECK_INTERVAL)

def main():
    rospy.init_node("image_understand_node", anonymous=True)
    rospy.Subscriber("move_base/status", GoalStatusArray, goal_status_callback)
    rospy.Subscriber("user_commands", String, user_command_callback)
    rospy.Subscriber("/camera/color/image_raw", Image, image_callback)  # 订阅图像话题
    pub = rospy.Publisher("response", String, queue_size=10, latch=True)

    status_check_thread = threading.Thread(target=check_goal_status, args=(pub,))
    status_check_thread.daemon = True
    status_check_thread.start()

    rospy.spin()

if __name__ == '__main__':
    main()