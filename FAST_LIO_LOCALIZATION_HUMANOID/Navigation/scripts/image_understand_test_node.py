#!/usr/bin/env python3
# coding: utf-8

import rospy
import threading
import math
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseActionGoal
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from openai import OpenAI
import cv2
import json
import base64
import urllib
import requests
import time
import tf

# 全局变量
goal_pose = None
current_pose_map_frame = None
bridge = CvBridge()
last_image = None
tf_listener = None

# 误差阈值
POSITION_TOLERANCE = 0.4  # 位置误差阈值 (米)
ANGLE_TOLERANCE = 0.3  # 角度误差阈值 (弧度)

# API 密钥和路径
API_KEY = "w1tpR06Fq5yhnFGCYFl88eA2"
SECRET_KEY = "fWsqb2A0GO1ngJaB5xSFqXuw6tCvz80I"
IMAGE_PATH = "/home/nuc1003a/ARTS_test/src/hdl_localization/images/current_image.jpg"
QUESTION = "你是一台机器人,这张图像你当前看到的内容,请结合user prompt对图像内容进行十分简洁的回答(一句话),不需要描述情绪等抽象内容,以第一视角回答"
TASK_ID_FILE = "task_id.txt"
RETRY_INTERVAL = 10  # 秒
CHECK_INTERVAL = 0.1  # 检查 GoalStatus 间隔

GPT4_API_KEY = "sk-iP0rxrsx3P8WevFlAe14F026043c4fC3B33b19E6DfE2Ae7e"
GPT4_API_BASE = "https://api.gpt.ge/v1"

user_command = ""  # 存储用户命令的全局变量

def pose_distance(pose1, pose2):
    """计算两个位姿之间的欧氏距离和角度差"""
    position_dist = math.sqrt(
        (pose1.position.x - pose2.position.x) ** 2 +
        (pose1.position.y - pose2.position.y) ** 2
    )
    
    # 提取四元数中的偏航角（绕Z轴旋转）
    yaw1 = 2 * math.atan2(pose1.orientation.z, pose1.orientation.w)
    yaw2 = 2 * math.atan2(pose2.orientation.z, pose2.orientation.w)
    angle_dist = abs(yaw1 - yaw2)

    rospy.loginfo(f"位置差: {position_dist}")
    rospy.loginfo(f"角度差: {angle_dist}")
    
    return position_dist, angle_dist

def goal_callback(msg):
    global goal_pose
    goal_pose = msg.goal.target_pose.pose  # 从MoveBaseActionGoal中提取目标位姿
    rospy.loginfo("目标点已更新")

def robot_pose_callback(msg):
    global current_pose_map_frame
    try:
        # 获取 Pose 和其对应的 frame_id
        pose_odom = msg.pose.pose
        header = msg.header

        # 创建 PoseStamped 消息并赋予 header
        pose_stamped = PoseStamped()
        pose_stamped.pose = pose_odom
        pose_stamped.header = header

        # 使用最近的时间戳进行转换
        pose_stamped.header.stamp = rospy.Time(0)

        # 将 Pose 转换到 map 坐标系下
        current_pose_map_frame = tf_listener.transformPose("map", pose_stamped).pose
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        rospy.logerr(f"转换为map坐标系失败: {e}")

def image_callback(msg):
    global last_image
    last_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

def save_image(image, path):
    if image is not None:
        cv2.imwrite(path, image)
        rospy.loginfo(f"图像已保存到: {path}")
    else:
        rospy.loginfo("未收到图像，无法保存")

def check_goal_reached(pub):
    global goal_pose, current_pose_map_frame, last_image, user_command
    
    while not rospy.is_shutdown():
        if goal_pose is not None and current_pose_map_frame is not None:
            position_dist, angle_dist = pose_distance(goal_pose, current_pose_map_frame)
            
            if position_dist < POSITION_TOLERANCE and angle_dist < ANGLE_TOLERANCE:
                rospy.loginfo("机器人已到达目标点，开始图像保存和处理流程...")
                image_path = IMAGE_PATH
                save_image(last_image, image_path)
                
                # 检查用户指令是否包含 "看" 或 "检查"
                if "看" in user_command or "检查" in user_command:
                    image_processing_workflow(pub)
                else:
                    rospy.loginfo("未检测到用户指令中包含 '看' 或 '检查'，不执行图像处理工作流。")

                goal_pose = None  # 防止重复处理相同目标
        rospy.sleep(0.1)

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
            description_to_llm = response_data.get("result", {}).get("description")
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
            # gpt_response = get_chatgpt_response(description_to_llm)
            gpt_response = description_to_llm
            print("WALLE:", gpt_response)
            pub.publish(gpt_response)
        else:
            print("未能获取description_to_llm")
    else:
        print("请求图像理解失败。")

def user_command_callback(msg):
    global user_command
    user_command = msg.data
    rospy.loginfo(f"接收到用户指令: {user_command}")

def main():
    global tf_listener

    time.sleep(5)

    rospy.init_node("image_understand_node", anonymous=True)
    tf_listener = tf.TransformListener()
    rospy.Subscriber("/move_base/goal", MoveBaseActionGoal, goal_callback)
    rospy.Subscriber("/robot_pose_ekf/odom_combined", PoseWithCovarianceStamped, robot_pose_callback)
    rospy.Subscriber("/camera/color/image_raw", Image, image_callback)  # 订阅图像话题
    rospy.Subscriber("user_commands", String, user_command_callback)  # 订阅用户指令话题
    pub = rospy.Publisher("response", String, queue_size=10, latch=True)

    goal_check_thread = threading.Thread(target=check_goal_reached, args=(pub,))
    goal_check_thread.daemon = True
    goal_check_thread.start()

    rospy.spin()

if __name__ == '__main__':
    main()

    