#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rosbag
import cv2
import numpy as np
from cv_bridge import CvBridge
import os
from collections import defaultdict
import argparse
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
import rospy
from datetime import datetime
import tf.transformations as tf_trans

class RosbagImageExtractor:
    def __init__(self, bag_path, image_topic, odom_topic, output_dir="output_images", sample_interval=1.0):
        """
        初始化提取器
        
        Args:
            bag_path: rosbag文件路径
            image_topic: 图像话题名称
            odom_topic: 里程计话题名称
            output_dir: 输出目录
            sample_interval: 采样间隔（秒）
        """
        self.bag_path = bag_path
        self.image_topic = image_topic
        self.odom_topic = odom_topic
        self.output_dir = output_dir
        self.sample_interval = sample_interval
        self.bridge = CvBridge()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 存储数据
        self.images = []  # [(timestamp, image_msg)]
        self.odometry = []  # [(timestamp, x, y, yaw)]
        
    def extract_data_from_bag(self):
        """从rosbag中提取图像和里程计数据"""
        print(f"正在读取rosbag: {self.bag_path}")
        
        with rosbag.Bag(self.bag_path, 'r') as bag:
            # 获取bag信息
            info = bag.get_type_and_topic_info()
            topics = info.topics
            
            print(f"可用话题:")
            for topic_name, topic_info in topics.items():
                print(f"  {topic_name}: {topic_info.msg_type}")
            
            # 提取图像数据
            print(f"\n正在提取图像数据: {self.image_topic}")
            for topic, msg, t in bag.read_messages(topics=[self.image_topic]):
                timestamp = t.to_sec()
                self.images.append((timestamp, msg))
            
            # 提取里程计数据
            print(f"正在提取里程计数据: {self.odom_topic}")
            for topic, msg, t in bag.read_messages(topics=[self.odom_topic]):
                timestamp = t.to_sec()
                
                # 根据消息类型提取位置信息和姿态信息
                if hasattr(msg, 'pose'):
                    if hasattr(msg.pose, 'pose'):  # Odometry消息
                        position = msg.pose.pose.position
                        orientation = msg.pose.pose.orientation
                    else:  # PoseStamped消息
                        position = msg.pose.position
                        orientation = msg.pose.orientation
                elif hasattr(msg, 'position'):  # 直接包含position
                    position = msg.position
                    orientation = getattr(msg, 'orientation', None)
                else:
                    print(f"警告: 无法从里程计消息中提取位置信息")
                    continue
                    
                x, y = position.x, position.y
                
                # 计算yaw角（从四元数转换）
                if orientation:
                    quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
                    euler = tf_trans.euler_from_quaternion(quaternion)
                    yaw = euler[2]  # yaw是绕z轴的旋转
                else:
                    yaw = 0.0
                    
                self.odometry.append((timestamp, x, y, yaw))
        
        print(f"提取完成:")
        print(f"  图像数量: {len(self.images)}")
        print(f"  里程计数据点: {len(self.odometry)}")
    
    def find_closest_odom(self, target_time):
        """找到最接近指定时间的里程计数据"""
        if not self.odometry:
            return None
        
        min_diff = float('inf')
        closest_odom = None
        
        for timestamp, x, y, yaw in self.odometry:
            diff = abs(timestamp - target_time)
            if diff < min_diff:
                min_diff = diff
                closest_odom = (x, y, yaw)
        
        return closest_odom
    
    def add_subtitle_to_image(self, image, timestamp, position):
        """在图像右上角添加字幕"""
        # 转换ROS图像消息为OpenCV格式
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        except Exception as e:
            print(f"图像转换失败: {e}")
            return None
        
        # 将时间戳转换为日期时间格式
        dt = datetime.fromtimestamp(timestamp)
        time_str = dt.strftime("Time: %Y/%m/%d %H:%M:%S")
        
        # 准备位置信息文本
        if position:
            x, y, yaw = position
            pos_str = f"Pos: ({x:.3f}, {y:.3f}, {yaw:.3f})"
        else:
            pos_str = "Pos: N/A"
        
        # 设置字体参数
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)  # 白色
        thickness = 2
        bg_color = (0, 0, 0)  # 黑色背景
        
        # 获取图像尺寸
        height, width = cv_image.shape[:2]
        
        # 计算文本尺寸
        (time_w, time_h), _ = cv2.getTextSize(time_str, font, font_scale, thickness)
        (pos_w, pos_h), _ = cv2.getTextSize(pos_str, font, font_scale, thickness)
        
        # 设置文本位置（右上角）
        margin = 20
        time_x = width - time_w - margin
        time_y = margin + time_h
        pos_x = width - pos_w - margin
        pos_y = time_y + pos_h + 5
        
        # 绘制背景矩形
        # 扩大背景区域的边界
        bg_margin_x = 20
        bg_margin_y = 20
        top_left = (min(time_x, pos_x) - bg_margin_x, time_y - time_h - bg_margin_y)
        bottom_right = (width - margin + bg_margin_x, pos_y + bg_margin_y)

        cv2.rectangle(cv_image, top_left, bottom_right, bg_color, -1)

        
        # 绘制文本
        cv2.putText(cv_image, time_str, (time_x, time_y), font, font_scale, color, thickness)
        cv2.putText(cv_image, pos_str, (pos_x, pos_y), font, font_scale, color, thickness)
        
        return cv_image
    
    def process_and_save_images(self):
        """处理图像并保存"""
        if not self.images:
            print("没有找到图像数据")
            return
        
        print(f"\n开始处理图像，采样间隔: {self.sample_interval}秒")
        
        # 获取起始时间
        start_time = self.images[0][0]
        
        # 按采样间隔处理图像
        image_count = 0
        next_sample_time = start_time
        
        for timestamp, image_msg in self.images:
            # 检查是否到达下一个采样点
            if timestamp >= next_sample_time:
                # 找到最接近的里程计数据
                position = self.find_closest_odom(timestamp)
                
                # 添加字幕
                processed_image = self.add_subtitle_to_image(image_msg, timestamp, position)
                
                if processed_image is not None:
                    # 保存图像
                    filename = f"{image_count:06d}.jpg"
                    filepath = os.path.join(self.output_dir, filename)
                    cv2.imwrite(filepath, processed_image)
                    
                    print(f"保存图像 {image_count}: {filename}")
                    image_count += 1
                
                # 更新下一个采样时间
                next_sample_time += self.sample_interval
        
        print(f"\n处理完成，共保存 {image_count} 张图像到 {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='从ROS bag中提取图像和里程计数据并添加字幕')
    parser.add_argument('bag_path', help='ROS bag文件路径')
    parser.add_argument('--image_topic', required=True, help='图像话题名称')
    parser.add_argument('--odom_topic', required=True, help='里程计话题名称')
    parser.add_argument('--output_dir', default='../data/images', help='输出目录 (默认: output_images)')
    parser.add_argument('--interval', type=float, default=1.0, help='采样间隔秒数 (默认: 1.0)')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.bag_path):
        print(f"错误: 找不到文件 {args.bag_path}")
        return
    
    # 创建提取器并运行
    extractor = RosbagImageExtractor(
        bag_path=args.bag_path,
        image_topic=args.image_topic,
        odom_topic=args.odom_topic,
        output_dir=args.output_dir,
        sample_interval=args.interval
    )
    
    try:
        extractor.extract_data_from_bag()
        extractor.process_and_save_images()
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 如果直接运行，可以在这里设置默认参数进行测试
    main()
    
    # 示例用法:
    # print("示例用法:")
    # print("python rosbag_extractor.py your_bag.bag --image_topic /camera/image_raw --odom_topic /odom")
    # print("python rosbag_extractor.py your_bag.bag --image_topic /camera/image_raw --odom_topic /odom --interval 0.5 --output_dir my_output")