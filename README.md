# VLN_TEST
## 介绍
基于长时记忆的语义导航，支持多种问答模式，多点导航。
## 安装
测试环境：Ubuntu20.04，ROS Noetic，RTX4090 24GB

创建虚拟环境：

`conda create -n vln python=3.10`

安装依赖：
```bash
pip install transformers==4.51.3 accelerate
pip install qwen-vl-utils[decord]
pip install modelscope
pip install dashscope
python -m pip install -U sentence-transformers
pip install openai==0.28
```

## 使用
### 1. 录制rosbag
包含里程计topic和图像topic。测试中使用fastlio2作为里程计数据来源。
### 2. 构建memory
处理rosbag
```bash
cd memory/src
# 对rosbag解包
python rosbag_extractor.py your_bag.bag --image_topic /your_image_topic --odom_topic /your_odom_topic
```
处理完成后，会在data/images文件夹看到一系列图片。


