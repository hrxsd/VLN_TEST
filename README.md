# VLN_TEST
## 介绍
本项目实现了基于长期记忆的语义导航，支持多种问答模式和多点导航功能。
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
处理完成后，会在`data/images`文件夹看到一系列图片。

构建向量数据库。
```bash
# 如果使用Qwen视觉语言模型的API(API额度有限，不建议此方式)
python build_database_QwenAPI.py ../data/images
# 如果使用本地部署的QwenVL2.5-7B模型(对显卡及显存要求较高)
python build_database_QwenLocalModel.py ../data/images
```
该过程可能会持续一段时间，完成后可以在`memory/data`文件夹下看到`vector_database.json`文件
### 3. 导航
需启动机器人move-base导航系统。
```bash
# 单目标点
python navigator_with_api.py config/config.yaml
# 多目标点
python multi_step_navigation_ros.py
# 若只想测试LLM检索引擎，不需要实际导航
python multi_step_navigation_test.py
# 带语音功能
python navigator_with_api_audio.py
```

