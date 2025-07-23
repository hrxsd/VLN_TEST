# VLN_TEST
## 介绍
本项目实现了基于长期记忆的语义导航，支持多种问答模式和多点导航功能。
## 安装
### memory相关
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
### 建图 & 导航相关
建图和定位方案采用开源方案FAST_LIO_LOCALIZATION_HUMANOID，专用于人形机器人的3D SLAM定位系统，针对宇树G1在头部向下的雷达排布做了适配，可支持ROS原生的定位初始化。具体配置可以参考https://mn2ehoz71j.feishu.cn/docx/LTRRdTXikomDoSxWmqucRFMinrh?from=from_copylink

注意将FAST_LIO_LOCALIZATION_HUMANOID文件夹放在单独的ros-workspace内。
### G1机器人控制相关
unitree_sdk_python依赖安装参考宇树官方GitHub：https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/README%20zh.md
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
需先启动机器人move-base导航系统：
```bash
# 进入机器人导航workspace
source devel/setup.bash
# 启动movebase导航服务
roslaunch navigation nav_test.launch
# 开启mid360
roslaunch livox_ros_driver2 msg_MID360.launch
```
启动memory功能脚本：
```bash
# 单目标点
python navigator_with_api.py config/config.yaml
# 多目标点（尚在测试）
python multi_step_navigation_ros.py
# 若只想测试LLM检索引擎，不需要实际导航
python multi_step_navigation_test.py
# 带语音功能
python navigator_with_api_audio.py
```
### 4. G1运控
开机 & 机器人初始化参考宇树官方文档操作说明：https://support.unitree.com/home/zh/G1_developer/Operational_guidance
```bash
cd VLN_TEST/g1_base_controller/loco
# 启动控制服务
./g1_base_control.sh start
# 关闭控制服务
./g1_base_control.sh stop
```
机器人运动过程中，按下遥控器select键为软急停。
