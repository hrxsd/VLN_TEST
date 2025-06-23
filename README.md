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

