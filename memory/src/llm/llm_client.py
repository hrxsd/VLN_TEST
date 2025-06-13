#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM客户端模块
负责与OpenAI API交互，进行意图分析和答案生成
"""

import os
import json
import openai
from typing import Dict, List
from datetime import datetime


class LLMClient:
    """LLM客户端类"""
    
    def __init__(self, api_key: str, api_base: str, model: str = "gpt-4"):
        """
        初始化LLM客户端
        
        Args:
            api_key: API密钥
            api_base: API基础URL
            model: 模型名称
        """
        openai.api_key = api_key
        openai.api_base = api_base
        self.model = model
        
        # 加载prompts
        self._load_prompts()
    
    def _load_prompts(self):
        """加载prompt模板"""
        prompt_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'prompts'
        )
        
        # 加载意图分析prompt
        intent_path = os.path.join(prompt_dir, 'intent_analysis.txt')
        with open(intent_path, 'r', encoding='utf-8') as f:
            self.intent_prompt = f.read()
        
        # 加载答案生成prompt
        answer_path = os.path.join(prompt_dir, 'answer_generation.txt')
        with open(answer_path, 'r', encoding='utf-8') as f:
            self.answer_prompt = f.read()
    
    def analyze_intent(self, user_input: str) -> str:
        """
        分析用户意图，生成检索函数调用
        
        Args:
            user_input: 用户输入
            
        Returns:
            str: 函数调用字符串
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.intent_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"LLM intent analysis error: {e}")
            # 默认返回文本检索
            return f'fl("{user_input}")'
    
    def generate_answer(self, user_input: str, 
                       retrieved_data: List[Dict]) -> Dict:
        """
        基于检索结果生成最终答案
        
        Args:
            user_input: 用户输入
            retrieved_data: 检索到的数据
            
        Returns:
            Dict: 答案JSON对象
        """
        # 构造上下文
        context = self._build_context(retrieved_data)
        
        # 构造用户消息
        user_message = f"用户问题：{user_input}\n\n{context}"
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.answer_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            answer_text = response.choices[0].message.content.strip()
            
            # 解析JSON响应
            try:
                return json.loads(answer_text)
            except json.JSONDecodeError:
                # 如果解析失败，返回默认格式
                return {
                    "text": answer_text,
                    "time": datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                }
                
        except Exception as e:
            print(f"LLM answer generation error: {e}")
            return {
                "text": "抱歉，我无法处理您的请求。",
                "time": datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            }
    
    def _build_context(self, retrieved_data: List[Dict]) -> str:
        """
        构建检索结果的上下文
        
        Args:
            retrieved_data: 检索结果
            
        Returns:
            str: 格式化的上下文
        """
        if not retrieved_data:
            return "未找到相关的记忆片段。"
        
        context = "检索到的相关记忆片段：\n"
        for i, item in enumerate(retrieved_data):
            context += f"\n记忆{i+1}:\n"
            context += f"- 描述: {item['caption']}\n"
            context += f"- 时间: {item['time']}\n"
            context += f"- 位置: {item['position']}\n"
            
            # 添加额外信息（如果有）
            if 'similarity' in item:
                context += f"- 相关度: {item['similarity']:.3f}\n"
            elif 'distance' in item:
                context += f"- 距离: {item['distance']:.2f}米\n"
            elif 'time_diff' in item:
                context += f"- 时间差: {item['time_diff']:.0f}秒\n"
        
        return context