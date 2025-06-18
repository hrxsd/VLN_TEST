#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检索引擎核心模块
负责执行文本、位置和时间检索
"""

import re
import math
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np

from core.embedding_model import EmbeddingModel


class RetrievalEngine:
    """检索引擎类"""
    
    def __init__(self, data: List[Dict], model_id: str):
        """
        初始化检索引擎
        
        Args:
            data: 向量数据库数据
            model_id: 嵌入模型ID
        """
        self.data = data
        self.embedding_model = EmbeddingModel(model_id)
        
    def execute_retrieval(self, function_call: str) -> List[Dict]:
        """
        执行函数调用
        
        Args:
            function_call: 函数调用字符串
            
        Returns:
            List[Dict]: 检索结果
        """
        try:
            if function_call.startswith("fl("):
                return self._parse_and_execute_fl(function_call)
            elif function_call.startswith("fp("):
                return self._parse_and_execute_fp(function_call)
            elif function_call.startswith("ft("):
                return self._parse_and_execute_ft(function_call)
            else:
                print(f"Unknown function call: {function_call}")
                return []
        except Exception as e:
            print(f"Error executing retrieval: {e}")
            return []
    
    def _parse_and_execute_fl(self, function_call: str) -> List[Dict]:
        """解析并执行文本检索"""
        match = re.search(r'fl\(["\']([^"\']+)["\']\)', function_call)
        if match:
            query = match.group(1)
            return self.text_lookup(query)
        return []
    
    def _parse_and_execute_fp(self, function_call: str) -> List[Dict]:
        """解析并执行位置检索"""
        match = re.search(r'fp\(\(([^)]+)\)\)', function_call)
        if match:
            coords = match.group(1).split(',')
            x, y, w = map(float, coords)
            return self.position_lookup((x, y, w))
        return []
    
    def _parse_and_execute_ft(self, function_call: str) -> List[Dict]:
        """解析并执行时间检索"""
        match = re.search(r'ft\(["\']([^"\']+)["\']\)', function_call)
        if match:
            time_str = match.group(1)
            return self.time_lookup(time_str)
        return []
    
    def text_lookup(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        文本语义检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: Top-k最相关的记忆片段
        """
        # 获取查询向量
        query_embedding = self.embedding_model.encode_query(query)
        
        # 计算相似度
        results = []
        for idx, item in enumerate(self.data):
            similarity = self.embedding_model.compute_similarity(
                query_embedding, 
                np.array(item['embedding'])
            )
            
            results.append({
                'index': idx,
                'caption': item['caption'],
                'time': item['time'],
                'position': item['position'],
                'similarity': float(similarity)
            })
        
        # 排序并返回Top-k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def position_lookup(self, target_position: Tuple[float, float, float], 
                       top_k: int = 5) -> List[Dict]:
        """
        空间位置检索
        
        Args:
            target_position: 目标位置 (x, y, w)
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: Top-k最近的记忆片段
        """
        x_target, y_target, _ = target_position
        
        results = []
        for idx, item in enumerate(self.data):
            x, y, _ = item['position']
            # 计算欧几里得距离
            distance = math.sqrt((x - x_target) ** 2 + (y - y_target) ** 2)
            
            results.append({
                'index': idx,
                'caption': item['caption'],
                'time': item['time'],
                'position': item['position'],
                'distance': distance
            })
        
        # 按距离排序
        results.sort(key=lambda x: x['distance'])
        return results[:top_k]
    
    def time_lookup(self, time_query: str, top_k: int = 5) -> List[Dict]:
        """
        时间检索
        
        Args:
            time_query: 目标时间字符串 "YYYY/MM/DD HH:MM:SS"
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: Top-k时间最接近的记忆片段
        """
        # 解析查询时间
        try:
            target_time = datetime.strptime(time_query, "%Y/%m/%d %H:%M:%S")
        except ValueError:
            print(f"无法解析查询时间: {time_query}")
            return []
        
        results = []
        for idx, item in enumerate(self.data):
            try:
                # 尝试两种时间格式
                item_time_str = item['time']
                
                # 格式1: "YYYY/MM/DD HH:MM:SS"
                try:
                    item_time = datetime.strptime(item_time_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    # 格式2: "YYYY/MM/DD/HH/MM/SS" 
                    try:
                        item_time = datetime.strptime(item_time_str, "%Y/%m/%d/%H/%M/%S")
                    except ValueError:
                        print(f"无法解析数据库时间: {item_time_str}")
                        continue
                
                # 计算时间差
                time_diff = abs((target_time - item_time).total_seconds())
                
                results.append({
                    'index': idx,
                    'caption': item['caption'],
                    'time': item['time'],
                    'position': item['position'],
                    'time_diff': time_diff
                })
                
            except Exception as e:
                print(f"处理索引 {idx} 时出错: {e}")
                continue
        
        # 按时间差排序
        results.sort(key=lambda x: x['time_diff'])
        return results[:top_k]