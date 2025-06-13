#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据加载工具模块
负责加载和验证向量数据库
"""

import json
import os
from typing import List, Dict


class DataLoader:
    """数据加载器类"""
    
    def __init__(self, database_path: str):
        """
        初始化数据加载器
        
        Args:
            database_path: 数据库文件路径
        """
        self.database_path = database_path
        
    def load_data(self) -> List[Dict]:
        """
        加载向量数据库
        
        Returns:
            List[Dict]: 数据库内容
        """
        if not os.path.exists(self.database_path):
            raise FileNotFoundError(
                f"Database file not found: {self.database_path}"
            )
        
        try:
            with open(self.database_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证数据格式
            self._validate_data(data)
            
            print(f"Successfully loaded {len(data)} entries from database")
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in database: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load database: {e}")
    
    def _validate_data(self, data: List[Dict]):
        """
        验证数据格式
        
        Args:
            data: 数据列表
        """
        if not isinstance(data, list):
            raise ValueError("Database must be a list of entries")
        
        required_fields = ['caption', 'time', 'position', 'embedding']
        
        for idx, entry in enumerate(data):
            if not isinstance(entry, dict):
                raise ValueError(f"Entry {idx} is not a dictionary")
            
            # 检查必需字段
            for field in required_fields:
                if field not in entry:
                    raise ValueError(
                        f"Entry {idx} missing required field: {field}"
                    )
            
            # 验证position格式
            if not isinstance(entry['position'], list) or len(entry['position']) != 3:
                raise ValueError(
                    f"Entry {idx} has invalid position format. "
                    f"Expected [x, y, w], got {entry['position']}"
                )
            
            # 验证embedding格式
            if not isinstance(entry['embedding'], list):
                raise ValueError(
                    f"Entry {idx} has invalid embedding format. "
                    f"Expected list of floats"
                )
    
    def save_data(self, data: List[Dict], output_path: str = None):
        """
        保存数据（用于更新或备份）
        
        Args:
            data: 数据列表
            output_path: 输出路径，默认覆盖原文件
        """
        if output_path is None:
            output_path = self.database_path
        
        # 创建备份
        if output_path == self.database_path and os.path.exists(self.database_path):
            backup_path = self.database_path + '.backup'
            os.rename(self.database_path, backup_path)
            print(f"Created backup at: {backup_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Successfully saved {len(data)} entries to {output_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save data: {e}")