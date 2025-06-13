#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
嵌入模型管理模块
负责文本向量化和相似度计算
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
from typing import Dict
from transformers import AutoModel, AutoTokenizer
from sentence_transformers.util import cos_sim


class EmbeddingModel:
    """嵌入模型类"""
    
    def __init__(self, model_id: str):
        """
        初始化嵌入模型
        
        Args:
            model_id: 模型ID
        """
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        
        # 如果有GPU可用，使用GPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'
    
    def transform_query(self, query: str) -> str:
        """
        转换查询语句以适应嵌入模型
        
        Args:
            query: 原始查询
            
        Returns:
            str: 转换后的查询
        """
        return f"Represent this sentence for searching relevant passages: {query}"
    
    def pooling(self, outputs: torch.Tensor, inputs: Dict, 
                strategy: str = 'cls') -> np.ndarray:
        """
        池化操作
        
        Args:
            outputs: 模型输出
            inputs: 模型输入
            strategy: 池化策略 ('cls' or 'mean')
            
        Returns:
            np.ndarray: 池化后的向量
        """
        if strategy == 'cls':
            outputs = outputs[:, 0]
        elif strategy == 'mean':
            outputs = torch.sum(
                outputs * inputs["attention_mask"][:, :, None], dim=1
            ) / torch.sum(inputs["attention_mask"], dim=1, keepdim=True)
        else:
            raise NotImplementedError(f"Unknown pooling strategy: {strategy}")
        
        return outputs.detach().cpu().numpy()
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        编码查询文本
        
        Args:
            query: 查询文本
            
        Returns:
            np.ndarray: 查询向量
        """
        # 转换查询
        query_prompted = self.transform_query(query)
        
        # 编码
        inputs = self.tokenizer(
            query_prompted, 
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state
        
        # 池化
        query_embedding = self.pooling(outputs, inputs, 'cls')
        
        return query_embedding
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          doc_embedding: np.ndarray) -> float:
        """
        计算相似度
        
        Args:
            query_embedding: 查询向量
            doc_embedding: 文档向量
            
        Returns:
            float: 余弦相似度
        """
        query_tensor = torch.from_numpy(query_embedding.astype(np.float32))
        doc_tensor = torch.from_numpy(doc_embedding.astype(np.float32)).unsqueeze(0)
        
        similarity = cos_sim(query_tensor, doc_tensor)
        
        return similarity.item()