import os
import json
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import torch
from PIL import Image
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sentence_transformers import SentenceTransformer

class ImageProcessor:
    def __init__(self, model_name: str = 'Qwen/Qwen2.5-VL-7B-Instruct'):
        """
        初始化图像处理器
        
        Args:
            model_name: 使用的本地模型名称
        """
        self.model_name = model_name
        
        # 初始化视觉语言模型
        print("正在加载视觉语言模型...")
        self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype="auto", 
            device_map="auto"
        )
        self.vlm_processor = AutoProcessor.from_pretrained(model_name)
        print("视觉语言模型加载完成")
        
        # 初始化文本嵌入模型
        print("正在加载文本嵌入模型...")
        self.embedding_model = SentenceTransformer(
            "mixedbread-ai/mxbai-embed-large-v1", 
            truncate_dim=1024  # 设置为1024维
        )
        print("文本嵌入模型加载完成")
        
    def extract_metadata_and_caption_batch(self, image_paths: List[str], batch_size: int = 4) -> List[Optional[Dict]]:
        """
        批量处理多张图像，同时提取图像描述和右上角的元数据信息
        
        Args:
            image_paths: 图像文件路径列表
            batch_size: 批处理大小
            
        Returns:
            包含描述、时间和位置信息的字典列表
        """
        all_results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            print(f"处理批次 {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}, 包含 {len(batch_paths)} 张图像")
            
            try:
                # 构建批量消息格式
                messages = []
                for path in batch_paths:
                    message = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": f"file://{os.path.abspath(path)}"},
                                {"type": "text", "text": '''Please analyze this image and provide the following information in JSON format:

1. First, carefully look at the TOP RIGHT corner of the image where there should be text showing:
   - Time information in format: YYYY/MM/DD/HH/MM/SS
   - Position information showing coordinates (x, y, orientation)
2. Then, describe what you see in the main content of the image (objects, people, activities, environment) from the first point of view".

Please return your response in this exact JSON format:
{
    "caption": "description of the main image content",
    "time": "YYYY/MM/DD/HH/MM/SS",
    "position": [x, y, orientation]
}

Focus carefully on reading the text in the top right corner for accurate time and position data.'''},
                            ],
                        }
                    ]
                    messages.append(message)
                
                # 准备批量输入
                texts = [
                    self.vlm_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                    for msg in messages
                ]
                
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.vlm_processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                
                # 移动到GPU（如果可用）
                if torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                
                # 批量生成响应
                with torch.no_grad():
                    generated_ids = self.vlm_model.generate(
                        **inputs, 
                        max_new_tokens=512,
                        do_sample=False,
                        temperature=0.1
                    )
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    response_texts = self.vlm_processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                
                # 解析每个响应
                batch_results = []
                for j, (response_text, image_path) in enumerate(zip(response_texts, batch_paths)):
                    print(f"  图像 {os.path.basename(image_path)} 响应: {response_text[:100]}...")
                    
                    result = self.parse_response(response_text, image_path)
                    batch_results.append(result)
                
                all_results.extend(batch_results)
                
            except Exception as e:
                print(f"批处理失败 (批次 {i//batch_size + 1}): {e}")
                # 为失败的批次添加None结果
                batch_results = [None] * len(batch_paths)
                all_results.extend(batch_results)
            
            # 添加短暂延迟避免GPU过载
            time.sleep(0.2)
        
        return all_results
    
    def parse_response(self, response_text: str, image_path: str) -> Optional[Dict]:
        """
        解析模型响应文本，提取JSON信息
        
        Args:
            response_text: 模型响应文本
            image_path: 图像路径（用于后备处理）
            
        Returns:
            解析后的字典或None
        """
        try:
            # 先尝试提取markdown代码块中的JSON
            json_code_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_code_match:
                json_str = json_code_match.group(1)
            else:
                # 如果没有代码块，直接查找JSON
                json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                else:
                    print(f"  无法从响应中提取JSON，尝试生成简单描述")
                    # 生成简单描述作为后备
                    simple_caption = self.extract_simple_description(response_text)
                    if simple_caption:
                        return {
                            "caption": simple_caption,
                            "time": "2025/1/1/0:0:0",
                            "position": [0.0, 0.0, 0.0]
                        }
                    return None
            
            # 解析JSON
            result = json.loads(json_str)
            
            # 验证必需字段
            if 'caption' in result and 'time' in result and 'position' in result:
                return result
            else:
                print(f"  JSON缺少必需字段: {result}")
                return None
                
        except json.JSONDecodeError as e:
            print(f"  JSON解析失败: {e}")
            # 尝试从响应中提取简单描述
            simple_caption = self.extract_simple_description(response_text)
            if simple_caption:
                return {
                    "caption": simple_caption,
                    "time": "2025/1/1/0:0:0",
                    "position": [0.0, 0.0, 0.0]
                }
            return None
        except Exception as e:
            print(f"  处理响应时出错: {e}")
            return None
    
    def extract_simple_description(self, response_text: str) -> Optional[str]:
        """
        从响应文本中提取简单描述（作为后备方案）
        
        Args:
            response_text: 响应文本
            
        Returns:
            简单描述或None
        """
        try:
            # 尝试提取第一句话作为描述
            sentences = response_text.split('.')
            if sentences:
                first_sentence = sentences[0].strip()
                if len(first_sentence) > 10:  # 确保描述有一定长度
                    return first_sentence
            
            # 如果第一句话太短，返回前100个字符
            if len(response_text) > 10:
                return response_text[:100].strip()
                
            return None
        except:
            return None
    
    def generate_simple_caption(self, image_path: str) -> Optional[str]:
        """
        生成简单的图像描述（作为后备方案）
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            简单的图像描述，如果失败返回None
        """
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                        {"type": "text", "text": "Describe what you see in this image in one concise sentence. Focus on the main objects, people, and activities."},
                    ],
                }
            ]
            
            # 准备输入
            text = self.vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info([messages])
            inputs = self.vlm_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            # 生成简单描述
            with torch.no_grad():
                generated_ids = self.vlm_model.generate(
                    **inputs, 
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=0.1
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                caption = self.vlm_processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            
            return caption.strip()
            
        except Exception as e:
            print(f"生成简单描述失败: {e}")
            return None
    
    def generate_embeddings_batch(self, captions: List[str]) -> List[Optional[List[float]]]:
        """
        批量生成文本嵌入向量
        
        Args:
            captions: 图像描述文本列表
            
        Returns:
            1024维嵌入向量列表的列表
        """
        try:
            print(f"批量生成 {len(captions)} 个嵌入向量")
            
            # 使用sentence_transformers批量生成嵌入向量
            embeddings = self.embedding_model.encode(captions)
            
            # 转换为列表格式
            embedding_lists = []
            for embedding in embeddings:
                embedding_list = embedding.tolist()
                
                # 验证向量维度
                if len(embedding_list) == 1024:
                    embedding_lists.append(embedding_list)
                else:
                    print(f"嵌入向量维度异常: {len(embedding_list)}, 期望1024维")
                    embedding_lists.append(None)
            
            print(f"成功生成 {sum(1 for e in embedding_lists if e is not None)} 个有效的1024维嵌入向量")
            return embedding_lists
                
        except Exception as e:
            print(f"批量生成嵌入失败: {e}")
            return [None] * len(captions)
    
    def process_batch_images(self, image_paths: List[str]) -> List[Optional[Dict]]:
        """
        批量处理多张图像，生成完整的记录
        
        Args:
            image_paths: 图像文件路径列表
            
        Returns:
            包含嵌入、描述、时间和位置的记录字典列表
        """
        print(f"批量处理 {len(image_paths)} 张图像")
        
        # 使用视觉语言模型批量提取描述和元数据
        results = self.extract_metadata_and_caption_batch(image_paths)
        
        # 初始化records列表，长度与image_paths相同
        records = [None] * len(image_paths)
        
        # 收集有效的结果
        valid_results = []
        valid_indices = []
        
        for i, result in enumerate(results):
            if result and result.get('caption'):
                valid_results.append(result)
                valid_indices.append(i)
        
        if valid_results:
            # 批量生成嵌入向量
            captions = [result['caption'] for result in valid_results]
            embeddings = self.generate_embeddings_batch(captions)
            
            # 将结果填充到records中
            for j, (result, embedding) in enumerate(zip(valid_results, embeddings)):
                if embedding is not None:
                    original_index = valid_indices[j]
                    records[original_index] = {
                        'embedding': embedding,
                        'caption': result['caption'],
                        'time': result['time'],
                        'position': result['position']
                    }
        
        return records
    
    def process_images_batch(self, images_folder: str, output_file: str, 
                           start_id: int = 0, end_id: Optional[int] = None,
                           batch_size: int = 8) -> List[Dict]:
        """
        批量处理图像文件夹中的图像
        
        Args:
            images_folder: 图像文件夹路径
            output_file: 输出JSON文件路径
            start_id: 开始处理的图像ID
            end_id: 结束处理的图像ID（如果为None则处理所有图像）
            batch_size: 批处理大小
            
        Returns:
            所有成功处理的记录列表
        """
        all_records = []
        
        # 获取所有图像文件
        image_files = []
        for filename in os.listdir(images_folder):
            if filename.endswith('.jpg') and filename.replace('.jpg', '').isdigit():
                image_id = int(filename.replace('.jpg', ''))
                if image_id >= start_id and (end_id is None or image_id <= end_id):
                    image_files.append((image_id, filename))
        
        # 按ID排序
        image_files.sort(key=lambda x: x[0])
        
        print(f"找到 {len(image_files)} 张图像需要处理，批处理大小: {batch_size}")
        
        # 批量处理图像
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            batch_paths = [os.path.join(images_folder, filename) for _, filename in batch_files]
            
            print(f"\n处理批次 {i//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size}")
            print(f"批次文件: {[filename for _, filename in batch_files]}")
            
            # 批量处理当前批次的图像
            batch_records = self.process_batch_images(batch_paths)
            
            # 统计成功处理的图像并添加到总记录中
            successful_count = 0
            for j, record in enumerate(batch_records):
                if record is not None:
                    all_records.append(record)
                    successful_count += 1
                    print(f"  ✓ {batch_files[j][1]} 处理成功")
                else:
                    print(f"  ✗ {batch_files[j][1]} 处理失败")
            
            print(f"批次处理完成: {successful_count}/{len(batch_records)} 张图像成功处理")
        
        # 保存结果
        self.save_records(all_records, output_file)
        
        print(f"\n批量处理完成！共处理 {len(all_records)} 张图像")
        print(f"结果已保存到: {output_file}")
        
        return all_records
    
    def save_records(self, records: List[Dict], output_file: str):
        """
        保存记录到JSON文件
        
        Args:
            records: 记录列表
            output_file: 输出文件路径
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存文件失败: {e}")

def main():
    # 配置参数
    MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"  # 本地模型名称
    IMAGES_FOLDER = "../data/images" 
    OUTPUT_FILE = "../data/vector_database.json"
    
    # 创建处理器
    processor = ImageProcessor(MODEL_NAME)
    
    # 批量处理图像
    try:
        records = processor.process_images_batch(
            images_folder=IMAGES_FOLDER,
            output_file=OUTPUT_FILE,
            start_id=0,  # 从000000.jpg开始
            end_id=None,  # 处理所有图像，也可以指定结束ID
            batch_size=8  # 可以根据GPU内存调整批处理大小
        )
        
        print(f"\n处理完成！生成了 {len(records)} 条记录")
        
        # 显示第一条记录作为示例（不显示完整的embedding向量）
        if records:
            print("\n第一条记录示例:")
            sample_record = records[0].copy()
            sample_record['embedding'] = f"[{len(records[0]['embedding'])}维向量]"
            print(json.dumps(sample_record, ensure_ascii=False, indent=2))
            
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

if __name__ == "__main__":
    main()