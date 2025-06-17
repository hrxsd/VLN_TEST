import dashscope
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from sentence_transformers import SentenceTransformer

class ImageProcessor:
    def __init__(self, api_key: str, model_name: str = 'qwen-vl-plus'):
        """
        初始化图像处理器
        
        Args:
            api_key: DashScope API密钥
            model_name: 使用的模型名称
        """
        dashscope.api_key = api_key
        self.model_name = model_name
        
        # 初始化文本嵌入模型
        print("正在加载文本嵌入模型...")
        self.embedding_model = SentenceTransformer(
            "mixedbread-ai/mxbai-embed-large-v1", 
            truncate_dim=1024  # 设置为1024维
        )
        print("文本嵌入模型加载完成")
        
    def extract_metadata_and_caption(self, image_path: str) -> Optional[Dict]:
        """
        使用视觉语言模型同时提取图像描述和右上角的元数据信息
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            包含描述、时间和位置信息的字典，如果提取失败返回None
        """
        try:
            messages = [{
                'role': 'user',
                'content': [
                    {
                        'image': image_path
                    },
                    {
                        'text': '''Please analyze this image and provide the following information in JSON format:

1. First, carefully look at the TOP RIGHT corner of the image where there should be text showing:
   - Time information in format: YYYY/MM/DD/HH/MM/SS
   - Position information showing coordinates (x, y, orientation)
2. Then, describe what you see in the main content of the image (objects, people, activities, environment) from the first point of view.

Please return your response in this exact JSON format:
{
    "caption": "description of the main image content",
    "time": "YYYY/MM/DD/HH/MM/SS",
    "position": [x, y, orientation]
}

Focus carefully on reading the text in the top right corner for accurate time and position data.'''
                    },
                ]
            }]
            
            response = dashscope.MultiModalConversation.call(
                model=self.model_name, 
                messages=messages
            )
            
            if response.status_code == 200:
                response_content = response.output.choices[0].message.content
                
                # 处理响应内容，可能是字符串或列表
                if isinstance(response_content, list):
                    # 如果是列表，提取第一个元素的text字段
                    if len(response_content) > 0 and 'text' in response_content[0]:
                        response_text = response_content[0]['text']
                    else:
                        print(f"列表响应格式异常: {response_content}")
                        return None
                elif isinstance(response_content, str):
                    response_text = response_content
                else:
                    print(f"未知的响应内容类型: {type(response_content)}")
                    print(f"响应内容: {response_content}")
                    return None
                
                print(f"收到响应: {response_text[:200]}...")  # 打印前200字符用于调试
                
                # 尝试解析JSON响应
                try:
                    # 先尝试提取markdown代码块中的JSON
                    json_code_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                    if json_code_match:
                        json_str = json_code_match.group(1)
                        print(f"从代码块提取的JSON: {json_str}")
                    else:
                        # 如果没有代码块，直接查找JSON
                        json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group()
                            print(f"直接提取的JSON: {json_str}")
                        else:
                            print(f"无法从响应中提取JSON: {response_text}")
                            return None
                    
                    # 解析JSON
                    result = json.loads(json_str)
                    
                    # 验证必需字段
                    if 'caption' in result and 'time' in result and 'position' in result:
                        print(f"成功解析JSON: caption={result['caption'][:50]}..., time={result['time']}, position={result['position']}")
                        return result
                    else:
                        print(f"JSON缺少必需字段: {result}")
                        return None
                        
                except json.JSONDecodeError as e:
                    print(f"JSON解析失败: {e}")
                    print(f"尝试解析的JSON字符串: {json_str if 'json_str' in locals() else 'N/A'}")
                    return None
                except Exception as e:
                    print(f"处理响应时出错: {e}")
                    return None
            else:
                print(f"API调用失败: {response}")
                return None
                
        except Exception as e:
            print(f"提取信息失败 {image_path}: {e}")
            return None
    

    
    def generate_embedding(self, caption: str, image_path: str) -> Optional[List[float]]:
        """
        使用mxbai-embed-large-v1模型生成文本嵌入向量
        
        Args:
            caption: 图像描述文本
            image_path: 图像文件路径（用于调试）
            
        Returns:
            1024维嵌入向量列表，如果生成失败返回None
        """
        try:
            # 使用sentence_transformers生成嵌入向量
            embedding = self.embedding_model.encode(caption)
            
            # 转换为列表格式
            embedding_list = embedding.tolist()
            
            # 验证向量维度
            if len(embedding_list) == 1024:
                print(f"成功生成1024维嵌入向量")
                return embedding_list
            else:
                print(f"嵌入向量维度异常: {len(embedding_list)}, 期望1024维")
                return None
                
        except Exception as e:
            print(f"生成嵌入失败 {image_path}: {e}")
            return None
    
    def process_single_image(self, image_path: str) -> Optional[Dict]:
        """
        处理单张图像，生成完整的记录
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            包含嵌入、描述、时间和位置的记录字典
        """
        print(f"处理图像: {image_path}")
        
        # 使用视觉语言模型同时提取描述和元数据
        result = self.extract_metadata_and_caption(image_path)
        if not result:
            return None
        
        # 生成嵌入
        embedding = self.generate_embedding(result['caption'], image_path)
        if not embedding:
            return None
        
        # 构建记录
        record = {
            'embedding': embedding,
            'caption': result['caption'],
            'time': result['time'],
            'position': result['position']
        }
        
        return record
    
    def process_images_batch(self, images_folder: str, output_file: str, 
                           start_id: int = 0, end_id: Optional[int] = None) -> List[Dict]:
        """
        批量处理图像文件夹中的图像
        
        Args:
            images_folder: 图像文件夹路径
            output_file: 输出JSON文件路径
            start_id: 开始处理的图像ID
            end_id: 结束处理的图像ID（如果为None则处理所有图像）
            
        Returns:
            所有成功处理的记录列表
        """
        records = []
        
        # 获取所有图像文件
        image_files = []
        for filename in os.listdir(images_folder):
            if filename.endswith('.jpg') and filename.replace('.jpg', '').isdigit():
                image_id = int(filename.replace('.jpg', ''))
                if image_id >= start_id and (end_id is None or image_id <= end_id):
                    image_files.append((image_id, filename))
        
        # 按ID排序
        image_files.sort(key=lambda x: x[0])
        
        print(f"找到 {len(image_files)} 张图像需要处理")
        
        # 处理每张图像
        for image_id, filename in image_files:
            image_path = os.path.join(images_folder, filename)
            
            # 处理图像
            record = self.process_single_image(image_path)
            
            if record:
                records.append(record)
                print(f"成功处理: {filename}")
            else:
                print(f"处理失败: {filename}")
            
            # 添加延迟避免API限制
            time.sleep(0.5)
        
        # 保存结果
        self.save_records(records, output_file)
        
        print(f"批量处理完成！共处理 {len(records)} 张图像")
        print(f"结果已保存到: {output_file}")
        
        return records
    
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
    API_KEY = "sk-ed74a228b720431981472a559c66b9b5"
    IMAGES_FOLDER = "../data/images" 
    OUTPUT_FILE = "../data/vector_database.json"
    
    # 创建处理器
    processor = ImageProcessor(API_KEY)
    
    # 批量处理图像
    try:
        records = processor.process_images_batch(
            images_folder=IMAGES_FOLDER,
            output_file=OUTPUT_FILE,
            start_id=0,  # 从000000.jpg开始
            end_id=None  # 处理所有图像，也可以指定结束ID
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